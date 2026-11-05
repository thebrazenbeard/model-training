import unittest
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from trainingday.voice_protocol import CASES, VoiceProtocolError, main, screening_report, validate_transcripts


def batch(response_fn=None, arm="zero"):
    response_fn = response_fn or (lambda case: "I am Vera.")
    return [
        {"arm": arm, "case_id": case[0], "response": response_fn(case),
         "origin_claim": "unverified-local-llama", "base_sha256": "a" * 64,
         "adapter_sha256": "b" * 64}
        for case in CASES
    ]


class VoiceProtocolTests(unittest.TestCase):
    def test_canned_identity_does_not_pass_personality(self):
        out = screening_report(batch())
        arm = out["arms"]["zero"]
        self.assertEqual(arm["public_case_count"], 12)
        self.assertEqual(arm["exact_repeat_max"], 12)
        self.assertTrue(arm["parroting_warning"])
        self.assertEqual(arm["review_status"], "REVIEW_REQUIRED")
        self.assertFalse(out["semantic_effect_established"])

    def test_distinct_text_still_does_not_self_certify(self):
        out = screening_report(batch(lambda c: f"Reply to {c[0]}"))
        self.assertFalse(out["arms"]["zero"]["parroting_warning"])
        self.assertFalse(out["independent_review_performed"])
        self.assertFalse(out["heldout"])
        self.assertFalse(out["multi_turn_evaluated"])
        self.assertFalse(out["session_retention_evaluated"])

    def test_every_arm_must_cover_every_case(self):
        with self.assertRaisesRegex(VoiceProtocolError, "incomplete"):
            validate_transcripts(batch()[:-1])
        with self.assertRaisesRegex(VoiceProtocolError, "duplicate"):
            validate_transcripts(batch() + [batch()[0]])

    def test_cannot_change_claimed_weights_mid_arm(self):
        rows = batch()
        rows[1]["adapter_sha256"] = "c" * 64
        with self.assertRaisesRegex(VoiceProtocolError, "silently changed"):
            validate_transcripts(rows)

    def test_rejects_extra_fields_that_could_contain_private_data(self):
        rows = batch()
        rows[0]["private_prompt"] = "do not leak"
        with self.assertRaisesRegex(VoiceProtocolError, "schema"):
            validate_transcripts(rows)

    def test_wrong_hash_is_rejected(self):
        rows = batch()
        rows[2]["base_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(VoiceProtocolError, "commitment"):
            validate_transcripts(rows)

    def test_no_adapter_baseline_has_explicit_sentinel_not_fake_hash(self):
        rows = batch(arm="official_base")
        for row in rows:
            row["adapter_sha256"] = "NO_ADAPTER"
        out = screening_report(rows)
        self.assertEqual(out["arms"]["official_base"]["model_identity_level"],
                         "DECLARED_UNVERIFIED")
        self.assertEqual(out["arms"]["official_base"]["adapter_declaration"],
                         "NO_ADAPTER_CLAIMED")
        rows[1]["adapter_sha256"] = "b" * 64
        with self.assertRaisesRegex(VoiceProtocolError, "silently changed"):
            validate_transcripts(rows)

    def test_arm_id_cannot_be_used_as_path_or_terminal_control(self):
        for dangerous in ("../correct8", "base\ntrained", "../../private"):
            with self.subTest(dangerous=dangerous):
                rows = batch(arm=dangerous)
                with self.assertRaisesRegex(VoiceProtocolError, "arm"):
                    validate_transcripts(rows)

    def test_cli_emits_only_aggregate_not_private_response(self):
        rows = batch(lambda c: f"PRIVATE_CANARY_REPLY_{c[0]}")
        with TemporaryDirectory() as folder:
            input_path = Path(folder) / "responses.jsonl"
            input_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["--input", str(input_path)]), 0)
        emitted = stdout.getvalue()
        self.assertNotIn("PRIVATE_CANARY_REPLY", emitted)
        self.assertEqual(json.loads(emitted)["arms"]["zero"]["review_status"], "REVIEW_REQUIRED")

    def test_cli_rejects_ambiguous_duplicate_json_key_without_echo(self):
        with TemporaryDirectory() as folder:
            input_path = Path(folder) / "responses.jsonl"
            input_path.write_text('{"arm":"zero","arm":"secret"}\n', encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(main(["--input", str(input_path)]), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertNotIn("secret", err.getvalue())


if __name__ == "__main__":
    unittest.main()
