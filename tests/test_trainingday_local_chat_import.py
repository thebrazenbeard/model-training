"""All conversations here are invented schema fixtures, never live model output."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trainingday.local_chat_import import (
    LocalChatImportError,
    adapt_export_records,
    main,
    screen_local_exports,
)


def browser_fixture(arm="base", session="mock_session_a"):
    return {
        "schema": "VERA_DECLARED_LOCAL_CHAT_EXPORT_V1",
        "arm": arm,
        "case_id": "fictional_visible_history",
        "session_id": session,
        "source_claim": "mock_test",
        "model_selection_claim": "unverified-ui-choice",
        "base_sha256": "a" * 64,
        "adapter_sha256": "NO_ADAPTER",
        "prompt_sha256": "b" * 64,
        "generation": {
            "context_window_tokens": 4096,
            "max_reply_tokens": 64,
            "temperature_milli": 700,
            "rng_seed": 42,
        },
        "used_saved_recall": False,
        "restored_from_vault": False,
        "messages": [
            {"role": "user", "content": "Fictional Aster visits the garden."},
            {"role": "assistant", "content": "Synthetic response one."},
            {"role": "user", "content": "Correction: fictional Aster is Rowan; remember this visible turn."},
            {"role": "assistant", "content": "Synthetic response two."},
        ],
        "terminal_failure": None,
    }


class LocalChatImportTests(unittest.TestCase):
    def test_visible_correction_turns_are_preserved_without_semantic_scoring(self):
        converted = adapt_export_records([browser_fixture()])
        self.assertEqual(len(converted[0]["turns"]), 2)
        self.assertEqual(converted[0]["turns"][1]["user"],
                         "Correction: fictional Aster is Rowan; remember this visible turn.")
        self.assertEqual(converted[0]["turns"][0]["assistant"], "Synthetic response one.")
        result = screen_local_exports([browser_fixture()])
        self.assertFalse(result["model_identity_attested"])
        self.assertFalse(result["actual_multi_turn_conversation_attested"])
        self.assertFalse(result["semantic_personality_qualified"])
        self.assertFalse(result["human_review_performed"])
        self.assertTrue(result["local_export_import_format_validated"])

    def test_distinct_model_replies_are_allowed_with_identical_fictional_user_context(self):
        a = browser_fixture()
        b = browser_fixture("trained", "mock_session_b")
        b["adapter_sha256"] = "c" * 64
        b["messages"][3]["content"] = "I am Vera. This is only mock text."
        report = screen_local_exports([a, b])
        self.assertTrue(report["matched_user_context_validated"])
        self.assertTrue(report["matched_generation_settings_validated"])
        self.assertFalse(report["cross_arm_model_quality_compared"])

    def test_hidden_saved_recall_and_restoration_are_excluded(self):
        for flag in ("used_saved_recall", "restored_from_vault"):
            with self.subTest(flag=flag):
                obj = browser_fixture()
                obj[flag] = True
                with self.assertRaisesRegex(LocalChatImportError, "hidden saved context"):
                    adapt_export_records([obj])

    def test_browser_system_bubble_is_not_model_turn(self):
        obj = browser_fixture()
        obj["messages"][3]["role"] = "system"
        with self.assertRaisesRegex(LocalChatImportError, "alternating"):
            adapt_export_records([obj])

    def test_dangling_user_message_must_not_invent_assistant_reply(self):
        obj = browser_fixture()
        obj["messages"] = obj["messages"][:3]
        with self.assertRaisesRegex(LocalChatImportError, "complete pairs"):
            adapt_export_records([obj])

    def test_explicit_terminal_failure_is_separate_from_saved_messages(self):
        obj = browser_fixture()
        obj["messages"] = obj["messages"][:2]
        obj["terminal_failure"] = {
            "user": "Synthetic second attempted correction", "outcome": "truncated",
        }
        norm = adapt_export_records([obj])[0]
        self.assertIsNone(norm["turns"][1]["assistant"])
        self.assertEqual(norm["turns"][1]["outcome"], "truncated")
        report = screen_local_exports([obj])
        self.assertEqual(report["arms"]["base"]["technical_censoring"]["truncated"], 1)
        self.assertFalse(report["actual_multi_turn_conversation_attested"])

    def test_failure_cannot_embed_partial_model_text(self):
        obj = browser_fixture()
        obj["terminal_failure"] = {
            "user": "Synthetic failed retry", "outcome": "truncated", "assistant": "partial"}
        with self.assertRaisesRegex(LocalChatImportError, "terminal failure"):
            adapt_export_records([obj])

    def test_browser_message_window_is_bounded_to_actual_last_twelve(self):
        obj = browser_fixture()
        obj["messages"] = obj["messages"] * 4
        with self.assertRaisesRegex(LocalChatImportError, "message window"):
            adapt_export_records([obj])

    def test_missing_generation_parameters_must_not_be_guessed(self):
        obj = browser_fixture()
        obj.pop("generation")
        with self.assertRaisesRegex(LocalChatImportError, "schema"):
            adapt_export_records([obj])

    def test_extra_metadata_private_field_is_rejected(self):
        obj = browser_fixture()
        obj["private_history"] = "LEAK_CANARY"
        with self.assertRaisesRegex(LocalChatImportError, "schema"):
            adapt_export_records([obj])

    def test_cli_only_emits_aggregate_not_private_dialogue(self):
        obj = browser_fixture()
        obj["messages"][3]["content"] = "PRIVATE_CANARY_DIALOGUE_TEXT"
        with TemporaryDirectory() as folder:
            path = Path(folder) / "owner_local.jsonl"
            path.write_text(json.dumps(obj) + "\n", encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main(["--input", str(path)]), 0)
            self.assertNotIn("PRIVATE_CANARY", stdout.getvalue() + stderr.getvalue())
            self.assertTrue(json.loads(stdout.getvalue())["local_export_import_format_validated"])
            path.write_text('{"schema":"VERA_DECLARED_LOCAL_CHAT_EXPORT_V1","schema":"LEAK_CANARY"}\n',
                            encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main(["--input", str(path)]), 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotIn("LEAK_CANARY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
