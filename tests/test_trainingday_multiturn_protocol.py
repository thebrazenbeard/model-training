"""Synthetic format fixtures, never evidence of a real Vera model conversation."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trainingday.multiturn_protocol import (
    DialogueSessionError,
    main,
    screen_sessions,
    validate_sessions,
)


def specimen(arm="base", case="fictional_case_1", sid="session_base_1"):
    return {
        "arm": arm,
        "case_id": case,
        "session_id": sid,
        "source_claim": "mock_test",
        "origin_claim": "synthetic-test-only",
        "base_sha256": "a" * 64,
        "adapter_sha256": "NO_ADAPTER",
        "prompt_sha256": "b" * 64,
        "generation": {
            "context_window_tokens": 4096,
            "max_reply_tokens": 64,
            "temperature_milli": 700,
            "rng_seed": 42,
        },
        "turns": [
            {"user": "Synthetic fictional first question", "assistant": "Mock first reply", "outcome": "complete"},
            {"user": "Synthetic correction about that reply", "assistant": "Mock second reply", "outcome": "complete"},
        ],
    }


class MultiTurnProtocolTests(unittest.TestCase):
    def test_two_arm_format_never_claims_authenticity_or_semantic_pass(self):
        base = specimen()
        adapter = specimen("trained", sid="session_trained_1")
        adapter["adapter_sha256"] = "c" * 64
        adapter["turns"][1]["assistant"] = "I am Vera, my personality is authentic"
        report = screen_sessions([base, adapter])
        self.assertTrue(report["declared_multiple_turn_attempts_format_validated"])
        self.assertFalse(report["actual_multi_turn_conversation_attested"])
        self.assertTrue(report["cross_arm_input_format_compared"])
        self.assertFalse(report["cross_arm_model_quality_compared"])
        self.assertTrue(report["matched_user_context_validated"])
        self.assertTrue(report["matched_generation_settings_validated"])
        self.assertFalse(report["model_identity_attested"])
        self.assertFalse(report["human_review_performed"])
        self.assertFalse(report["heldout_independence_verified"])
        self.assertFalse(report["semantic_personality_qualified"])
        self.assertEqual(report["arms"]["trained"]["review_status"], "REVIEW_REQUIRED")

    def test_same_case_must_have_identical_user_turns_between_arms(self):
        base = specimen()
        trained = specimen("trained", sid="trained1")
        trained["turns"][1]["user"] = "Changed fictional correction"
        with self.assertRaisesRegex(DialogueSessionError, "user contexts"):
            validate_sessions([base, trained])

    def test_same_case_must_have_identical_generation_settings(self):
        base = specimen()
        trained = specimen("trained", sid="trained1")
        trained["generation"]["max_reply_tokens"] = 96
        with self.assertRaisesRegex(DialogueSessionError, "generation"):
            validate_sessions([base, trained])

    def test_same_case_cannot_switch_source_claim_between_arms(self):
        base = specimen()
        trained = specimen("trained", sid="trained1")
        trained["source_claim"] = "private_unverified"
        with self.assertRaisesRegex(DialogueSessionError, "source declarations"):
            validate_sessions([base, trained])

    def test_declared_technical_censoring_is_counted_and_never_scored(self):
        case = specimen()
        case["turns"][1] = {"user": case["turns"][1]["user"], "assistant": None, "outcome": "truncated"}
        out = screen_sessions([case])
        self.assertEqual(out["arms"]["base"]["declared_complete_turns"], 1)
        self.assertEqual(out["arms"]["base"]["technical_censoring"]["truncated"], 1)
        self.assertFalse(out["semantic_personality_qualified"])

    def test_all_attempts_censored_cannot_be_called_real_dialogue(self):
        case = specimen()
        for turn in case["turns"]:
            turn["assistant"] = None
            turn["outcome"] = "timeout"
        out = screen_sessions([case])
        self.assertEqual(out["arms"]["base"]["declared_complete_turns"], 0)
        self.assertEqual(out["arms"]["base"]["technical_censoring"]["timeout"], 2)
        self.assertTrue(out["declared_multiple_turn_attempts_format_validated"])
        self.assertFalse(out["actual_multi_turn_conversation_attested"])
        self.assertFalse(out["semantic_personality_qualified"])

    def test_truncation_must_not_smuggle_partial_assistant_text(self):
        case = specimen()
        case["turns"][1].update(assistant="untrusted partial text", outcome="truncated")
        with self.assertRaisesRegex(DialogueSessionError, "technical outcome"):
            validate_sessions([case])

    def test_one_turn_is_not_multi_turn(self):
        case = specimen()
        case["turns"] = case["turns"][:1]
        with self.assertRaisesRegex(DialogueSessionError, "two turns"):
            validate_sessions([case])

    def test_one_arm_necessarily_has_no_cross_arm_comparison(self):
        report = screen_sessions([specimen()])
        self.assertFalse(report["cross_arm_input_format_compared"])
        self.assertFalse(report["cross_arm_model_quality_compared"])
        self.assertFalse(report["matched_user_context_validated"])
        self.assertFalse(report["matched_generation_settings_validated"])
        self.assertFalse(report["process_session_continuity_attested"])

    def test_duplicate_case_or_session_is_rejected(self):
        case = specimen()
        with self.assertRaisesRegex(DialogueSessionError, "duplicate"):
            validate_sessions([case, dict(case)])
        second = specimen("trained", sid=case["session_id"])
        with self.assertRaisesRegex(DialogueSessionError, "duplicate"):
            validate_sessions([case, second])

    def test_arm_cannot_change_declared_model_across_cases(self):
        a = specimen()
        b = specimen(case="fictional_case_2", sid="session_base_2")
        b["base_sha256"] = "d" * 64
        with self.assertRaisesRegex(DialogueSessionError, "model declaration"):
            validate_sessions([a, b])

    def test_arm_case_coverage_must_match(self):
        a = specimen()
        b = specimen(case="fictional_case_2", sid="session_base_2")
        c = specimen("trained", sid="session_trained_1")
        with self.assertRaisesRegex(DialogueSessionError, "coverage"):
            validate_sessions([a, b, c])

    def test_extra_fields_and_unsafe_ids_are_rejected(self):
        case = specimen()
        case["private_history"] = "NEVER_DUMP"
        with self.assertRaisesRegex(DialogueSessionError, "schema"):
            validate_sessions([case])
        case = specimen(arm="../path")
        with self.assertRaisesRegex(DialogueSessionError, "identifier"):
            validate_sessions([case])

    def test_boolean_numbers_are_not_valid_generation_parameters(self):
        case = specimen()
        case["generation"]["rng_seed"] = True
        with self.assertRaisesRegex(DialogueSessionError, "generation"):
            validate_sessions([case])

    def test_malformed_source_claim_and_adapter_hash_fail_closed(self):
        case = specimen()
        case["source_claim"] = ["private_unverified"]
        with self.assertRaisesRegex(DialogueSessionError, "source claim"):
            validate_sessions([case])
        case = specimen()
        case["adapter_sha256"] = "NO_ADAPTER_DIGEST"
        with self.assertRaisesRegex(DialogueSessionError, "digest"):
            validate_sessions([case])

    def test_cli_never_prints_raw_dialogue_even_on_failure(self):
        case = specimen()
        case["turns"][1]["assistant"] = "CANARY_PRIVATE_DIALOGUE_TEXT"
        with TemporaryDirectory() as folder:
            path = Path(folder) / "private.jsonl"
            path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main(["--input", str(path)]), 0)
            self.assertNotIn("CANARY_PRIVATE_DIALOGUE_TEXT", stdout.getvalue() + stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["arms"]["base"]["review_status"], "REVIEW_REQUIRED")
            path.write_text('{"arm":"base","arm":"LEAK_CANARY"}\n', encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main(["--input", str(path)]), 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotIn("LEAK_CANARY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
