"""Synthetic protocol tests; no physical Vera inference and no private user data."""

import contextlib
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from trainingday.dialogue_rubric_receipts import (
    DIMENSIONS, LEGACY_TO_PROPOSED_HINTS, PROPOSED_SCHEMA, RubricReceiptError,
    screen_blinded_reviews, main,
)


def session(arm="base", case="correction", budget=64, *, censor=False):
    turns = [
        {"user": "I prefer tea.", "assistant": None if censor else "Understood.",
         "outcome": "timeout" if censor else "complete"},
        {"user": "Correction: I prefer water. What do I prefer now?",
         "assistant": None if censor else "You prefer water.",
         "outcome": "truncated" if censor else "complete"},
    ]
    return {
        "arm": arm, "case_id": case, "session_id": f"{arm}_{case}",
        "source_claim": "mock_test", "origin_claim": "synthetic-only",
        "base_sha256": "a" * 64,
        "adapter_sha256": "NO_ADAPTER" if arm == "base" else "b" * 64,
        "prompt_sha256": "c" * 64,
        "generation": {"context_window_tokens": 4096, "max_reply_tokens": budget,
                       "temperature_milli": 700, "rng_seed": 17},
        "turns": turns,
    }


def assignment(row, blind_id="blind_one", arm_kind="base"):
    raw = json.dumps(row["turns"], sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return {"blind_id": blind_id, "arm": row["arm"],
            "case_id": row["case_id"], "session_id": row["session_id"],
            "arm_kind": arm_kind, "turns_sha256": sha256(raw).hexdigest()}


def review(blind_id="blind_one", *, score=2, kind="human", ne=False):
    values = {key: "NOT_EVALUABLE" if ne else score for key in DIMENSIONS}
    reasons = {key: "TECHNICAL_CENSORING" for key in DIMENSIONS} if ne else {}
    return {"blind_id": blind_id, "reviewer_id": "rater_opaque", "reviewer_kind": kind,
            "rubric_schema": PROPOSED_SCHEMA, "scores": values,
            "not_evaluable_reasons": reasons}


class RubricReceiptsTests(unittest.TestCase):
    def test_declared_human_score_is_not_an_actual_model_metric(self):
        row = session()
        result = screen_blinded_reviews([row], [assignment(row)], [review()], budget=64)
        self.assertEqual(result["evidence_class"], "UNVERIFIED_DECLARED_REVIEW_RECEIPTS")
        self.assertFalse(result["actual_model_behavior_measured"])
        self.assertFalse(result["independent_human_review_attested"])
        self.assertFalse(result["semantic_improvement_established"])
        self.assertFalse(result["protected_eval_custody_attested"])
        self.assertNotIn("scores", result)
        self.assertEqual(result["review_receipt_count_unverified"], 1)
        self.assertEqual(result["review_receipt_types_unverified"]["human"], 1)

    def test_no_model_response_no_numeric_metric(self):
        row = session(censor=True)
        result = screen_blinded_reviews([row], [assignment(row)], [review(ne=True)], budget=64)
        self.assertEqual(result["declared_complete_turns"], 0)
        self.assertEqual(result["not_evaluable_counts"], {key: 1 for key in DIMENSIONS})
        self.assertFalse(result["actual_model_behavior_measured"])

    def test_all_censored_disallows_scored_reply(self):
        row = session(censor=True)
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [review()], budget=64)

    def test_only_one_complete_turn_disallows_carryover_and_flow_score(self):
        row = session()
        row["turns"][1]["outcome"] = "timeout"
        row["turns"][1]["assistant"] = None
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [review()], budget=64)

    def test_review_only_without_actual_transcripts_rejected(self):
        with self.assertRaises(ValueError):
            screen_blinded_reviews([], [], [review()], budget=64)

    def test_missing_reviews_produces_no_model_metric(self):
        row = session()
        result = screen_blinded_reviews([row], [assignment(row)], [], budget=64)
        self.assertEqual(result["status"], "AWAITING_BLINDED_REVIEW_RECEIPTS")
        self.assertEqual(result["review_receipt_count_unverified"], 0)
        self.assertFalse(result["actual_model_behavior_measured"])

    def test_rejects_budget_outside_explicit_64_96_128(self):
        row = session(budget=32)
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [], budget=32)
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [], budget=64)

    def test_accepts_96_and_128_separately(self):
        for budget in (96, 128):
            with self.subTest(budget=budget):
                row = session(budget=budget)
                result = screen_blinded_reviews([row], [assignment(row)], [], budget=budget)
                self.assertEqual(result["declared_budget_tokens"], budget)

    def test_cross_arm_user_context_must_match(self):
        a, b = session("base"), session("trained")
        b["turns"][0]["user"] = "Changed input!"
        with self.assertRaises(ValueError):
            screen_blinded_reviews([a, b], [assignment(a), assignment(b, "blind_two", "r16_correct_pilot")], [], budget=64)

    def test_cross_arm_matching_and_blind_assignments(self):
        a, b = session("base"), session("trained")
        b["turns"][1]["assistant"] = "Water, acknowledged."
        rows = [assignment(a), assignment(b, "blind_two", "r16_correct_pilot")]
        result = screen_blinded_reviews([a, b], rows, [review(), review("blind_two")], budget=64)
        self.assertEqual(result["declared_arms_count"], 2)
        self.assertTrue(result["declared_cross_arm_matched_inputs"])
        self.assertFalse(result["cross_arm_semantic_quality_compared"])

    def test_partial_blind_coverage_must_be_explicit(self):
        a, b = session("base"), session("trained")
        entries = [assignment(a), assignment(b, "blind_two", "r16_correct_pilot")]
        result = screen_blinded_reviews([a, b], entries, [review()], budget=64)
        self.assertEqual(result["status"], "PARTIAL_DECLARED_REVIEW_RECEIPTS")
        self.assertEqual(result["reviewed_assignment_count"], 1)
        self.assertEqual(result["unreviewed_assignment_count"], 1)
        self.assertFalse(result["all_assignments_reviewed_declared"])

    def test_technically_censored_reason_requires_technical_turn(self):
        row = session()
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [review(ne=True)], budget=64)

    def test_insufficient_turns_reason_cannot_cover_two_complete_replies(self):
        row = session()
        recorded = review(ne=True)
        recorded["not_evaluable_reasons"] = {
            key: "INSUFFICIENT_TURNS" for key in DIMENSIONS
        }
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [recorded], budget=64)

    def test_rejects_changed_transcript_after_blind_assignment(self):
        row = session()
        entry = assignment(row)
        row["turns"][0]["assistant"] = "Manipulated reply"
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [entry], [review()], budget=64)

    def test_rejects_duplicate_or_unbound_reviews(self):
        row = session()
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [review(), review()], budget=64)
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [review("unknown")], budget=64)

    def test_rejects_scale_mixing_and_bool(self):
        row = session()
        r = review()
        r["rubric_schema"] = "VERA_PUBLIC_PAIRED_SMOKE_0_2_V1"
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [r], budget=64)
        r = review(score=True)
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [r], budget=64)
        r = review(score=4)
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [r], budget=64)

    def test_not_evaluable_requires_constrained_reason(self):
        row = session()
        r = review(ne=True)
        r["not_evaluable_reasons"]["I"] = "private user data must never be printed"
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [r], budget=64)
        r["not_evaluable_reasons"]["I"] = ["TECHNICAL_CENSORING"]
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row)], [r], budget=64)

    def test_model_reviewer_is_not_independent_human_review(self):
        row = session()
        result = screen_blinded_reviews([row], [assignment(row)], [review(kind="model")], budget=64)
        self.assertEqual(result["review_receipt_types_unverified"], {"model": 1})
        self.assertFalse(result["independent_human_review_attested"])

    def test_duplicate_assignment_to_one_session_rejected(self):
        row = session()
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([row], [assignment(row), assignment(row, "second")], [], budget=64)

    def test_arm_kind_change_within_arm_rejected(self):
        first, second = session(case="one"), session(case="two")
        with self.assertRaises(RubricReceiptError):
            screen_blinded_reviews([first, second],
                                   [assignment(first), assignment(second, "second", "persona_base")],
                                   [], budget=64)

    def test_old_scale_never_numeric_converts_to_new_scale(self):
        self.assertTrue(LEGACY_TO_PROPOSED_HINTS)
        for old, mapping in LEGACY_TO_PROPOSED_HINTS.items():
            self.assertIsInstance(old, str)
            self.assertTrue(all(key in DIMENSIONS for key in mapping))

    def test_cli_never_prints_private_dialogue_or_exception(self):
        row = session()
        row["turns"][0]["user"] = "owner-private-string-do-not-echo"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sessions_path, assignments_path, reviews_path = (
                path / "sessions.jsonl", path / "assignments.jsonl", path / "reviews.jsonl")
            for target, records in ((sessions_path, [row]),
                                    (assignments_path, [assignment(row)]),
                                    (reviews_path, [review()])):
                target.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = main(["--sessions", str(sessions_path), "--assignments", str(assignments_path),
                           "--reviews", str(reviews_path), "--budget", "64"])
            self.assertEqual(rc, 0)
            self.assertNotIn("owner-private-string-do-not-echo", stdout.getvalue() + stderr.getvalue())
            self.assertNotIn("You prefer water", stdout.getvalue() + stderr.getvalue())
            self.assertFalse(json.loads(stdout.getvalue())["actual_model_behavior_measured"])


if __name__ == "__main__":
    unittest.main()
