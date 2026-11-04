"""Public-only evaluator falsification tests; zero protected gold."""

import copy
import unittest

from trainingday.public_transfer_evaluator import (
    PublicTransferEvaluationError, evaluate_public_fixture,
)


def sample():
    cases = [
        {"case_id": "a1", "prompt": "Which gate?", "expected_semantic": "gate-a", "stratum": "transfer-a", "family_id": "fa1", "source_group_id": "ga1", "template_id": "ta1"},
        {"case_id": "a2", "prompt": "Which latch?", "expected_semantic": "latch-b", "stratum": "transfer-a", "family_id": "fa2", "source_group_id": "ga2", "template_id": "ta2"},
        {"case_id": "b1", "prompt": "Which relay?", "expected_semantic": "relay-c", "stratum": "transfer-b", "family_id": "fb1", "source_group_id": "gb1", "template_id": "tb1"},
        {"case_id": "b2", "prompt": "Which anchor?", "expected_semantic": "anchor-d", "stratum": "transfer-b", "family_id": "fb2", "source_group_id": "gb2", "template_id": "tb2"},
    ]
    predictions = {}
    for arm in ("trained", "untrained", "sham", "shuffled_labels", "corrupted_labels"):
        predictions[arm] = [
            {"case_id": c["case_id"], "semantic_answer": c["expected_semantic"] if arm == "trained" else "wrong"}
            for c in cases
        ]
    return {"fixture": {"visibility": "public", "purpose": "development", "split": "transfer", "cases": cases},
            "train_rows": [], "predictions": predictions}


class PublicTransferEvaluatorTest(unittest.TestCase):
    def test_public_differential_is_never_q6(self):
        result = evaluate_public_fixture(sample())
        self.assertEqual(result["result"], "PUBLIC_DEV_DIFFERENTIAL_ONLY")
        self.assertEqual(result["q6_status"], "HOLD")
        self.assertFalse(result["qualifies_q6"])
        self.assertEqual(result["paired_against_trained"]["sham"]["trained_wins"], 4)
        self.assertEqual(result["provenance"]["physical_model_sha256"], "UNKNOWN")

    def test_corrupted_tie_rejects_claim(self):
        b = sample()
        b["predictions"]["corrupted_labels"] = copy.deepcopy(b["predictions"]["trained"])
        self.assertEqual(evaluate_public_fixture(b)["result"], "NO_DIFFERENTIAL_SIGNAL")

    def test_each_stratum_must_beat_controls(self):
        b = sample()
        for row in b["predictions"]["trained"]:
            if row["case_id"].startswith("b"):
                row["semantic_answer"] = "wrong"
        self.assertEqual(evaluate_public_fixture(b)["result"], "NO_DIFFERENTIAL_SIGNAL")

    def test_sham_control_required(self):
        b = sample()
        del b["predictions"]["sham"]
        with self.assertRaises(PublicTransferEvaluationError):
            evaluate_public_fixture(b)

    def test_rejects_nonpublic_or_sealed_claims(self):
        for mutation in ({"visibility": "private"}, {"sealed": True}, {"purpose": "final"}):
            b = sample()
            b["fixture"].update(mutation)
            with self.assertRaises(PublicTransferEvaluationError):
                evaluate_public_fixture(b)

    def test_exact_prompt_leakage(self):
        b = sample()
        b["train_rows"] = [{"prompt": "  WHICH   GATE? "}]
        with self.assertRaisesRegex(PublicTransferEvaluationError, "prompt leakage"):
            evaluate_public_fixture(b)

    def test_template_family_and_source_leakage(self):
        for field, value in (("family_id", "fa1"), ("source_group_id", "gb2"), ("template_id", "ta2")):
            b = sample()
            b["train_rows"] = [{field: value}]
            with self.assertRaisesRegex(PublicTransferEvaluationError, "leakage"):
                evaluate_public_fixture(b)

    def test_duplicate_and_missing_cases_fail(self):
        for mutate in (lambda rows: rows.append(rows[0]), lambda rows: rows.pop()):
            b = sample()
            mutate(b["predictions"]["trained"])
            with self.assertRaises(PublicTransferEvaluationError):
                evaluate_public_fixture(b)

    def test_numeric_suffix_does_not_count_as_semantic(self):
        b = sample()
        for rows in b["predictions"].values():
            for row in rows:
                row["semantic_answer"] = "16"
        self.assertEqual(evaluate_public_fixture(b)["scores"]["trained"]["correct"], 0)

    def test_receipt_determinism_and_redaction(self):
        b = sample()
        a = evaluate_public_fixture(b)
        self.assertEqual(a, evaluate_public_fixture(copy.deepcopy(b)))
        self.assertNotIn("Which gate?", str(a))
        self.assertNotIn("gate-a", str(a))
        self.assertEqual(len(a["receipt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
