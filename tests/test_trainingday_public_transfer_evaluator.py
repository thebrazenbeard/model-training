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
            "train_manifest": {"manifest_id": "public-dev-training-v1"},
            "train_rows": [{"prompt": "Unrelated public prompt", "family_id": "training-only-family",
                            "source_group_id": "training-only-source", "template_id": "training-only-template"}],
            "predictions": predictions}


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
        b["train_rows"][0]["prompt"] = "  WHICH   GATE? "
        with self.assertRaisesRegex(PublicTransferEvaluationError, "prompt leakage"):
            evaluate_public_fixture(b)

    def test_template_family_and_source_leakage(self):
        for field, value in (("family_id", "fa1"), ("source_group_id", "gb2"), ("template_id", "ta2")):
            b = sample()
            b["train_rows"][0][field] = value
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

    def test_arbitrary_gold_and_predictions_are_never_qualified(self):
        b = sample()
        for case in b["fixture"]["cases"]:
            case["expected_semantic"] = "arbitrary-caller-label"
        for row in b["predictions"]["trained"]:
            row["semantic_answer"] = "arbitrary-caller-label"
        result = evaluate_public_fixture(b)
        self.assertEqual(result["result"], "PUBLIC_DEV_DIFFERENTIAL_ONLY")
        self.assertFalse(result["qualifies_q6"])
        self.assertEqual(result["q6_status"], "HOLD")
        for field in ("gold", "predictions", "provenance"):
            self.assertEqual(result["input_trust"][field], "UNVERIFIED_CALLER_INPUT")

    def test_missing_or_empty_training_manifest_rejected(self):
        for mutation in ("missing_rows", "empty_rows", "missing_manifest", "blank_manifest"):
            b = sample()
            if mutation == "missing_rows":
                del b["train_rows"]
            elif mutation == "empty_rows":
                b["train_rows"] = []
            elif mutation == "missing_manifest":
                del b["train_manifest"]
            else:
                b["train_manifest"]["manifest_id"] = "  "
            with self.assertRaises(PublicTransferEvaluationError):
                evaluate_public_fixture(b)

    def test_blank_case_and_training_group_identifiers_rejected(self):
        for part in ("fixture", "train_rows"):
            for field in ("family_id", "source_group_id", "template_id"):
                b = sample()
                row = b[part]["cases"][0] if part == "fixture" else b[part][0]
                row[field] = "  "
                with self.assertRaises(PublicTransferEvaluationError):
                    evaluate_public_fixture(b)

    def test_strata_normalize_and_cannot_fake_two_groups(self):
        b = sample()
        for case in b["fixture"]["cases"]:
            case["stratum"] = " TRANSFER-A " if case["case_id"].startswith("a") else "transfer-a"
        with self.assertRaisesRegex(PublicTransferEvaluationError, "two transfer strata"):
            evaluate_public_fixture(b)

    def test_duplicate_eval_prompt_after_normalization_rejected(self):
        b = sample()
        b["fixture"]["cases"][1]["prompt"] = " WHICH   GATE? "
        with self.assertRaisesRegex(PublicTransferEvaluationError, "Duplicate eval prompt"):
            evaluate_public_fixture(b)

    def test_nonstring_case_id_raises_domain_error(self):
        for invalid in ([], {}, 99, None):
            b = sample()
            b["predictions"]["trained"][0]["case_id"] = invalid
            with self.assertRaises(PublicTransferEvaluationError):
                evaluate_public_fixture(b)
            b = sample()
            b["fixture"]["cases"][0]["case_id"] = invalid
            with self.assertRaises(PublicTransferEvaluationError):
                evaluate_public_fixture(b)

    def test_private_custody_claim_is_not_echoed(self):
        b = sample()
        b["provenance"] = {"evaluation_bank_id": "SEALED-PRIVATE-FAKE",
                           "evaluation_custodian": "PRIVATE-CLAIM-NOT-VERIFIED"}
        result = evaluate_public_fixture(b)
        self.assertEqual(result["provenance"]["evaluation_bank_id"], "UNKNOWN")
        self.assertEqual(result["provenance"]["evaluation_custodian"], "UNKNOWN")
        self.assertNotIn("PRIVATE-CLAIM", str(result))
        self.assertNotIn("SEALED-PRIVATE-FAKE", str(result))


if __name__ == "__main__":
    unittest.main()
