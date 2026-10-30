"""R20 public (nonsealed) retention aggregation; synthetic rubric declarations only."""

import copy
import hashlib
import unittest
from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.retention_canary import IDS, preregister
from trainingday.retention_comparison import compare_retention_receipts


REPO = Path(__file__).resolve().parents[1]


def sha(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def receipt(arm, bits, *, base=sha("base")):
    initial = sha("initial")
    return {
        "arm": arm,
        "source_sha256": sha("placeholder"),
        "case_id_sha256": sha("placeholder"),
        "trial_binding_sha256": sha("one matched experiment"),
        "base_binding_sha256": base,
        "trainable_before_sha256": initial,
        "trainable_after_sha256": initial if arm in ("frozen", "zero") else sha(arm),
        "data_partition": "PUBLIC_NONSEALED_RETENTION_ONLY",
        "private_eval_access": False,
        "canary_used_for_training": False,
        "method": "DECLARED_RUBRIC_JUDGMENTS_NOT_ATTESTED",
        "rows": [{"id": key, "rubric_pass": bit} for key, bit in zip(IDS, bits)],
    }


class RetentionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.preflight = preregister(REPO)
        self.receipts = [
            receipt("frozen", [True, True, False, False, False]),
            receipt("zero", [True, True, False, False, False]),
            receipt("correct", [False, True, True, True, False]),
            receipt("corrupt", [False, False, False, False, False]),
        ]
        for arm in self.receipts:
            arm["source_sha256"] = self.preflight["source_sha256"]
            arm["case_id_sha256"] = self.preflight["case_id_sha256"]

    def test_aggregate_gain_does_not_hide_regression_on_frozen_pass_case(self):
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["pass_counts"], {
            "frozen": 2, "zero": 2, "correct": 3, "corrupt": 0,
        })
        self.assertEqual(result["correct_vs_frozen"]["regressions"], 1)
        self.assertEqual(result["correct_vs_frozen"]["recoveries"], 2)
        self.assertEqual(result["status"], "DECLARED_RETENTION_REGRESSION")
        self.assertFalse(result["model_improvement_proven"])
        self.assertFalse(result["promotion_allowed"])
        self.assertNotIn("user_turns", str(result))

    def test_no_declared_regression_is_still_not_physical_qualification(self):
        self.receipts[2]["rows"] = [
            {"id": key, "rubric_pass": bit}
            for key, bit in zip(IDS, [True, True, True, True, True])
        ]
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["status"], "NO_DECLARED_REGRESSION_UNVERIFIED")
        self.assertEqual(result["correct_vs_frozen"]["regressions"], 0)
        self.assertFalse(result["promotion_allowed"])

    def test_corrupted_label_matching_correct_is_explicitly_flagged(self):
        for arm in self.receipts[2:]:
            for row in arm["rows"]:
                row["rubric_pass"] = True
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["status"], "CORRUPT_CONTROL_NOT_SEPARATED")
        self.assertEqual(result["correct_vs_corrupt"]["correct_only_passes"], 0)
        self.assertEqual(result["correct_vs_corrupt"]["corrupt_only_passes"], 0)
        self.assertTrue(result["correct_vs_corrupt"]["corrupt_matches_or_exceeds_correct"])
        self.assertFalse(result["promotion_allowed"])

    def test_corrupted_label_exceeding_correct_is_explicitly_flagged(self):
        self.receipts[2]["rows"] = [
            {"id": key, "rubric_pass": bit}
            for key, bit in zip(IDS, [True, True, False, False, False])
        ]
        self.receipts[3]["rows"] = [
            {"id": key, "rubric_pass": bit}
            for key, bit in zip(IDS, [True, True, True, False, False])
        ]
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["status"], "CORRUPT_CONTROL_NOT_SEPARATED")
        self.assertEqual(result["correct_vs_corrupt"]["corrupt_only_passes"], 1)
        self.assertTrue(result["correct_vs_corrupt"]["corrupt_matches_or_exceeds_correct"])

    def test_corruption_anomaly_is_visible_even_when_retention_regresses(self):
        for row in self.receipts[3]["rows"]:
            row["rubric_pass"] = True
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["status"], "DECLARED_RETENTION_REGRESSION")
        self.assertTrue(result["correct_vs_corrupt"]["corrupt_matches_or_exceeds_correct"])

    def test_zero_frozen_success_cannot_prove_retention(self):
        for arm in self.receipts[:2]:
            for row in arm["rows"]:
                row["rubric_pass"] = False
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["status"], "FROZEN_COMPETENCE_UNPROVEN")
        self.assertFalse(result["model_improvement_proven"])

    def test_missing_duplicate_or_misbound_arm_rejected(self):
        alterations = [
            self.receipts[:-1],
            self.receipts + [copy.deepcopy(self.receipts[2])],
        ]
        wrong_base = copy.deepcopy(self.receipts)
        wrong_base[2]["base_binding_sha256"] = sha("other-base")
        alterations.append(wrong_base)
        wrong_input = copy.deepcopy(self.receipts)
        wrong_input[1]["trial_binding_sha256"] = sha("other-trial")
        alterations.append(wrong_input)
        for bad in alterations:
            with self.subTest(length=len(bad)):
                with self.assertRaises(AdmissionError):
                    compare_retention_receipts(self.preflight, bad)

    def test_changed_or_leaked_case_rows_rejected(self):
        for mutation in (
            lambda a: a[2]["rows"].reverse(),
            lambda a: a[2]["rows"][0].update(id=IDS[1]),
            lambda a: a[2]["rows"][0].update(rubric_pass=1),
            lambda a: a[2]["rows"][0].update(prompt="PRIVATE OR PUBLIC TEXT"),
            lambda a: a[2].update(case_id_sha256="0" * 64),
        ):
            bad = copy.deepcopy(self.receipts)
            mutation(bad)
            with self.assertRaises(AdmissionError):
                compare_retention_receipts(self.preflight, bad)

    def test_control_and_training_contamination_rejected(self):
        for change in (
            lambda a: a[1].update(trainable_after_sha256=sha("changed-zero")),
            lambda a: a[0].update(trainable_after_sha256=sha("changed-frozen")),
            lambda a: a[2].update(trainable_after_sha256=sha("initial")),
            lambda a: a[3].update(trainable_after_sha256=sha("initial")),
            lambda a: a[2].update(canary_used_for_training=True),
            lambda a: a[2].update(private_eval_access=True),
            lambda a: a[2].update(data_partition="TRAIN"),
            lambda a: a[2].update(method="SELF_CERTIFIED_TRUE"),
        ):
            bad = copy.deepcopy(self.receipts)
            change(bad)
            with self.assertRaises(AdmissionError):
                compare_retention_receipts(self.preflight, bad)

    def test_zero_control_disagreement_is_not_hidden_by_high_correct_score(self):
        self.receipts[1]["rows"][0]["rubric_pass"] = False
        result = compare_retention_receipts(self.preflight, self.receipts)
        self.assertEqual(result["status"], "ZERO_CONTROL_DISAGREEMENT")
        self.assertEqual(result["frozen_zero_case_disagreements"], 1)
        self.assertFalse(result["promotion_allowed"])

    def test_correct_and_corrupt_training_must_have_distinct_declared_final_adapters(self):
        # Sharing one purported final adapter defeats the corruption control.
        bad = copy.deepcopy(self.receipts)
        bad[3]["trainable_after_sha256"] = bad[2]["trainable_after_sha256"]
        with self.assertRaisesRegex(AdmissionError, "identical final adapter"):
            compare_retention_receipts(self.preflight, bad)

    def test_false_source_preflight_cannot_upgrade_result(self):
        bad = copy.deepcopy(self.preflight)
        bad["checkout_sha256"] = sha("not-the-public-source")
        with self.assertRaises(AdmissionError):
            compare_retention_receipts(bad, self.receipts)


if __name__ == "__main__":
    unittest.main()
