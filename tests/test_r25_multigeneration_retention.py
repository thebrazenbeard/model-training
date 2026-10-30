"""Public/synthetic A→B→C retention provenance and negative-case tests."""

from dataclasses import replace
import hashlib
import json
import math
import unittest

from trainingday.admission import AdmissionError
from trainingday.r25_multigeneration_retention import (
    GenerationEvidence, PublicBankManifest, RetentionPolicy,
    assess_generation_retention, case_order_sha256, public_score_vector_sha256,
    wilson_lower_bound,
)


def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def example():
    ids = tuple(f"synthetic-{n:02d}" for n in range(16))
    manifest = PublicBankManifest(digest("public-source"), digest("frozen-base"),
                                  digest("initial-adapter"), ids)
    previous = manifest.initial_adapter_sha256
    records = []
    for name in "ABC":
        current = digest(f"adapter-{name}")
        records.append(GenerationEvidence(
            name=name, previous_adapter_sha256=previous,
            adapter_sha256=current, base_sha256=manifest.base_sha256,
            public_case_sha256=case_order_sha256(ids),
            score_receipt_sha256=digest(f"score-{name}"),
            score_vector_sha256=public_score_vector_sha256(
                case_order_sha256(ids), (True,) * len(ids)),
            training_receipt_sha256=digest(f"train-{name}"),
            case_passes=(True,) * len(ids),
            zero_control_present=True, corrupt_control_present=True,
            independent_scorer_claimed=True,
        ))
        previous = current
    return manifest, tuple(records)


def with_scores(record, scores, manifest):
    return replace(record, case_passes=scores,
                   score_vector_sha256=public_score_vector_sha256(
                       case_order_sha256(manifest.case_ids), scores))


class MultigenerationRetentionTests(unittest.TestCase):
    def test_stable_synthetic_chain_is_still_non_authorizing(self):
        manifest, records = example()
        result = assess_generation_retention(manifest, records)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["hold_reasons"],
                         ["DECLARED_STABILITY_NOT_INDEPENDENTLY_ATTESTED"])
        self.assertEqual([item["transition"] for item in result["comparisons"]],
                         ["A->B", "B->C", "A->C"])
        self.assertTrue(all(item["declared_regressions"] == 0
                            for item in result["comparisons"]))
        self.assertFalse(result["neural_improvement_proven"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["private_eval_unsealed"])
        self.assertNotIn("synthetic-00", json.dumps(result))

    def test_declared_forgetting_detects_regression_per_transition(self):
        manifest, records = example()
        b = with_scores(records[1], (False,) + (True,) * 15, manifest)
        result = assess_generation_retention(manifest, (records[0], b, records[2]))
        self.assertIn("DECLARED_CATASTROPHIC_FORGETTING_RISK", result["hold_reasons"])
        self.assertEqual(result["comparisons"][0]["declared_regressions"], 1)
        self.assertEqual(result["comparisons"][1]["declared_recoveries"], 1)
        self.assertFalse(result["promotion_allowed"])

    def test_a_to_c_regression_not_hidden_by_intermediate_recovery(self):
        manifest, records = example()
        c = with_scores(records[2], (True,) * 15 + (False,), manifest)
        result = assess_generation_retention(manifest, (records[0], records[1], c))
        self.assertEqual(result["comparisons"][2]["declared_regressions"], 1)

    def test_rejects_changed_parent_or_missing_generation(self):
        manifest, records = example()
        with self.assertRaisesRegex(AdmissionError, "lineage"):
            assess_generation_retention(manifest, (records[0], replace(
                records[1], previous_adapter_sha256=digest("other")), records[2]))
        with self.assertRaisesRegex(AdmissionError, "exact A-B-C"):
            assess_generation_retention(manifest, records[:2])
        with self.assertRaisesRegex(AdmissionError, "identity or order"):
            assess_generation_retention(manifest, tuple(reversed(records)))

    def test_rejects_forged_or_duplicate_adapter_digest_declarations(self):
        manifest, records = example()
        with self.assertRaisesRegex(AdmissionError, "lineage"):
            assess_generation_retention(manifest, (records[0], replace(
                records[1], adapter_sha256=records[0].adapter_sha256), records[2]))
        with self.assertRaisesRegex(AdmissionError, "lineage"):
            assess_generation_retention(manifest, (records[0], replace(
                records[1], score_receipt_sha256="a" * 63), records[2]))

    def test_rejects_changed_or_reordered_public_bank_commitment(self):
        manifest, records = example()
        changed = replace(manifest, case_ids=tuple(reversed(manifest.case_ids)))
        with self.assertRaisesRegex(AdmissionError, "commitment"):
            assess_generation_retention(changed, records)
        with self.assertRaisesRegex(AdmissionError, "duplicated"):
            case_order_sha256(("same", "same"))

    def test_rejects_private_or_training_contamination(self):
        manifest, records = example()
        for wrong in (
            replace(manifest, partition="PRIVATE_SEALED"),
            replace(manifest, answers_used_for_training=True),
        ):
            with self.subTest(wrong=wrong):
                with self.assertRaisesRegex(AdmissionError, "public-only"):
                    assess_generation_retention(wrong, records)
        for field in ("private_bank_accessed", "public_cases_used_for_training"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(AdmissionError, "leakage"):
                    assess_generation_retention(manifest, (replace(records[0], **{field: True}),) + records[1:])

    def test_overbudget_and_missing_or_wrong_score_rows_are_rejected(self):
        manifest, records = example()
        for malformed in (
            replace(records[0], time_budget_status="TIME_BUDGET_OVERRUN_INVALID"),
            replace(records[0], case_passes=records[0].case_passes[:-1]),
            replace(records[0], case_passes=(1,) + records[0].case_passes[1:]),
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(AdmissionError):
                    assess_generation_retention(manifest, (malformed,) + records[1:])

    def test_missing_controls_and_scorer_only_report_declared_hold(self):
        manifest, records = example()
        b = replace(records[1], zero_control_present=False, independent_scorer_claimed=False)
        result = assess_generation_retention(manifest, (records[0], b, records[2]))
        self.assertIn("CONTROL_RECEIPTS_UNVERIFIED", result["hold_reasons"])
        self.assertIn("SCORER_INDEPENDENCE_UNVERIFIED", result["hold_reasons"])
        self.assertFalse(result["independent_model_scoring_verified"])

    def test_baseline_without_demonstrated_success_cannot_graduate(self):
        manifest, records = example()
        a = with_scores(records[0], (False,) * 16, manifest)
        result = assess_generation_retention(manifest, (a, records[1], records[2]))
        self.assertIn("BASELINE_COMPETENCE_UNPROVEN", result["hold_reasons"])
        self.assertFalse(result["promotion_allowed"])

    def test_statistical_lower_bound_is_computed_not_trusted(self):
        self.assertAlmostEqual(wilson_lower_bound(16, 16), 0.80639232, places=6)
        manifest, records = example()
        result = assess_generation_retention(manifest, records,
                                             RetentionPolicy(minimum_wilson_lower_95=0.9))
        self.assertIn("INSUFFICIENT_DECLARED_CONFIDENCE", result["hold_reasons"])
        self.assertFalse(result["promotion_allowed"])

    def test_policy_cannot_relax_zero_regression_or_use_nan(self):
        manifest, records = example()
        for policy in (RetentionPolicy(max_case_regressions=1),
                       RetentionPolicy(minimum_wilson_lower_95=float("nan")),
                       RetentionPolicy(minimum_wilson_lower_95=True),
                       RetentionPolicy(minimum_wilson_lower_95=10**1000)):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(AdmissionError, "policy"):
                    assess_generation_retention(manifest, records, policy)

    def test_score_vector_tamper_with_stale_receipt_commitment_fails_closed(self):
        manifest, records = example()
        changed = replace(records[1], case_passes=(False,) + (True,) * 15)
        with self.assertRaisesRegex(AdmissionError, "score vector commitment"):
            assess_generation_retention(manifest, (records[0], changed, records[2]))
        rebound = with_scores(changed, changed.case_passes, manifest)
        reported = assess_generation_retention(manifest, (records[0], rebound, records[2]))
        self.assertIn("DECLARED_CATASTROPHIC_FORGETTING_RISK", reported["hold_reasons"])
        self.assertFalse(reported["promotion_allowed"])

    def test_invalid_bank_manifest_or_case_ids_are_rejected(self):
        manifest, records = example()
        with self.assertRaises(AdmissionError):
            assess_generation_retention(replace(manifest, source_sha256="z" * 64), records)
        with self.assertRaises(AdmissionError):
            assess_generation_retention(replace(manifest, case_ids=("../escape", "x")), records)

    def test_no_neural_gain_from_declared_controls_or_hashes(self):
        manifest, records = example()
        result = assess_generation_retention(manifest, records)
        for name in ("actual_tensor_change_verified", "fresh_process_adapter_reload_verified",
                     "independent_model_scoring_verified", "neural_improvement_proven",
                     "promotion_allowed"):
            self.assertIs(result[name], False)


if __name__ == "__main__":
    unittest.main()
