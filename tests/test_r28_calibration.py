"""Public synthetic calibration checks: never a private-bank or live-model evaluator."""

import hashlib
import math
import unittest
from dataclasses import replace

from trainingday.r27_pilot_preregistration import (
    PilotCase, PilotProtocol, preregister_pilot,
)
from trainingday.r28_calibration import (
    CalibrationError, CalibrationPlan, CalibrationSample,
    preregister_calibration, score_synthetic_calibration,
)


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        families = ("ARITHMETIC", "PROGRAM_REASONING", "SOURCE_PRIORITY")
        self.cases = tuple(PilotCase(
            f"case-{i:03d}", families[i % 3], digest(f"prompt-{i}"), digest("rights-public"),
        ) for i in range(120))
        self.protocol = PilotProtocol(
            "pilot-public-r28", digest("base"), digest("tokenizer"),
            digest("train-set"), (digest("other-train-prompt"),), self.cases,
            ("blind-A01", "blind-B02", "blind-C03", "blind-D04"), 0.10, 0.05,
        )
        self.pilot_receipt = preregister_pilot(self.protocol)
        self.plan = CalibrationPlan(
            "calibration-r28-public", "public-unseen-shadow-v1",
            self.pilot_receipt.manifest_sha256, 10, 0.05, 0.02,
        )
        self.receipt = preregister_calibration(self.protocol, self.pilot_receipt, self.plan)
        self.roles = {
            "blind-A01": "FROZEN",
            "blind-B02": "ZERO_UPDATE",
            "blind-C03": "CORRECT_LABEL",
            "blind-D04": "CORRUPTED_LABEL",
        }

    def samples(self, *, bad_confidence=False, bad_format=False, bad_refusal=False):
        out = {}
        for arm, role in self.roles.items():
            rows = {}
            for i, case in enumerate(self.cases):
                if role == "CORRECT_LABEL":
                    correct, confidence = i % 10 != 0, 0.88
                    if bad_confidence:
                        confidence = 0.99
                elif role == "CORRUPTED_LABEL":
                    correct, confidence = i % 3 == 0, 0.34
                else:
                    correct, confidence = i % 2 == 0, 0.50
                fmt, refused = True, False
                if role == "CORRECT_LABEL" and bad_format and i % 8 == 0:
                    correct, fmt = False, False
                if role == "CORRECT_LABEL" and bad_refusal and i % 8 == 0:
                    correct, refused = False, True
                rows[case.case_id] = CalibrationSample(
                    case.case_id, case.prompt_sha256, float(confidence),
                    bool(correct), fmt, refused,
                )
            out[arm] = rows
        return out

    def score(self, samples=None, plan=None, receipt=None, roles=None):
        return score_synthetic_calibration(
            self.protocol, self.pilot_receipt, plan or self.plan,
            receipt or self.receipt, roles or self.roles,
            samples if samples is not None else self.samples(),
        )

    def test_healthy_public_toy_calibration_still_cannot_claim_model_gain(self):
        result = self.score()
        self.assertEqual(result.status, "SYNTHETIC_CALIBRATION_THRESHOLD_MET_NO_CLAIM")
        self.assertTrue(result.synthetic_control_gates_passed)
        self.assertTrue(result.zero_update_parity_passed)
        self.assertTrue(result.calibration_nonregression_passed)
        self.assertLessEqual(result.correct_label.ece, result.frozen.ece + 0.02 + 1e-12)
        self.assertLess(result.correct_label.brier_score, result.frozen.brier_score)
        self.assertEqual(result.correct_label.case_count, 120)
        self.assertEqual(len(result.correct_label.bins), 10)
        self.assertAlmostEqual(result.frozen.brier_score, 0.25)
        self.assertAlmostEqual(result.correct_label.accuracy, 0.9)
        self.assertAlmostEqual(result.correct_label.ece, 0.02)
        self.assertLess(result.correct_minus_frozen_interval[0], result.correct_minus_frozen_gain)
        self.assertGreater(result.correct_minus_frozen_interval[1], result.correct_minus_frozen_gain)
        self.assertFalse(result.oracle_authenticated)
        self.assertFalse(result.blinded_roles_authenticated)
        self.assertFalse(result.physical_model_outputs_verified)
        self.assertFalse(result.neural_improvement_claim_authorized)
        self.assertFalse(result.release_authorized)

    def test_high_confidence_errors_can_block_despite_accuracy_gain(self):
        result = self.score(self.samples(bad_confidence=True))
        self.assertEqual(result.status, "NO_CLAIM_CALIBRATION_REGRESSION")
        self.assertTrue(result.synthetic_control_gates_passed)
        self.assertFalse(result.calibration_nonregression_passed)
        self.assertGreater(result.correct_label.overconfident_error_rate, 0)

    def test_single_high_confidence_error_blocks_even_if_average_ece_stays_in_tolerance(self):
        samples = self.samples()
        case = self.cases[0]
        old = samples["blind-C03"][case.case_id]
        self.assertFalse(old.correct)
        samples["blind-C03"][case.case_id] = replace(old, confidence=0.91)
        wider = replace(self.plan, max_calibration_regression=0.04)
        result = self.score(samples, plan=wider,
                            receipt=preregister_calibration(self.protocol, self.pilot_receipt, wider))
        self.assertTrue(result.synthetic_control_gates_passed)
        self.assertLessEqual(result.correct_label.ece, result.frozen.ece + wider.max_calibration_regression + 1e-12)
        self.assertEqual(result.status, "NO_CLAIM_CALIBRATION_REGRESSION")

    def test_refusal_or_format_regression_blocks(self):
        for change in ("bad_refusal", "bad_format"):
            with self.subTest(change=change):
                result = self.score(self.samples(**{change: True}))
                self.assertEqual(result.status, "NO_CLAIM_CALIBRATION_REGRESSION")
                self.assertFalse(result.calibration_nonregression_passed)

    def test_missing_control_or_no_gain_blocks_claim(self):
        samples = self.samples()
        samples["blind-C03"] = dict(samples["blind-A01"])
        result = self.score(samples)
        self.assertEqual(result.status, "NO_CLAIM_CONTROL_FAILURE")
        self.assertFalse(result.synthetic_control_gates_passed)

    def test_zero_update_must_preserve_frozen_confidence_not_just_correctness(self):
        samples = self.samples()
        samples["blind-B02"] = {
            k: replace(sample, confidence=0.0)
            for k, sample in samples["blind-B02"].items()
        }
        result = self.score(samples)
        self.assertEqual(result.status, "NO_CLAIM_ZERO_UPDATE_DIVERGENCE")
        self.assertFalse(result.zero_update_parity_passed)
        self.assertFalse(result.synthetic_control_gates_passed)
        self.assertAlmostEqual(result.frozen.ece, 0.0)
        self.assertAlmostEqual(result.zero_update.ece, 0.5)
        self.assertAlmostEqual(result.zero_update.brier_score, 0.5)
        self.assertFalse(result.neural_improvement_claim_authorized)

    def test_even_one_zero_update_probability_format_or_correctness_change_fails(self):
        for kind in ("confidence", "format_ok", "correct"):
            with self.subTest(kind=kind):
                samples = self.samples()
                case = self.cases[1]  # frozen incorrect; format false is consistent
                old = samples["blind-B02"][case.case_id]
                changes = ({"confidence": 0.5001} if kind == "confidence" else
                           {"format_ok": False} if kind == "format_ok" else
                           {"correct": True})
                samples["blind-B02"][case.case_id] = replace(old, **changes)
                result = self.score(samples)
                self.assertEqual(result.status, "NO_CLAIM_ZERO_UPDATE_DIVERGENCE")
                self.assertFalse(result.zero_update_parity_passed)

    def test_receipt_binding_run_split_and_policy_change_rejected(self):
        for changed in (
            replace(self.plan, run_id="another-run"),
            replace(self.plan, split_tag="other-split"),
            replace(self.plan, bin_count=5),
        ):
            with self.assertRaises(CalibrationError):
                self.score(plan=changed)
        with self.assertRaises(CalibrationError):
            self.score(receipt=replace(self.receipt, manifest_sha256=digest("forged")))

    def test_sample_digest_and_case_identity_tamper_rejected(self):
        samples = self.samples()
        first = self.cases[0]
        samples["blind-A01"][first.case_id] = replace(
            samples["blind-A01"][first.case_id], prompt_sha256=digest("wrong")
        )
        with self.assertRaises(CalibrationError):
            self.score(samples)

    def test_equality_overload_cannot_forge_sample_identity_or_receipt(self):
        class AnyEq:
            def __eq__(self, _other):
                return True

            def __ne__(self, _other):
                return False

        bad_samples = self.samples()
        first = self.cases[0]
        bad_samples["blind-A01"][first.case_id] = replace(
            bad_samples["blind-A01"][first.case_id],
            case_id=AnyEq(), prompt_sha256=AnyEq(),
        )
        with self.assertRaises(CalibrationError):
            self.score(bad_samples)
        with self.assertRaises(CalibrationError):
            self.score(receipt=replace(self.receipt,
                manifest_sha256=AnyEq(), canonical_json=AnyEq()))

        forged_pilot = replace(self.pilot_receipt,
            manifest_sha256=AnyEq(), canonical_json=AnyEq())
        with self.assertRaises(CalibrationError):
            score_synthetic_calibration(
                self.protocol, forged_pilot, self.plan, self.receipt,
                self.roles, self.samples(),
            )
        with self.assertRaises(CalibrationError):
            self.score(receipt=replace(self.receipt,
                oracle_authenticated=AnyEq(), release_authorized=AnyEq()))

    def test_spoofed_mapping_keys_cannot_count_as_committed_arms(self):
        class SpoofArm:
            def __hash__(self):
                return hash("blind-A01")

            def __eq__(self, _other):
                return True

            def __ne__(self, _other):
                return False

        original = self.samples()
        bad_arms = {SpoofArm(): original["blind-A01"],
                    **{k: v for k, v in original.items() if k != "blind-A01"}}
        with self.assertRaises(CalibrationError):
            self.score(bad_arms)

        bad_roles = {SpoofArm(): "FROZEN",
                     **{k: v for k, v in self.roles.items() if k != "blind-A01"}}
        with self.assertRaises(CalibrationError):
            self.score(roles=bad_roles)

    def test_nonfinite_malformed_or_inconsistent_declarations_rejected(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1, True, 1, "0.8", 10**1000):
            with self.subTest(bad=type(bad).__name__):
                samples = self.samples()
                first = self.cases[0]
                samples["blind-A01"][first.case_id] = replace(
                    samples["blind-A01"][first.case_id], confidence=bad
                )
                with self.assertRaises(CalibrationError):
                    self.score(samples)
        samples = self.samples()
        first = self.cases[0]
        samples["blind-A01"][first.case_id] = replace(
            samples["blind-A01"][first.case_id], correct=True, refused=True
        )
        with self.assertRaises(CalibrationError):
            self.score(samples)

    def test_oversized_maps_rejected_before_iterating_entries(self):
        class IterationSentinel(dict):
            def items(self):
                raise AssertionError("ITERATED_UNBOUNDED_MAP")

            def __iter__(self):
                raise AssertionError("ITERATED_UNBOUNDED_MAP")

        data = self.samples()
        data["blind-A01"] = IterationSentinel({**data["blind-A01"], **{
            f"extra-{i:06d}": 0 for i in range(50_000)
        }})
        with self.assertRaises(CalibrationError):
            self.score(data)

        data = IterationSentinel({**self.samples(), **{
            f"extra-arm-{i:06d}": {} for i in range(50_000)
        }})
        with self.assertRaises(CalibrationError):
            self.score(data)

    def test_confidence_boundaries_are_finite_and_wilson_interval_valid(self):
        data = self.samples()
        for sample in data["blind-A01"].values():
            data["blind-A01"][sample.case_id] = replace(sample, confidence=1.0 if sample.correct else 0.0)
        result = self.score(data)
        bins = result.frozen.bins
        self.assertEqual(bins[0].count, 60)
        self.assertEqual(bins[-1].count, 60)
        self.assertEqual(sum(b.count for b in bins), 120)
        for summary in (result.frozen, result.zero_update, result.correct_label, result.corrupted_label):
            self.assertTrue(0.0 <= summary.accuracy_interval[0] <= summary.accuracy <= summary.accuracy_interval[1] <= 1.0)
            self.assertTrue(math.isfinite(summary.ece))
            self.assertTrue(math.isfinite(summary.brier_score))


if __name__ == "__main__":
    unittest.main()
