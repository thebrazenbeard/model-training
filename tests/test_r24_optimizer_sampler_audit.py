"""Public-safe synthetic R24 MeZO/QLoRA optimizer and sampler fairness tests."""

import tempfile
import unittest
from pathlib import Path

from trainingday.r24_optimizer_sampler_audit import (
    ArmTimings,
    CaseTokens,
    assess_source_and_exposure,
    inspect_adamw8bit_weight_decay,
    simulate_source_exposure,
)


def cases(count=16):
    return tuple(CaseTokens(f"synthetic-{i}", 12 + i, 2 + i % 3)
                 for i in range(count))


def timings(steps, *, budget=100.0, each=10.0):
    return ArmTimings(tuple([each] * steps), budget)


class R24SourceOptimizerAndSamplerAuditTests(unittest.TestCase):
    def test_actual_qlora_source_omits_explicit_weight_decay(self):
        src = Path(__file__).resolve().parents[1] / "trainingday" / "r18_qlora.py"
        evidence = inspect_adamw8bit_weight_decay(src)
        self.assertEqual(evidence["status"], "UNDECLARED_DEFAULT_UNKNOWN")
        self.assertIsNone(evidence["weight_decay"])
        self.assertRegex(evidence["source_sha256"], r"^[a-f0-9]{64}$")

    def test_explicit_constant_weight_decay_is_syntactically_identified(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "fixture.py"
            src.write_text("def load():\n    return AdamW8bit([], lr=1e-4, weight_decay=0.0)\n")
            evidence = inspect_adamw8bit_weight_decay(src)
        self.assertEqual(evidence["status"], "EXPLICIT_SOURCE_CONSTANT")
        self.assertEqual(evidence["weight_decay"], 0.0)

    def test_dynamic_weight_decay_is_not_accepted_as_effective_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "fixture.py"
            src.write_text("def load(decay):\n    return AdamW8bit([], weight_decay=decay)\n")
            evidence = inspect_adamw8bit_weight_decay(src)
        self.assertEqual(evidence["status"], "DYNAMIC_VALUE_UNATTESTED")
        self.assertIsNone(evidence["weight_decay"])

    def test_multiple_optimizer_constructors_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "fixture.py"
            src.write_text("a = AdamW8bit([], weight_decay=0)\nb = AdamW8bit([])\n")
            evidence = inspect_adamw8bit_weight_decay(src)
        self.assertEqual(evidence["status"], "AMBIGUOUS_CONSTRUCTORS")

    def test_equal_one_sweep_has_identical_exposures_but_different_compute(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        self.assertEqual(mezo["presentation_counts"], qlora["presentation_counts"])
        self.assertEqual(mezo["example_presentations"], 16)
        self.assertEqual(mezo["duplicate_presentations"], 0)
        self.assertEqual(mezo["forward_sample_tokens"], 2 * qlora["forward_sample_tokens"])
        self.assertEqual(mezo["backward_sample_tokens"], 0)
        self.assertEqual(qlora["backward_sample_tokens"], qlora["forward_sample_tokens"])
        self.assertEqual(mezo["padded_forward_tokens"], 2 * qlora["padded_forward_tokens"])

    def test_mezo_cycles_but_qlora_rejects_second_epoch(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(3), max_seq_length=64)
        self.assertEqual(mezo["example_presentations"], 24)
        self.assertEqual(mezo["unique_examples_seen"], 16)
        self.assertEqual(mezo["duplicate_presentations"], 8)
        self.assertEqual(mezo["presentation_counts"]["synthetic-0"], 2)
        with self.assertRaisesRegex(ValueError, "single-sweep"):
            simulate_source_exposure("qlora", cases(), timings(3), max_seq_length=64)

    def test_mezo_configurable_batch_must_match_qlora_fixed_eight(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2),
                                        max_seq_length=64, examples_per_update=4)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.0})
        self.assertEqual(mezo["effective_batch_size"], 4)
        self.assertIn("EFFECTIVE_BATCH_SIZE_MISMATCH", result["hold_reasons"])
        with self.assertRaisesRegex(ValueError, "fixes"):
            simulate_source_exposure("qlora", cases(), timings(2),
                                     max_seq_length=64, examples_per_update=4)

    def test_source_hazards_hold_even_when_synthetic_steps_match(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        evidence = {"status": "UNDECLARED_DEFAULT_UNKNOWN", "weight_decay": None}
        result = assess_source_and_exposure(mezo, qlora, evidence)
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["authorizes_gpu_start"])
        self.assertFalse(result["proves_neural_improvement"])
        self.assertIn("QLORA_WEIGHT_DECAY_DEFAULT_UNVERIFIED", result["hold_reasons"])

    def test_cyclic_repeats_are_material_to_comparison(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(3), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.0})
        self.assertIn("CASE_EXPOSURE_MISMATCH", result["hold_reasons"])
        self.assertIn("REPEATED_EXAMPLES_IN_MEZO", result["hold_reasons"])
        self.assertIn("INPUT_TOKEN_EXPOSURE_MISMATCH", result["hold_reasons"])

    def test_equal_budget_but_unequal_active_time_is_flagged(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2, each=11), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2, each=10), max_seq_length=64)
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.0})
        self.assertIn("ACTIVE_OPTIMIZER_TIME_DIFFERENT", result["hold_reasons"])

    def test_run_exceeding_budget_is_invalid_even_if_equal_to_other_arm(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2, budget=10), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2, budget=10), max_seq_length=64)
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.0})
        self.assertIn("TIME_BUDGET_OVERRUN_INVALID", result["hold_reasons"])

    def test_nominal_synthetic_parity_never_authorizes_training(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.0})
        self.assertEqual(result["hold_reasons"], [])
        self.assertTrue(result["synthetic_schedule_matched"])
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["authorizes_gpu_start"])

    def test_nonzero_explicit_weight_decay_is_not_a_mezo_match(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.01})
        self.assertIn("OPTIMIZER_REGULARIZATION_MISMATCH", result["hold_reasons"])

    def test_missing_provenance_is_rejected_even_with_matching_counts(self):
        mezo = simulate_source_exposure("mezo", cases(), timings(2), max_seq_length=64)
        qlora = simulate_source_exposure("qlora", cases(), timings(2), max_seq_length=64)
        mezo["evidence_class"] = "HOST_ACTUAL"
        result = assess_source_and_exposure(mezo, qlora,
                                            {"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": 0.0})
        self.assertIn("SYNTHETIC_PROVENANCE_MISSING", result["hold_reasons"])

    def test_extreme_numeric_timing_is_controlled_invalid(self):
        with self.assertRaises(ValueError):
            simulate_source_exposure("mezo", cases(), ArmTimings((10**1000,), 100.0), 64)

    def test_finite_individual_steps_with_nonfinite_sum_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite cumulative"):
            simulate_source_exposure("mezo", cases(),
                                     ArmTimings((1e308, 1e308), 1e308), 64)

    def test_rejects_nonfinite_timing_and_bool_token_counts(self):
        with self.assertRaises(ValueError):
            simulate_source_exposure("mezo", cases(), timings(2, each=float("inf")), 64)
        malformed = (CaseTokens("case", True, 1),) + cases(7)
        with self.assertRaises(ValueError):
            simulate_source_exposure("mezo", malformed, timings(1), 64)

    def test_rejects_duplicate_case_ids_and_bad_fit(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            simulate_source_exposure("mezo", tuple([CaseTokens("x", 10, 3)] * 8), timings(1), 64)
        with self.assertRaisesRegex(ValueError, "fit"):
            simulate_source_exposure("mezo", cases(), timings(1), 2)


if __name__ == "__main__":
    unittest.main()
