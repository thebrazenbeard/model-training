"""Adversarial R29 public CPU toy optimizer/objective tests."""

import unittest
from pathlib import Path

from trainingday.r24_optimizer_sampler_audit import (
    ArmTimings, inspect_adamw8bit_weight_decay,
)
from trainingday.r29_optimizer_fairness import (
    ToyTokenCase, ClockObservation, DeclaredControls,
    scalar_objective_witness, audit_fair_optimizer_pair,
)


def cases():
    return tuple(
        ToyTokenCase(
            case_id=f"toy-{i}",
            input_ids=(1, i + 2, 3, 4),
            labels=(-100, i + 2, 3, 4) if i % 2 else (-100, -100, 3, 4),
            target_logits=((0.0, 0.2, 1.0, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * (3 if i % 2 else 2),
        )
        for i in range(8)
    )


def clock(elapsed=5.0, wall=5.0, standby=0.0, budget=6.0):
    return ClockObservation(
        monotonic_elapsed_seconds=elapsed,
        wall_elapsed_seconds=wall,
        standby_seconds=standby,
        planned_budget_seconds=budget,
    )


def controls():
    return DeclaredControls(
        zero_before_sha256="a" * 64,
        zero_after_sha256="a" * 64,
        correct_label_sha256="b" * 64,
        corrupted_label_sha256="c" * 64,
        correct_label_updates=1,
        corrupted_label_updates=1,
    )


def pair(left=None, right=None, *, mezo_steps=(5.0,), qlora_steps=(5.0,),
         mezo_clock=None, qlora_clock=None, control=None, decay=0.0):
    return audit_fair_optimizer_pair(
        mezo_cases=left if left is not None else cases(),
        qlora_cases=right if right is not None else cases(),
        mezo_timings=ArmTimings(mezo_steps, 6.0),
        qlora_timings=ArmTimings(qlora_steps, 6.0),
        mezo_clock=mezo_clock if mezo_clock is not None else clock(),
        qlora_clock=qlora_clock if qlora_clock is not None else clock(),
        qlora_weight_decay_source={"status": "EXPLICIT_SOURCE_CONSTANT", "weight_decay": decay},
        declared_controls=control if control is not None else controls(),
        max_seq_length=8,
        tokenizer_sha256="d" * 64,
        corpus_sha256="e" * 64,
    )


class TestObjective(unittest.TestCase):
    def test_target_weighted_scalar_cross_entropy_and_microbatch_partition(self):
        witness = scalar_objective_witness(cases(), cases(), (3, 5))
        self.assertEqual(witness["status"], "PASS_PUBLIC_SCALAR_OBJECTIVE")
        self.assertAlmostEqual(witness["mezo_scalar_cross_entropy"],
                               witness["qlora_target_weighted_cross_entropy"], places=12)
        self.assertEqual(witness["supervised_target_tokens"], 20)
        self.assertNotAlmostEqual(witness["qlora_naive_unweighted_microbatch_mean"],
                                  witness["qlora_target_weighted_cross_entropy"], places=6)
        self.assertEqual(witness["first_case_id"], "toy-0")
        self.assertEqual(witness["last_case_id"], "toy-7")
        self.assertEqual(len(witness["target_id_commitment_sha256"]), 64)

    def test_target_id_mismatch_is_not_objective_parity(self):
        altered = list(cases())
        case = altered[-1]
        altered[-1] = ToyTokenCase(case.case_id, case.input_ids,
                                  case.labels[:-1] + (5,), case.target_logits)
        witness = scalar_objective_witness(cases(), tuple(altered), (3, 5))
        self.assertEqual(witness["status"], "HOLD")
        self.assertIn("TARGET_TOKEN_ID_COMMITMENT_MISMATCH", witness["hold_reasons"])

    def test_logits_snapshot_mismatch_is_not_scalar_parity(self):
        altered = list(cases())
        case = altered[0]
        altered[0] = ToyTokenCase(case.case_id, case.input_ids, case.labels,
                                  case.target_logits[:-1] + ((10.0,) * 11,))
        witness = scalar_objective_witness(cases(), tuple(altered), (3, 5))
        self.assertIn("FROZEN_LOGITS_SNAPSHOT_MISMATCH", witness["hold_reasons"])

    def test_same_counts_different_first_last_order_is_hold(self):
        reversed_cases = tuple(reversed(cases()))
        witness = scalar_objective_witness(cases(), reversed_cases, (3, 5))
        self.assertEqual(witness["first_case_id"], "toy-0")
        self.assertEqual(witness["qlora_first_case_id"], "toy-7")
        self.assertIn("TARGET_TOKEN_ID_COMMITMENT_MISMATCH", witness["hold_reasons"])

    def test_nonfinite_logits_fail_closed(self):
        altered = list(cases())
        case = altered[0]
        altered[0] = ToyTokenCase(case.case_id, case.input_ids, case.labels,
                                  ((float("nan"),) * 11,) + case.target_logits[1:])
        with self.assertRaises(ValueError):
            scalar_objective_witness(tuple(altered), cases(), (3, 5))

    def test_negative_nonmask_label_and_oversized_input_id_rejected(self):
        altered = list(cases())
        case = altered[1]
        altered[1] = ToyTokenCase(case.case_id, case.input_ids,
                                  (-100, -99, 3, 4), case.target_logits)
        with self.assertRaises(ValueError):
            scalar_objective_witness(tuple(altered), cases(), (3, 5))
        altered[1] = ToyTokenCase(case.case_id, (1, 999, 3, 4),
                                  case.labels, case.target_logits)
        with self.assertRaises(ValueError):
            scalar_objective_witness(tuple(altered), cases(), (3, 5))

    def test_very_large_python_int_logits_rejected_cleanly(self):
        altered = list(cases())
        case = altered[0]
        altered[0] = ToyTokenCase(case.case_id, case.input_ids, case.labels,
                                  ((10**1000,) * 11,) + case.target_logits[1:])
        with self.assertRaises(ValueError):
            scalar_objective_witness(tuple(altered), cases(), (3, 5))

    def test_invalid_target_id_and_partition_rejected(self):
        with self.assertRaises(ValueError):
            scalar_objective_witness(cases(), cases(), (4, 3))
        corrupt = list(cases())
        case = corrupt[0]
        corrupt[0] = ToyTokenCase(case.case_id, case.input_ids,
                                  case.labels[:-1] + (999,), case.target_logits)
        with self.assertRaises(ValueError):
            scalar_objective_witness(tuple(corrupt), cases(), (8,))


class TestAccounting(unittest.TestCase):
    def test_current_r18_qlora_source_omits_adamw8bit_decay(self):
        source = Path(__file__).resolve().parents[1] / "trainingday" / "r18_qlora.py"
        observed = inspect_adamw8bit_weight_decay(source)
        self.assertEqual(observed["status"], "UNDECLARED_DEFAULT_UNKNOWN")
        self.assertFalse(observed["runtime_effective_weight_decay_verified"])

    def test_matched_public_toy_still_cannot_authorize_physical_training(self):
        report = pair()
        self.assertEqual(report["status"], "HOLD")
        self.assertTrue(report["public_toy_objective_parity"])
        self.assertTrue(report["synthetic_schedule_matched"])
        self.assertFalse(report["physical_equal_time_attested"])
        self.assertFalse(report["authorizes_training"])
        self.assertFalse(report["proves_neural_weight_improvement"])
        self.assertIn("PHYSICAL_EQUAL_TIME_NOT_ATTESTED", report["hold_reasons"])

    def test_modulo_recycle_and_single_sweep_mismatch(self):
        report = pair(mezo_steps=(2.5, 2.5), qlora_steps=(5.0,))
        self.assertIn("CASE_EXPOSURE_MISMATCH", report["hold_reasons"])
        self.assertIn("REPEATED_EXAMPLES_IN_MEZO", report["hold_reasons"])
        self.assertEqual(report["mezo_accounting"]["optimizer_steps"], 2)
        self.assertEqual(report["qlora_accounting"]["optimizer_steps"], 1)
        self.assertEqual(report["mezo_accounting"]["example_presentations"], 16)
        self.assertEqual(report["qlora_accounting"]["example_presentations"], 8)
        self.assertEqual(report["mezo_accounting"]["forward_sample_tokens"], 128)
        self.assertEqual(report["qlora_accounting"]["backward_sample_tokens"], 32)

    def test_invalid_budget_and_modern_standby_fail_closed(self):
        report = pair(mezo_clock=clock(elapsed=19813.562, wall=19813.562,
                                      standby=19698.314, budget=531.0))
        self.assertIn("TIME_BUDGET_OVERRUN_INVALID", report["hold_reasons"])
        self.assertIn("STANDBY_INTEGRITY_INVALID", report["hold_reasons"])
        self.assertIn("OPTIMIZER_WINDOW_CLOCK_MISMATCH", report["hold_reasons"])

    def test_effective_decay_not_inferred_from_constructor(self):
        report = pair(decay=0.1)
        self.assertIn("OPTIMIZER_REGULARIZATION_MISMATCH", report["hold_reasons"])
        self.assertFalse(report["runtime_weight_decay_attested"])

    def test_adamw8bit_undeclared_default_is_not_zero(self):
        report = audit_fair_optimizer_pair(
            mezo_cases=cases(), qlora_cases=cases(),
            mezo_timings=ArmTimings((5.0,), 6.0),
            qlora_timings=ArmTimings((5.0,), 6.0),
            mezo_clock=clock(), qlora_clock=clock(),
            qlora_weight_decay_source={"status": "UNDECLARED_DEFAULT_UNKNOWN", "weight_decay": None},
            declared_controls=controls(), max_seq_length=8,
            tokenizer_sha256="d" * 64, corpus_sha256="e" * 64)
        self.assertIn("QLORA_WEIGHT_DECAY_DEFAULT_UNVERIFIED", report["hold_reasons"])
        self.assertIn("RUNTIME_EFFECTIVE_ADAMW8BIT_DECAY_UNATTESTED", report["hold_reasons"])

    def test_wall_monotonic_divergence_is_not_equal_time(self):
        report = pair(qlora_clock=clock(elapsed=5.0, wall=6.0,
                                        standby=0.0, budget=6.0))
        self.assertIn("WALL_MONOTONIC_CLOCK_DIVERGENCE", report["hold_reasons"])

    def test_control_violations_cannot_graduate(self):
        bad = DeclaredControls("a" * 64, "b" * 64, "c" * 64, "c" * 64, 0, 1)
        report = pair(control=bad)
        self.assertIn("ZERO_UPDATE_TENSOR_CHANGED", report["hold_reasons"])
        self.assertIn("CONTROL_LABEL_COMMITMENTS_NOT_DISTINCT", report["hold_reasons"])
        self.assertIn("CORRECT_LABEL_NO_OPTIMIZER_UPDATE", report["hold_reasons"])

    def test_differing_target_ids_propagate_to_accounting(self):
        right = list(cases())
        case = right[-1]
        right[-1] = ToyTokenCase(case.case_id, case.input_ids,
                                 case.labels[:-1] + (5,), case.target_logits)
        report = pair(right=tuple(right))
        self.assertIn("TARGET_TOKEN_ID_COMMITMENT_MISMATCH", report["hold_reasons"])


if __name__ == "__main__":
    unittest.main()
