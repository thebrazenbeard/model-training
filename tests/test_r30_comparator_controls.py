"""R30 public-only matched-control schedule adversaries; never a GPU gate."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from trainingday.r29_optimizer_fairness import ToyTokenCase
import trainingday.r30_comparator_controls as gate


def cases():
    logits = ((0.1, 0.0, 0.0, 0.5, 1.0, 0.9, 0.0, 0.0),) * 2
    return (
        ToyTokenCase("sample-a", (1, 2, 3, 4), (-100, -100, 3, 4), logits),
        ToyTokenCase("sample-b", (1, 2, 3, 4), (-100, -100, 3, 4), logits),
    )


def quartet():
    correct = cases()
    changed = list(correct)
    changed[0] = replace(changed[0], input_ids=(1, 2, 3, 5),
                         labels=(-100, -100, 3, 5))
    def arm(role, rows, *, updates, after, presentations=()):
        return gate.ControlArm(
            role=role, optimizer_method="QLORA", cases=rows,
            presented_case_ids=presentations, optimizer_updates=updates,
            initial_adapter_sha256="a" * 64, final_adapter_sha256=after)
    return (
        arm("FROZEN", correct, updates=0, after="a" * 64),
        arm("ZERO_UPDATE", correct, updates=0, after="a" * 64),
        arm("CORRECT_LABEL", correct, updates=1, after="b" * 64,
            presentations=("sample-a", "sample-b")),
        arm("CORRUPTED_LABEL", tuple(changed), updates=1, after="c" * 64,
            presentations=("sample-a", "sample-b")),
    )


class ControlQuartetTests(unittest.TestCase):
    def test_valid_public_quartet_is_still_hold_and_not_real_evidence(self):
        report = gate.assess_control_quartet(quartet())
        self.assertTrue(report["synthetic_control_conditions_satisfied"])
        self.assertEqual(report["hold_reasons"], [])
        self.assertEqual(report["status"], "HOLD")
        self.assertFalse(report["neural_tensor_update_verified"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["independent_control_custody_verified"])
        self.assertEqual(report["correct_presentations"], 2)
        self.assertEqual(report["corrupt_presentations"], 2)

    def test_same_case_ids_but_changed_prompt_context_blocks(self):
        arms = list(quartet())
        corrupt = list(arms[3].cases)
        corrupt[1] = replace(corrupt[1], input_ids=(7, 2, 3, 4))
        arms[3] = replace(arms[3], cases=tuple(corrupt))
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("PROMPT_CONTEXT_MISMATCH", report["hold_reasons"])

    def test_same_counts_but_different_presentation_order_blocks(self):
        arms = list(quartet())
        arms[3] = replace(arms[3], presented_case_ids=("sample-b", "sample-a"))
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("CONTROL_EXPOSURE_ORDER_MISMATCH", report["hold_reasons"])

    def test_corrupted_labels_must_change_supervised_tokens(self):
        arms = list(quartet())
        arms[3] = replace(arms[3], cases=cases())
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("CORRUPTED_LABEL_NOT_DISTINCT", report["hold_reasons"])

    def test_mask_position_mismatch_blocks_without_treating_as_augmentation(self):
        arms = list(quartet())
        corrupt = list(arms[3].cases)
        corrupt[0] = replace(corrupt[0], labels=(-100, 2, 3, 5),
                             target_logits=corrupt[0].target_logits + (corrupt[0].target_logits[0],))
        arms[3] = replace(arms[3], cases=tuple(corrupt))
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("TARGET_MASK_MISMATCH", report["hold_reasons"])

    def test_zero_or_frozen_checkpoint_change_blocks(self):
        arms = list(quartet())
        arms[1] = replace(arms[1], final_adapter_sha256="d" * 64)
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("ZERO_UPDATE_CONTROL_CHANGED", report["hold_reasons"])

    def test_frozen_logit_snapshot_mismatch_blocks_toy_objective_comparison(self):
        arms = list(quartet())
        altered = list(arms[1].cases)
        first = altered[0]
        altered[0] = replace(first, target_logits=((10.0,) * 8,) + first.target_logits[1:])
        arms[1] = replace(arms[1], cases=tuple(altered))
        self.assertIn("FROZEN_LOGIT_SNAPSHOT_MISMATCH",
                      gate.assess_control_quartet(tuple(arms))["hold_reasons"])

    def test_same_prediction_prefix_must_have_same_frozen_logits(self):
        arms = list(quartet())
        changed = list(arms[3].cases)
        original = changed[0]
        # Corruption can change a target label without changing its causal
        # prefix. A single frozen model cannot supply different public toy
        # logits for that identical prediction context.
        changed[0] = replace(original, input_ids=arms[2].cases[0].input_ids,
                             target_logits=((9.0,) * 8,) + original.target_logits[1:])
        arms[3] = replace(arms[3], cases=tuple(changed))
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("FROZEN_PREFIX_LOGIT_MISMATCH", report["hold_reasons"])
        self.assertFalse(report["synthetic_control_conditions_satisfied"])

    def test_changed_future_input_cannot_change_earlier_prefix_logits(self):
        arms = list(quartet())
        changed = list(arms[3].cases)
        original = changed[0]
        # Only a later completion input differs, so the first supervised
        # target still has an identical causal prefix and logit prediction.
        changed[0] = replace(original, target_logits=((8.0,) * 8,) + original.target_logits[1:])
        arms[3] = replace(arms[3], cases=tuple(changed))
        self.assertIn("FROZEN_PREFIX_LOGIT_MISMATCH",
                      gate.assess_control_quartet(tuple(arms))["hold_reasons"])

    def test_logits_may_diverge_after_completion_input_prefix_actually_changes(self):
        arms = list(quartet())
        for arm_i, arm in enumerate(arms):
            extended = []
            for case in arm.cases:
                logits = case.target_logits + (case.target_logits[-1],)
                extended.append(replace(case, input_ids=case.input_ids + (6,),
                                        labels=case.labels + (6,), target_logits=logits))
            arms[arm_i] = replace(arm, cases=tuple(extended))
        corrupt = list(arms[3].cases)
        first = corrupt[0]
        new_logits = first.target_logits[:-1] + ((9.0,) * 8,)
        corrupt[0] = replace(first, target_logits=new_logits)
        arms[3] = replace(arms[3], cases=tuple(corrupt))
        report = gate.assess_control_quartet(tuple(arms))
        self.assertNotIn("FROZEN_PREFIX_LOGIT_MISMATCH", report["hold_reasons"])
        self.assertTrue(report["synthetic_control_conditions_satisfied"])

    def test_candidate_initialization_must_share_bytes(self):
        arms = list(quartet())
        arms[2] = replace(arms[2], initial_adapter_sha256="d" * 64)
        self.assertIn("INITIAL_ADAPTER_MISMATCH",
                      gate.assess_control_quartet(tuple(arms))["hold_reasons"])

    def test_correct_and_corrupt_must_both_have_nonzero_updates_and_change(self):
        arms = list(quartet())
        arms[2] = replace(arms[2], optimizer_updates=0,
                          final_adapter_sha256="a" * 64)
        report = gate.assess_control_quartet(tuple(arms))
        self.assertIn("LABELED_CONTROL_NO_UPDATES", report["hold_reasons"])
        self.assertIn("LABELED_CONTROL_TENSOR_UNCHANGED", report["hold_reasons"])

    def test_wrong_arm_order_and_ambiguous_method_are_rejected(self):
        arms = quartet()
        with self.assertRaises(ValueError):
            gate.assess_control_quartet((arms[1], arms[0], arms[2], arms[3]))
        with self.assertRaises(ValueError):
            gate.assess_control_quartet(arms[:2] +
                  (replace(arms[2], optimizer_method="MEZO"),) + arms[3:])

    def test_same_ids_different_row_order_cannot_pass(self):
        arms = list(quartet())
        arms[3] = replace(arms[3], cases=tuple(reversed(arms[3].cases)))
        self.assertIn("CASE_ORDER_MISMATCH",
                      gate.assess_control_quartet(tuple(arms))["hold_reasons"])

    def test_private_snapshot_cannot_be_rebound_by_caller_after_validation(self):
        arms = quartet()
        original = gate._validate_cases

        def mutate_after_initial_validation(private_cases):
            object.__setattr__(arms[3].cases[0], "labels", (-100, -100, 3, 4))
            return original(private_cases)

        with patch.object(gate, "_validate_cases", side_effect=mutate_after_initial_validation):
            result = gate.assess_control_quartet(arms)
        self.assertTrue(result["synthetic_control_conditions_satisfied"])
        self.assertFalse(result["training_authorized"])


if __name__ == "__main__":
    unittest.main()
