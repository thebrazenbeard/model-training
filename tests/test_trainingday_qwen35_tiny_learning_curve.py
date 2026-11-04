"""Dependency-free falsifiers for a predeclared tiny learning curve."""

import unittest

from trainingday import qwen35_tiny_heldout_eval as baseline
from trainingday import qwen35_tiny_learning_curve as curve


class TinyQwenLearningCurveTests(unittest.TestCase):
    def test_predeclared_nonoverlapping_seed_and_step_contract(self):
        self.assertEqual(curve.SEEDS, (35060, 35061))
        self.assertFalse(set(curve.SEEDS) & set(baseline.SEEDS))
        self.assertEqual(curve.CHECKPOINT_STEPS, (0, 4, 16))
        self.assertEqual(curve.BATCH_SIZE, 4)

    def test_all_optimizer_pairs_are_from_train_not_heldout(self):
        for seed in curve.SEEDS:
            train, heldout = baseline._pairs(seed)
            assigned = curve._assigned_pairs(seed, train)
            self.assertEqual(len(assigned), 64)
            self.assertTrue(set(assigned) <= set(train))
            self.assertFalse(set(assigned) & set(heldout))
            self.assertEqual(set(assigned), set(train))
            self.assertEqual(assigned, curve._assigned_pairs(seed, train))

    def test_feedback_ablation_keeps_exact_target_multiset(self):
        for seed in curve.SEEDS:
            train, _ = baseline._pairs(seed)
            assigned = curve._assigned_pairs(seed, train)
            truth, changed_truth = curve._assigned_answers(seed, assigned, shuffled=False)
            shuffled, changed = curve._assigned_answers(seed, assigned, shuffled=True)
            self.assertEqual(changed_truth, 0)
            self.assertGreaterEqual(changed, len(assigned) // 2)
            self.assertEqual(sorted(truth), sorted(shuffled))
            stable = {}
            for pair, answer in zip(assigned, shuffled):
                if pair in stable:
                    self.assertEqual(stable[pair], answer,
                                     "repeated train pair received inconsistent control labels")
                stable[pair] = answer
            self.assertEqual(shuffled,
                             curve._assigned_answers(seed, assigned, shuffled=True)[0])

    def test_scheduled_domain_includes_future_optimizer_pairs_at_four_steps(self):
        for seed in curve.SEEDS:
            train, _ = baseline._pairs(seed)
            assigned = curve._assigned_pairs(seed, train)
            # The fixed full-curve domain is not the 16 examples consumed by
            # the 4-step checkpoint. Do not label its NLL as already-seen.
            self.assertGreater(len(set(assigned)), curve.CHECKPOINT_STEPS[1] * curve.BATCH_SIZE)
            self.assertEqual(len(set(assigned)), len(train))

    def test_invalid_training_source_fails_closed(self):
        with self.assertRaises(AssertionError):
            curve._assigned_pairs(35060, [(1, 2)] * 30)
        with self.assertRaises(AssertionError):
            curve._assigned_pairs(35060, [(1, 2)])


if __name__ == "__main__":
    unittest.main()
