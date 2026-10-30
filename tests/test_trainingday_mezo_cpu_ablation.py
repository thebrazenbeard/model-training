"""CPU-only forward-only MeZO vs AdamW stand-in safeguards.

This does NOT prove official Qwen3.5-4B adapter effectiveness.
"""
import unittest
import torch

from trainingday.mezo_cpu_ablation import zo_update, run_arm, make_data


class MezoPilotTests(unittest.TestCase):
    def test_two_sided_quadratic_update_uses_no_backward_gradient(self):
        x = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float32))
        loss = lambda: (x ** 2).sum()
        value = zo_update([x], loss, epsilon=1e-3, lr=0.1, seed=4,
                          perturbations=[torch.ones_like(x)])
        self.assertAlmostEqual(float(x.item()), 1.2, places=3)
        self.assertIsNone(x.grad)
        self.assertEqual(value["forward_evaluations"], 2)

    def test_same_initial_weights_across_zero_correct_corrupt(self):
        zero = run_arm(method="mezo", mode="zero", steps=0, seed=29037)
        correct = run_arm(method="mezo", mode="correct", steps=5, seed=29037)
        corrupt = run_arm(method="mezo", mode="corrupt", steps=5, seed=29037)
        self.assertEqual(zero["initial_sha256"], correct["initial_sha256"])
        self.assertEqual(zero["initial_sha256"], corrupt["initial_sha256"])
        self.assertEqual(zero["initial_sha256"], zero["final_sha256"])
        self.assertNotEqual(correct["initial_sha256"], correct["final_sha256"])
        self.assertNotEqual(corrupt["initial_sha256"], corrupt["final_sha256"])
        self.assertEqual(correct["heldout_sha256"], corrupt["heldout_sha256"])
        self.assertFalse(correct["official_4b_model_trained"])

    def test_adamw_control_updates_same_small_adapter(self):
        candidate = run_arm(method="adamw", mode="correct", steps=5, seed=29037)
        self.assertNotEqual(candidate["initial_sha256"], candidate["final_sha256"])
        self.assertEqual(candidate["optimizer_steps"], 5)
        self.assertGreater(candidate["forward_evaluations"], 0)

    def test_holdout_is_not_training_data(self):
        a = make_data(seed=29037, count=48)
        b = make_data(seed=29038, count=32)
        self.assertFalse({tuple(x) for x in a[0].tolist()} &
                         {tuple(x) for x in b[0].tolist()})


if __name__ == "__main__":
    unittest.main()
