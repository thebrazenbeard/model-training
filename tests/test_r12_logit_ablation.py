"""R12 three-arm read-only differential scoring source regression checks."""
import importlib.util
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "successor/experiments/ablate_r12_voice_logits_20261015.py"
MODULE = importlib.util.spec_from_file_location("ablate_r12", SOURCE)
probe = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(probe)


class R12LogitAblationTests(unittest.TestCase):
    def test_nonzero_delta_and_no_top_token_flip_are_distinct(self):
        import torch
        r = probe.describe_delta(torch.tensor([2.1, 1.0]), torch.tensor([2.0, 1.0]))
        self.assertEqual(r["changed_logit_elements"], 1)
        self.assertFalse(r["top_token_differs"])
        self.assertAlmostEqual(r["max_abs_logit_delta"], 0.1, places=5)
    def test_ordered_top_token_flip(self):
        import torch
        r = probe.describe_delta(torch.tensor([0., 2.]), torch.tensor([3., 0.]))
        self.assertTrue(r["top_token_differs"])
        self.assertAlmostEqual(r["l2_logit_delta"], math.sqrt(13.), places=5)
    def test_bad_tensors_fail_closed(self):
        import torch
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            probe.describe_delta(torch.tensor([float("nan")]), torch.tensor([0.]))
        with self.assertRaisesRegex(ValueError, "shape"):
            probe.describe_delta(torch.tensor([1.,2.]), torch.tensor([1.]))
    def test_no_optimizer_and_explicit_base_ablation(self):
        s = SOURCE.read_text(encoding="utf8")
        self.assertIn("with model.disable_adapter()", s)
        self.assertIn("model.set_adapter(adapter)", s)
        self.assertIn('OUTPUT.open("x"', s)
        self.assertNotIn("trainer.train(", s)
