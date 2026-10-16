"""Conditional likelihood diagnostics: mask/alignment, controls, and evidence ceilings."""
import importlib.util
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "successor/experiments/compare_r12_conditional_margins.py"
mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("r12_margin", SOURCE))
mod.__spec__.loader.exec_module(mod)


class ConditionalLikelihoodTests(unittest.TestCase):
    def test_scores_only_continuation_with_shifted_logits(self):
        import torch
        seq = torch.tensor([[2, 1, 3, 0]])
        # gold at t=2 is 3, predicted by position 1; at t=3 gold 0 by position 2.
        logits = torch.zeros(1, 4, 4)
        logits[0, 1, 3] = 2.0
        logits[0, 2, 0] = 3.0
        result = mod.mean_continuation_logprob(logits, seq, prefix_len=2)
        wanted1 = 2.0 - math.log(math.exp(2.0)+3)
        wanted2 = 3.0 - math.log(math.exp(3.0)+3)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertAlmostEqual(result["sum_logprob"], wanted1+wanted2, places=5)
        self.assertAlmostEqual(result["mean_logprob_per_token"], (wanted1+wanted2)/2, places=5)

    def test_reject_wrong_alignment_and_nonfinite(self):
        import torch
        ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "boundary"):
            mod.mean_continuation_logprob(torch.zeros(1, 3, 5), ids, prefix_len=3)
        with self.assertRaisesRegex(ValueError, "lengths"):
            mod.mean_continuation_logprob(torch.zeros(1, 4, 5), ids, prefix_len=1)
        logits = torch.zeros(1, 3, 5)
        logits[0, 1, 3] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            mod.mean_continuation_logprob(logits, ids, prefix_len=2)

    def test_margin_is_directional_and_normalized(self):
        good = {"mean_logprob_per_token": -1.2, "completion_tokens": 5}
        bad = {"mean_logprob_per_token": -2.5, "completion_tokens": 8}
        self.assertAlmostEqual(mod.margin(good, bad), 1.3)
        self.assertAlmostEqual(mod.margin(bad, good), -1.3)

    def test_three_case_no_system_diagnostic_only(self):
        self.assertEqual(len(mod.CHOICES), 3)
        self.assertEqual({x["axis"] for x in mod.CHOICES.values()},
                         {"identity", "agency", "epistemics"})
        for value in mod.CHOICES.values():
            self.assertNotEqual(value["preferred"], value["counter"])
            self.assertTrue(value["preferred"])
        text = SOURCE.read_text(encoding="utf8")
        self.assertIn('with model.disable_adapter()', text)
        self.assertIn('OUT.open("x"', text)
        self.assertIn('enable_thinking=False', text)
        self.assertNotIn("trainer.train(", text)

    def test_reject_existing_output_without_weight_access(self):
        with TemporaryDirectory() as td:
            output = Path(td) / "exists.json"
            output.write_text("{}")
            with patch.object(mod, "OUT", output):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    mod.validate_inputs()


if __name__ == "__main__":
    unittest.main()
