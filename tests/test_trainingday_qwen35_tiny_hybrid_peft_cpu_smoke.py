"""Dependency-free source guard for the separately hosted tiny hybrid test.

The genuine PyTorch/Transformers/PEFT optimizer proof runs in GitHub Actions.
These checks cannot substitute for it.
"""

import ast
import inspect
import unittest

from trainingday import qwen35_tiny_hybrid_peft_cpu_smoke as smoke


class QwenTinyDynamicSourceTests(unittest.TestCase):
    def test_two_hybrid_layer_types_and_nine_unique_targets(self):
        self.assertEqual(smoke.LAYER_KINDS, ("linear_attention", "full_attention"))
        targets = smoke._targets()
        self.assertEqual(len(targets), 9)
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(len([t for t in targets if ".linear_attn." in t]), 5)
        self.assertEqual(len([t for t in targets if ".self_attn." in t]), 4)
        self.assertTrue(all(t.startswith("model.layers.") for t in targets))
        self.assertFalse(any(t.startswith("mtp.") or "visual" in t for t in targets))

    def test_smoke_never_calls_pretrained_weights_or_cuda(self):
        tree = ast.parse(inspect.getsource(smoke.run))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in calls if isinstance(n.func, (ast.Attribute, ast.Name))]
        self.assertIn("Qwen3_5ForCausalLM", names)
        self.assertIn("get_peft_model", names)
        self.assertIn("step", names)
        self.assertNotIn("from_pretrained", names)
        self.assertNotIn("cuda", names)

    def test_output_cannot_imply_authority_or_official_weights(self):
        source = inspect.getsource(smoke.run)
        for literal in ("training_authorized", "official_model_exercised",
                        "RANDOM_FROM_TINY_CONFIG_NOT_OFFICIAL_WEIGHTS",
                        "four_gb_gpu_fit", "UNKNOWN"):
            self.assertIn(literal, source)


if __name__ == "__main__":
    unittest.main()
