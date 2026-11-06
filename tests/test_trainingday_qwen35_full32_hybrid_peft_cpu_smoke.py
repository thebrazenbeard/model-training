"""Dependency-free guard: dynamic full32 hybrid test runs on hosted CPU."""

import ast
import inspect
import unittest

from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke


class Full32HybridPEFTSourceTests(unittest.TestCase):
    def test_exact_target_inventory_and_layer_pattern(self):
        self.assertEqual(len(smoke.LAYER_TYPES), 32)
        self.assertEqual(smoke.LAYER_TYPES.count("linear_attention"), 24)
        self.assertEqual(smoke.LAYER_TYPES.count("full_attention"), 8)
        self.assertEqual(smoke.LAYER_TYPES, (
            "linear_attention", "linear_attention", "linear_attention", "full_attention") * 8)
        targets = smoke.expected_target_modules()
        self.assertEqual(len(targets), 152)
        self.assertEqual(len(targets), len(set(targets)))
        self.assertTrue(all(t.startswith("model.layers.") for t in targets))
        self.assertEqual(len([t for t in targets if ".linear_attn." in t]), 120)
        self.assertEqual(len([t for t in targets if ".self_attn." in t]), 32)
        self.assertFalse(any(t.startswith("mtp.") or ".visual." in t for t in targets))

    def test_source_cannot_load_pretrained_weights_or_cuda(self):
        calls = [node for node in ast.walk(ast.parse(inspect.getsource(smoke.run)))
                 if isinstance(node, ast.Call)]
        names = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in calls if isinstance(n.func, (ast.Name, ast.Attribute))]
        self.assertIn("Qwen3_5ForCausalLM", names)
        self.assertIn("get_peft_model", names)
        self.assertIn("backward", names)
        self.assertIn("step", names)
        self.assertNotIn("from_pretrained", names)
        self.assertNotIn("cuda", names)

    def test_source_witnesses_each_distinct_lora_b_gradient_and_step_delta(self):
        src = inspect.getsource(smoke.run)
        for item in (
            "b_by_target", "len({id(p) for p in b_by_target.values()})",
            "before_by_target", "per_target_gradient_sums",
            "zero_gradient_targets", "unchanged_targets",
            "lora_b_targets_with_nonzero_gradients",
            "smallest_target_gradient_sum", "lora_b_targets_changed_after_step",
        ):
            self.assertIn(item, src)

    def test_source_reports_unknown_fit_and_denies_authority(self):
        src = inspect.getsource(smoke.run)
        for item in ("training_authorized", "four_gb_gpu_fit", "UNKNOWN",
                     "official_model_exercised", "cuda_runtime_initialized",
                     "sampled_base_embedding_and_qkv_unchanged"):
            self.assertIn(item, src)
        self.assertNotIn("frozen_base_parameters_verified", src)


if __name__ == "__main__":
    unittest.main()
