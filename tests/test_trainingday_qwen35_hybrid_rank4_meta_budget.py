"""Zero-GPU checks for the official-text hybrid LoRA memory probe."""

import unittest

from trainingday.qwen35_hybrid_rank4_meta_budget import (
    HybridMetaBudgetError,
    bound_rank4_adapter_memory,
    bound_observed_meta_fp32_scenario,
    expected_rank4_elements,
    hybrid_projection_targets,
)


class HybridMetaBudgetTests(unittest.TestCase):
    def test_thirty_two_layers_yield_exact_namespace_qualified_152_targets(self):
        kinds = ["linear_attention"] * 24 + ["full_attention"] * 8
        targets = hybrid_projection_targets(kinds)
        self.assertEqual(len(targets), 152)
        self.assertEqual(len(set(targets)), 152)
        self.assertIn("model.layers.0.linear_attn.in_proj_qkv", targets)
        self.assertIn("model.layers.24.self_attn.o_proj", targets)
        self.assertFalse(any(name.startswith("mtp.") for name in targets))

    def test_invalid_layer_contract_fails_closed(self):
        kinds = ["linear_attention"] * 24 + ["full_attention"] * 8
        for bad in (kinds[:-1], ["full_attention"] * 32,
                    kinds[:12] + ["surprise"] + kinds[13:],
                    tuple(kinds), kinds[:10] + [True] + kinds[11:]):
            with self.subTest(bad=str(bad)[:25]), self.assertRaises(HybridMetaBudgetError):
                hybrid_projection_targets(bad)

    def test_adapter_training_state_is_optimistic_lower_bound(self):
        actual = bound_rank4_adapter_memory(adapter_elements=1000,
                                            a_b_tensor_count=304,
                                            target_module_count=152)
        self.assertEqual(actual["optimistic_bf16_adapter_weight_bytes"], 2000)
        self.assertEqual(actual["optimistic_bf16_gradient_bytes"], 2000)
        self.assertEqual(actual["optimistic_fp32_adamw_moment_bytes"], 8000)
        self.assertEqual(actual["optimistic_training_state_bytes"], 12000)
        self.assertEqual(actual["physical_fit"], "UNKNOWN")
        self.assertFalse(actual["training_authorized"])

    def test_observed_meta_fp32_dtype_scenario_is_separately_labeled(self):
        scenario = bound_observed_meta_fp32_scenario(
            adapter_elements=1000, observed_meta_lora_dtypes=["torch.float32"])
        self.assertEqual(scenario["observed_dtype_fp32_adapter_weight_bytes"], 4000)
        self.assertEqual(scenario["observed_dtype_fp32_gradient_bytes"], 4000)
        self.assertEqual(scenario["observed_dtype_fp32_adamw_moment_bytes"], 8000)
        self.assertEqual(scenario["observed_dtype_fp32_training_state_bytes"], 16000)
        self.assertEqual(scenario["scenario_basis"], "OBSERVED_META_LORA_DTYPE_FP32_ONLY")
        self.assertFalse(scenario["optimizer_allocation_observed"])

    def test_observed_meta_fp32_scenario_rejects_dtype_drift(self):
        for elements, dtypes in (
            (True, ["torch.float32"]),
            (0, ["torch.float32"]),
            (1000, ["torch.bfloat16"]),
            (1000, ["torch.float32", "torch.bfloat16"]),
            (1000, []),
            (1000, "torch.float32"),
        ):
            with self.subTest(inputs=(elements, dtypes)), self.assertRaises(HybridMetaBudgetError):
                bound_observed_meta_fp32_scenario(
                    adapter_elements=elements, observed_meta_lora_dtypes=dtypes)

    def test_independent_shape_arithmetic_crosscheck(self):
        self.assertEqual(expected_rank4_elements([(2, 3)] * 152), 3040)
        self.assertEqual(expected_rank4_elements([(2, 3)] * 151 + [(5, 7)]), 3068)
        for bad in ([(2, 3)] * 151, [(2, 3)] * 151 + [(0, 7)],
                    [(2, 3)] * 151 + [(True, 7)],
                    [(2, 3)] * 151 + [(-1, 7)]):
            with self.subTest(bad=bad[-1]), self.assertRaises(HybridMetaBudgetError):
                expected_rank4_elements(bad)

    def test_bad_inputs_do_not_establish_false_memory_result(self):
        for elems, tensors, modules in ((True, 304, 152), (100, True, 152),
                                        (100, 304, True), (0, 304, 152),
                                        (100, 303, 152), (100, 304, 151)):
            with self.subTest(inputs=(elems, tensors, modules)), self.assertRaises(HybridMetaBudgetError):
                bound_rank4_adapter_memory(adapter_elements=elems,
                                           a_b_tensor_count=tensors,
                                           target_module_count=modules)


if __name__ == "__main__":
    unittest.main()
