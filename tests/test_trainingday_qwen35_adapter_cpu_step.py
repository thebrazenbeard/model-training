"""Guard the synthetic adapter-only CPU optimizer rehearsal boundary."""

from importlib.util import find_spec
import unittest

from trainingday.qwen35_hybrid_rank4_meta_budget import (
    HybridMetaBudgetError,
    rehearse_adapter_only_cpu_adamw,
)


@unittest.skipUnless(find_spec("torch") is not None, "CPU PyTorch available in hosted CI")
class AdapterOnlyCPURehearsalTests(unittest.TestCase):
    @staticmethod
    def tiny_meta_adapters():
        import torch

        return {f"model.layers.{idx}.lora_A.default.weight": torch.nn.Parameter(
            torch.empty((2, 3), device="meta", dtype=torch.float32))
            for idx in range(304)}

    def test_true_cpu_adamw_allocations_are_differentiated_from_official_training(self):
        report = rehearse_adapter_only_cpu_adamw(self.tiny_meta_adapters())
        elements = 304 * 6
        self.assertEqual(report["adapter_parameter_elements"], elements)
        for key in ("adapter_weight_allocated_tensor_bytes",
                    "adapter_gradient_allocated_tensor_bytes",
                    "adamw_exp_avg_allocated_tensor_bytes",
                    "adamw_exp_avg_sq_allocated_tensor_bytes"):
            self.assertEqual(report[key], 4 * elements)
        self.assertEqual(report["adamw_step_scalar_allocated_tensor_bytes"], 304 * 4)
        self.assertEqual(report["sum_allocated_tensor_bytes"], 16 * elements + 304 * 4)
        self.assertEqual(report["adapter_only_cpu_optimizer_steps_observed"], 1)
        self.assertIs(report["official_backbone_forward_or_backward"], False)
        self.assertIs(report["official_pretrained_weights_loaded"], False)
        self.assertIs(report["physical_training_authorized"], False)
        self.assertEqual(report["memory_fit"], "UNKNOWN")

    def test_incomplete_or_untrusted_meta_shape_inventory_fails_closed(self):
        import torch

        good = self.tiny_meta_adapters()
        for bad in (
            dict(list(good.items())[:-1]),
            {**dict(list(good.items())[1:]), "foreign_tensor": next(iter(good.values()))},
            {**good, "model.layers.0.lora_A.default.weight": torch.nn.Parameter(
                torch.zeros((2, 3), device="cpu"))},
            {**good, "model.layers.0.lora_A.default.weight": torch.nn.Parameter(
                torch.empty((2, 3), device="meta", dtype=torch.bfloat16))},
        ):
            with self.subTest(bad_count=len(bad)), self.assertRaises(HybridMetaBudgetError):
                rehearse_adapter_only_cpu_adamw(bad)


if __name__ == "__main__":
    unittest.main()
