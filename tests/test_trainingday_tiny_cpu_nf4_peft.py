"""Hosted CPU-only integration proof; absence of bitsandbytes is a hard failure."""

import json
import unittest


class TinyCpuNf4PeftTests(unittest.TestCase):
    def test_packed_nf4_peft_forward_backward_and_adamw_step(self):
        from trainingday.tiny_cpu_nf4_peft import run_tiny_cpu_nf4_peft

        result = run_tiny_cpu_nf4_peft()
        print("TINY_CPU_NF4_PEFT_EVIDENCE=" + json.dumps(result, sort_keys=True))
        self.assertEqual(result["status"], "TINY_SYNTHETIC_CPU_BNB_NF4_PEFT_STEP_ONLY")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["base_quantization"], "packed_nf4_uint8")
        self.assertEqual(result["lora_trainable_tensors"], 2)
        self.assertEqual(result["lora_a_b_updated"], ["A", "B"])
        self.assertEqual(result["optimizer_steps_observed"], 1)
        self.assertFalse(result["official_qwen_weights_loaded"])
        self.assertFalse(result["official_qwen_forward_or_backward"])
        self.assertFalse(result["cuda_used"])
        self.assertFalse(result["physical_training_authorized"])
        self.assertEqual(result["qwen35_4gb_hardware_fit"], "UNKNOWN")
        self.assertLess(result["max_rss_bytes"], 2 * 1024**3)
        self.assertLess(result["elapsed_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
