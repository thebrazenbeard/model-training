"""Independent, stdlib-only guards around the expensive public CPU probe."""

import unittest

from trainingday.qwen35_tiny_gguf_e2e_cpu import (
    SYNTHETIC_BASE, assert_adapter_lineage, assert_model_architecture,
)


GOOD_CONFIG = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForCausalLM"],
    "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                    "full_attention"] * 8,
}


class TinyGGUFLineageTests(unittest.TestCase):
    def test_exact_synthetic_adapter_base_passes(self):
        assert_adapter_lineage({"base_model_name_or_path": SYNTHETIC_BASE})

    def test_mismatched_qwen25_adapter_base_rejected(self):
        with self.assertRaisesRegex(ValueError, "base lineage mismatch"):
            assert_adapter_lineage({"base_model_name_or_path":
                                   "Qwen/Qwen2.5-7B-Instruct"})

    def test_missing_adapter_base_rejected(self):
        with self.assertRaises(ValueError):
            assert_adapter_lineage({})

    def test_exact_hybrid_architecture_passes(self):
        assert_model_architecture(GOOD_CONFIG)

    def test_wrong_qwen25_architecture_rejected(self):
        with self.assertRaisesRegex(ValueError, "not Qwen3.5"):
            assert_model_architecture({**GOOD_CONFIG, "model_type": "qwen2"})

    def test_missing_gdn_layer_rejected(self):
        invalid = GOOD_CONFIG["layer_types"].copy()
        invalid[0] = "full_attention"
        with self.assertRaisesRegex(ValueError, "32-layer"):
            assert_model_architecture({**GOOD_CONFIG, "layer_types": invalid})


if __name__ == "__main__":
    unittest.main()
