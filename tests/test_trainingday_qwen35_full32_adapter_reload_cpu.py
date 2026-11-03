"""Dependency-free guards for the hosted full32 PEFT adapter persistence run."""

import ast
from pathlib import Path
import unittest

from trainingday import qwen35_full32_adapter_reload_cpu as reload


class Full32AdapterReloadSourceTests(unittest.TestCase):
    def test_32_layer_contract_and_explicit_targets(self):
        self.assertEqual(len(reload.smoke.LAYER_TYPES), 32)
        self.assertEqual(reload.smoke.LAYER_TYPES.count("linear_attention"), 24)
        self.assertEqual(reload.smoke.LAYER_TYPES.count("full_attention"), 8)
        self.assertEqual(len(set(reload.smoke.expected_target_modules())), 152)

    def test_separate_process_and_exact_adapter_format_are_required(self):
        path = Path(reload.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("subprocess.run", source)
        self.assertIn("PeftModel.from_pretrained", source)
        self.assertIn("model.save_pretrained", source)
        self.assertIn("safe_serialization=True", source)
        self.assertIn("adapter_model.safetensors", source)
        self.assertIn("adapter_config.json", source)
        self.assertIn("if trained_logits_sha == base_logits_sha:", source)
        self.assertIn('"missing-adapter"', source)
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "_reload_in_child"
                            for n in tree.body))

    def test_no_pretrained_base_or_GPU_operator_in_source(self):
        source = Path(reload.__file__).read_text(encoding="utf-8")
        self.assertNotIn("Qwen3_5ForCausalLM.from_pretrained", source)
        self.assertNotIn(".cuda()", source)
        self.assertNotIn('to("cuda")', source)
        self.assertIn('HF_HUB_OFFLINE', source)
        self.assertIn('TRAININGDAY_QWEN35_TINY_FULL32_ADAPTER_RELOAD_CPU_V1', source)


if __name__ == "__main__":
    unittest.main()
