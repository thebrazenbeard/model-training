"""Dependency-free guards for the hosted full32 PEFT adapter persistence run."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from trainingday import qwen35_full32_adapter_reload_cpu as reload


class Full32AdapterReloadSourceTests(unittest.TestCase):
    def test_adapter_a_mutation_changes_full_tensor_witness(self):
        class Tensor:
            dtype = np.uint8

            def __init__(self, data):
                self.data = np.asarray(data, dtype=np.uint8)

            def detach(self):
                return self

            def to(self, **_):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return self.data

        class Model:
            def __init__(self, a, b):
                self.parameters = {"layer.lora_A.default.weight": Tensor(a),
                                   "layer.lora_B.default.weight": Tensor(b)}

            def named_parameters(self):
                return self.parameters.items()

        with patch.object(reload.smoke, "TARGET_COUNT", 1):
            before, count = reload._adapter_ab_hash(Model([1, 2], [7, 8]))
            after, after_count = reload._adapter_ab_hash(Model([9, 2], [7, 8]))
        self.assertEqual((count, after_count), (2, 2))
        self.assertNotEqual(before, after)  # LoRA-B held constant.

    def test_persisted_a_and_config_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            model = artifact / "adapter_model.safetensors"
            config = artifact / "adapter_config.json"
            model.write_bytes(b"synthetic header|lora_A=abcd|lora_B=efgh")
            config.write_text('{"r": 4, "lora_alpha": 8}', encoding="utf-8")
            baseline = reload._artifact_hashes(artifact)

            model.write_bytes(b"synthetic header|lora_A=zbcd|lora_B=efgh")
            self.assertEqual(len(model.read_bytes()), len(b"synthetic header|lora_A=abcd|lora_B=efgh"))
            with self.assertRaisesRegex(AssertionError, "LoRA-A/B"):
                reload._assert_artifact_hashes(baseline, reload._artifact_hashes(artifact))
            model.write_bytes(b"synthetic header|lora_A=abcd|lora_B=efgh")

            config.write_text('{"r": 8, "lora_alpha": 8}', encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "adapter config"):
                reload._assert_artifact_hashes(baseline, reload._artifact_hashes(artifact))

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
