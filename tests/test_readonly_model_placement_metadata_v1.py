"""Metadata-only disk inspection never constructs a model runtime."""
import os
from pathlib import Path
import unittest
from inspect_model_placement_v1 import MODELS,inspect

class GGUFMetadataOnlyTests(unittest.TestCase):
    def setUp(self):
        path=os.environ.get("VERA_MODELS_ROOT")
        if not path or not Path(path).is_dir():
            self.skipTest("Lappy D: model store unavailable")
        self.root=Path(path)

    def test_qwen3_4b_declared_layers_and_kv(self):
        result=inspect(self.root/MODELS["qwen3_4b_ollama"])
        self.assertEqual(result["gguf_architecture"],"qwen3")
        self.assertEqual(result["declared_layers"],36)
        self.assertEqual(result["scenarios"][1]["fp16_kv_mib"],576)
        self.assertIsNone(result["observed_backend_offload_layers"])
        self.assertFalse(result["provider_admitted"])

    def test_glm_gguf_quantizations_not_full_gpu_at_4k(self):
        for k in ("glm24b_iq2xs","glm24b_q4km"):
            result=inspect(self.root/MODELS[k])
            self.assertEqual(result["declared_layers"],40)
            self.assertFalse(result["scenarios"][1]["full_weight_with_reserved_vram_budget"])
            self.assertEqual(result["scenarios"][1]["fp16_kv_mib"],640)

    def test_qwen35_4b_q5ks_is_not_identical_qwen3(self):
        meta=inspect(self.root/MODELS["qwen35_4b_q5ks"])
        self.assertEqual(meta["gguf_architecture"],"qwen35")
        self.assertEqual(meta["declared_layers"],33)
        self.assertEqual(meta["scenarios"][1]["fp16_kv_mib"],528)
