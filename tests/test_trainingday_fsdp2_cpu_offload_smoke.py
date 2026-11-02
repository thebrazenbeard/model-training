"""Dependency-free source boundaries for the optional synthetic GPU probe."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "trainingday" / "fsdp2_cpu_offload_smoke.py"

class FSDP2SmokeContractTests(unittest.TestCase):
    def test_source_parses_without_gpu_dependencies(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "main"
                            for n in tree.body))
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "digest"
                            for n in tree.body))

    def test_source_has_no_model_download_or_protected_training_binding(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        self.assertTrue({"torch", "tempfile", "hashlib"}.issubset(modules))
        self.assertTrue(modules.isdisjoint({"transformers", "huggingface_hub",
                                           "requests", "peft", "deepspeed"}))
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"qwen_weights_used": False', text)
        self.assertIn('"training_authorized": False', text)
        self.assertIn("CPUOffloadPolicy(pin_memory=False)", text)
        self.assertIn('"gradient_device": "cpu"', text)
        self.assertIn('"optimizer_state_device": "cpu"', text)
        self.assertIn('if any(g.device.type != "cpu" for g in grads):', text)
        self.assertIn('for v in optimizer_tensors)', text)
        self.assertIn("nn.Linear(64, 64)", text)
        self.assertIn("nn.Linear(64, 8)", text)
        self.assertIn("range(1, 4)", text)
        self.assertNotIn("from_pretrained(", text)
        self.assertNotIn("load_state_dict(", text)

    def test_entrypoint_is_explicit(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        self.assertIsInstance(tree.body[-1], ast.If)
        guard = tree.body[-1].test
        self.assertIsInstance(guard, ast.Compare)
        self.assertIsInstance(guard.left, ast.Name)
        self.assertEqual(guard.left.id, "__name__")

if __name__ == "__main__":
    unittest.main()
