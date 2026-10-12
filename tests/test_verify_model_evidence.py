import hashlib
import tempfile
import unittest
from pathlib import Path

from training.classify_model_evidence import verify_bundle


class VerifyBundleTests(unittest.TestCase):
    def test_verifies_declared_bytes_without_qualifying_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {}
            for name in ("dataset", "recipe", "run_log", "weights", "eval_manifest", "eval_results", "baseline_results"):
                path = root / (name + ".bin")
                path.write_bytes((name + "-bytes").encode())
                paths[name] = path
            evidence = {"base_revision": "b" * 40}
            for name, path in paths.items():
                evidence[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {"schema": "VERA_NEURAL_EVIDENCE_V1", "training_kind": "adapter_finetune",
                       "claims": {"weight_update": True, "artifact": True, "evaluation": True},
                       "evidence": evidence, "held_out": True, "independent_evaluator": True}
            result = verify_bundle(receipt, {key: str(value) for key, value in paths.items()})
            self.assertTrue(result["hashes_verified"])
            self.assertTrue(result["training_executed_verified"])
            self.assertFalse(result["model_qualified"])
            self.assertEqual(result["classification"], "EVALUATION_VERIFIED")

    def test_rejects_changed_weight_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.bin"
            path.write_bytes(b"actual")
            actual = hashlib.sha256(b"actual").hexdigest()
            receipt = {"schema": "VERA_NEURAL_EVIDENCE_V1", "training_kind": "adapter_finetune",
                       "claims": {"weight_update": True, "artifact": True, "evaluation": False},
                       "evidence": {"base_revision": "b" * 40, "dataset_sha256": actual,
                                    "recipe_sha256": actual, "run_log_sha256": actual,
                                    "weights_sha256": "0" * 64},
                       "held_out": False, "independent_evaluator": False}
            with self.assertRaisesRegex(ValueError, "weights digest mismatch"):
                verify_bundle(receipt, {"dataset": str(path), "recipe": str(path),
                                        "run_log": str(path), "weights": str(path)})

    def test_rejects_missing_required_file(self):
        receipt = {"schema": "VERA_NEURAL_EVIDENCE_V1", "training_kind": "adapter_finetune",
                   "claims": {"weight_update": True, "artifact": False, "evaluation": False},
                   "evidence": {"base_revision": "b" * 40, "dataset_sha256": "a" * 64,
                                "recipe_sha256": "a" * 64, "run_log_sha256": "a" * 64},
                   "held_out": False, "independent_evaluator": False}
        with self.assertRaisesRegex(ValueError, "missing dataset file"):
            verify_bundle(receipt, {})


if __name__ == "__main__":
    unittest.main()
