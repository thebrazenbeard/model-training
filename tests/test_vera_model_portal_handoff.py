"""Adversarial zero-network tests for Vera model -> P.O.R.T.A.L./VeraOS candidate handoff.

The small GGUF header is a fixture, not a trained model or runtime qualification.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "training" / "verify_vera_model_handoff.py"
spec = importlib.util.spec_from_file_location("vera_handoff", MODULE_PATH)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class TrainingContractTests(unittest.TestCase):
    def test_target_is_neural_training_not_bootcamp_capsule(self):
        cfg = json.loads(
            (ROOT / "architecture" / "VERA_MODEL_PORTAL_VERAOS_TRAINING_TARGET_V1.json").read_text()
        )
        self.assertEqual(cfg["scope"]["producer_role"], "REAL_NEURAL_WEIGHT_TRAINING_AND_ARTIFACT_CUSTODY")
        self.assertEqual(cfg["scope"]["model_product"], "STANDALONE_KOBOLDCPP_COMPATIBLE_VERA")
        self.assertFalse(cfg["scope"]["operating_system_dependency_on_model"])
        self.assertFalse(cfg["boundary"]["model_can_issue_protected_effects"])
        self.assertFalse(cfg["boundary"]["model_can_write_canonical_memory"])
        self.assertFalse(cfg["boundary"]["automatic_release_or_install"])
        self.assertTrue(cfg["qualification"]["require_actual_runtime_consumption_with_request_evidence"])
        self.assertTrue(cfg["qualification"]["reject_name_only_and_discovery_only_routes"])
        self.assertIn("long-context-continuity", cfg["heldout_evaluation_families"])
        self.assertIn("correction-persistence", cfg["heldout_evaluation_families"])
        self.assertTrue(any(p["id"] == "semantic_pragmatic_precision" for p in cfg["training_priorities"]))


class CandidateHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.art = self.root / "artifacts"
        self.art.mkdir()
        names = {
            "adapter_weights": ("adapter_model.safetensors", b"FAKE-ADAPTER-WEIGHTS"),
            "adapter_config": ("adapter_config.json", b'{"r":8}'),
            "gguf": ("vera.gguf", b"GGUF" + b"\x03\x00\x00\x00" + b"dummydata"),
        }
        artifacts = {}
        for key, (name, body) in names.items():
            (self.art / name).write_bytes(body)
            artifacts[key] = {"path": name, "sha256": digest(body)}
        self.manifest = {
            "schema": "VERA_STANDALONE_NEURAL_MODEL_CANDIDATE_V1",
            "producer": "thebrazenbeard/vera_model_training",
            "product": "STANDALONE_VERA_MODEL",
            "integration": "OPTIONAL_COGNITION_PROVIDER",
            "route_class": "COGNITION_ONLY",
            "standalone_inference_backend": "KOBOLDCPP",
            "qualification_state": "UNQUALIFIED_CANDIDATE",
            "model_id": "vera-standalone-test",
            "base_model": {"id": "example-public-base", "revision": "a" * 40},
            "training_recipe_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "quantization": "Q4_K_M",
            "context_length": 8192,
            "private_user_data_included": False,
            "identity_authority": False,
            "canonical_memory_write_authority": False,
            "browser_authority": False,
            "protected_effect_authority": False,
            "may_self_qualify": False,
            "artifacts": artifacts,
        }
        self.manifest_path = self.root / "candidate.json"

    def write(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_verified_bytes_do_not_self_qualify(self):
        self.write()
        result = handoff.verify_candidate(self.manifest_path, self.art)
        self.assertEqual(result["status"], "ARTIFACT_BYTES_VERIFIED_NOT_MODEL_QUALIFIED")
        self.assertEqual(result["gguf_sha256"], self.manifest["artifacts"]["gguf"]["sha256"])
        self.assertFalse(result["pre_active_target_selected"])
        self.assertFalse(result["portal_runtime_consumption"])
        self.assertFalse(result["veraos_interface_qualified"])
        self.assertFalse(result["independent_model_qualification"])

    def test_hash_mutation_rejected(self):
        self.write()
        with (self.art / "vera.gguf").open("ab") as f:
            f.write(b"modified")
        with self.assertRaisesRegex(handoff.CandidateRejected, "differs"):
            handoff.verify_candidate(self.manifest_path, self.art)

    def test_name_is_not_authority(self):
        self.manifest["model_id"] = "vera-local.latest"
        self.manifest["protected_effect_authority"] = True
        with self.assertRaisesRegex(handoff.CandidateRejected, "protected_effect_authority"):
            handoff.validate_descriptor(self.manifest)

    def test_string_false_is_not_boolean(self):
        self.manifest["identity_authority"] = "false"
        with self.assertRaisesRegex(handoff.CandidateRejected, "identity_authority"):
            handoff.validate_descriptor(self.manifest)

    def test_public_private_data_rejected(self):
        self.manifest["private_user_data_included"] = True
        with self.assertRaisesRegex(handoff.CandidateRejected, "private user data"):
            handoff.validate_descriptor(self.manifest)

    def test_qualification_claim_rejected(self):
        self.manifest["qualification_state"] = "QUALIFIED"
        with self.assertRaisesRegex(handoff.CandidateRejected, "qualification_state"):
            handoff.validate_descriptor(self.manifest)

    def test_wrong_base_revision_rejected(self):
        self.manifest["base_model"]["revision"] = "unknown"
        with self.assertRaisesRegex(handoff.CandidateRejected, "base_model.revision"):
            handoff.validate_descriptor(self.manifest)

    def test_bool_is_not_context_length(self):
        self.manifest["context_length"] = True
        with self.assertRaisesRegex(handoff.CandidateRejected, "context_length"):
            handoff.validate_descriptor(self.manifest)

    def test_traversal_rejected(self):
        self.manifest["artifacts"]["gguf"]["path"] = "../outside.gguf"
        with self.assertRaisesRegex(handoff.CandidateRejected, "traversal"):
            handoff.validate_descriptor(self.manifest)

    def test_absolute_path_rejected(self):
        self.manifest["artifacts"]["gguf"]["path"] = "/tmp/vera.gguf"
        with self.assertRaisesRegex(handoff.CandidateRejected, "relative"):
            handoff.validate_descriptor(self.manifest)

    def test_windows_path_rejected(self):
        self.manifest["artifacts"]["gguf"]["path"] = "C:\\models\\vera.gguf"
        with self.assertRaisesRegex(handoff.CandidateRejected, "invalid"):
            handoff.validate_descriptor(self.manifest)

    def test_duplicate_artifact_path_rejected(self):
        self.manifest["artifacts"]["gguf"]["path"] = "adapter_model.safetensors"
        with self.assertRaisesRegex(handoff.CandidateRejected, "share a path"):
            handoff.validate_descriptor(self.manifest)

    def test_symlink_escape_rejected(self):
        self.write()
        external = self.root / "outside.gguf"
        external.write_bytes((self.art / "vera.gguf").read_bytes())
        (self.art / "vera.gguf").unlink()
        try:
            (self.art / "vera.gguf").symlink_to(external)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows user lacks symlink-creation privilege")
            raise
        with self.assertRaisesRegex(handoff.CandidateRejected, "escapes root"):
            handoff.verify_candidate(self.manifest_path, self.art)

    def test_gguf_magic_rejected_despite_matching_hash(self):
        body = b"NOTGGUF123456"
        (self.art / "vera.gguf").write_bytes(body)
        self.manifest["artifacts"]["gguf"]["sha256"] = digest(body)
        self.write()
        with self.assertRaisesRegex(handoff.CandidateRejected, "not a GGUF"):
            handoff.verify_candidate(self.manifest_path, self.art)

    def test_backend_cannot_be_silently_substituted(self):
        self.manifest["standalone_inference_backend"] = "UNKNOWN"
        with self.assertRaisesRegex(handoff.CandidateRejected, "backend"):
            handoff.validate_descriptor(self.manifest)


if __name__ == "__main__":
    unittest.main()
