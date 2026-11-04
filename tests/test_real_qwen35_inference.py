"""Negative, offline checks for the guarded official-GGUF inference seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday.real_qwen35_inference import (
    BASE_ID, BASE_REVISION, OFFICIAL_SHARD_SHA256, SCHEMA,
    InferenceHold, build_command, main, probe, require_probe_resources,
    verify_manifest,
)


def record(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class RealQwenInferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = Path(self.temp.name)
        self.base = p / "official-Q4_K_M.gguf"
        self.base.write_bytes(b"GGUF" + b"\x01" * 64)
        self.adapter = p / "r16-correct8.gguf"
        self.adapter.write_bytes(b"GGUF" + b"\x02" * 32)
        self.server = p / "llama-server"
        self.server.write_bytes(b"dummy test-only server file")
        self.manifest = {
            "schema": SCHEMA, "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "official_shard_sha256": dict(OFFICIAL_SHARD_SHA256),
            "base_gguf": record(self.base),
            "llama_server": record(self.server),
            "r16_correct8_adapter_gguf": {
                **record(self.adapter), "source_run": "r16-v2-correct8-20261030-b",
            },
        }

    def test_preflight_base_and_trained_distinguish_artifacts(self):
        base = verify_manifest(self.manifest, "base")
        trained = verify_manifest(self.manifest, "trained")
        self.assertFalse(base["runtime_started"])
        self.assertEqual(base["adapter_gguf_sha256"], None)
        self.assertEqual(trained["adapter_gguf_sha256"], record(self.adapter)["sha256"])
        self.assertFalse(trained["conversion_lineage_independently_verified"])
        self.assertFalse(trained["official_shards_freshly_checked"])
        self.assertEqual(build_command(base, port=18888)[-2:], ["--n-gpu-layers", "0"])
        self.assertEqual(build_command(trained, port=18888)[-2:],
                         ["--lora", str(self.adapter)])
        self.assertIn("127.0.0.1", build_command(base, port=18888))
        self.assertEqual(build_command(base, port=18888).count("--threads"), 1)
        self.assertEqual(build_command(base, port=18888).count("--threads-batch"), 1)

    def test_probe_refuses_without_owner_idle_ack_before_process_launch(self):
        preflight = verify_manifest(self.manifest, "base")
        with patch("trainingday.real_qwen35_inference.subprocess.Popen") as launch:
            with self.assertRaisesRegex(InferenceHold, "owner confirmation"):
                probe(preflight)
            launch.assert_not_called()

    def test_probe_refuses_low_available_ram_before_process_launch(self):
        preflight = verify_manifest(self.manifest, "base")
        with patch("trainingday.real_qwen35_inference.available_ram_bytes", return_value=2 * 1024**3):
            with patch("trainingday.real_qwen35_inference.subprocess.Popen") as launch:
                with self.assertRaisesRegex(InferenceHold, "insufficient available physical RAM"):
                    probe(preflight, host_idle_acknowledged=True)
                launch.assert_not_called()

    def test_preflight_ram_floor_scales_with_actual_model_size(self):
        preflight = verify_manifest(self.manifest, "base")
        preflight["base_gguf_bytes"] = 9 * 1024**3
        with patch("trainingday.real_qwen35_inference.available_ram_bytes", return_value=8 * 1024**3):
            with self.assertRaisesRegex(InferenceHold, "insufficient available"):
                require_probe_resources(preflight, host_idle_acknowledged=True)

    def test_edited_gguf_is_rejected(self):
        self.base.write_bytes(b"GGUF" + b"\x00" * 64)
        with self.assertRaisesRegex(InferenceHold, "bytes do not match"):
            verify_manifest(self.manifest, "base")

    def test_bad_gguf_magic_is_rejected_even_with_matching_hash(self):
        self.base.write_bytes(b"FAKE" + b"\x01" * 64)
        self.manifest["base_gguf"] = record(self.base)
        with self.assertRaisesRegex(InferenceHold, "GGUF magic mismatch"):
            verify_manifest(self.manifest, "base")

    def test_corrupted_labels_rejected_as_trained_vera(self):
        self.manifest["r16_correct8_adapter_gguf"]["source_run"] = "r16-v2-corrupt8-20261030-b"
        with self.assertRaisesRegex(InferenceHold, "correct-label"):
            verify_manifest(self.manifest, "trained")

    def test_missing_adapter_rejected_for_trained_but_not_base(self):
        del self.manifest["r16_correct8_adapter_gguf"]
        verify_manifest(self.manifest, "base")
        with self.assertRaisesRegex(InferenceHold, "requires"):
            verify_manifest(self.manifest, "trained")

    def test_wrong_official_shard_digest_rejected(self):
        self.manifest["official_shard_sha256"]["model.safetensors-00001-of-00002.safetensors"] = "0" * 64
        with self.assertRaisesRegex(InferenceHold, "shard digest"):
            verify_manifest(self.manifest, "base")

    def test_relative_and_symlink_paths_rejected(self):
        self.manifest["base_gguf"]["path"] = "./official-Q4_K_M.gguf"
        with self.assertRaisesRegex(InferenceHold, "relative"):
            verify_manifest(self.manifest, "base")
        linked = Path(self.temp.name) / "link.gguf"
        try:
            linked.symlink_to(self.base)
        except OSError:
            self.skipTest("host disallows user symlinks")
        self.manifest["base_gguf"] = record(self.base)
        self.manifest["base_gguf"]["path"] = str(linked)
        with self.assertRaisesRegex(InferenceHold, "symlink"):
            verify_manifest(self.manifest, "base")

    def test_cli_preflight_produces_true_nonruntime_receipt(self):
        manifest_path = Path(self.temp.name) / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.assertEqual(main(["preflight", "--manifest", str(manifest_path),
                               "--mode", "base"]), 0)


if __name__ == "__main__":
    unittest.main()
