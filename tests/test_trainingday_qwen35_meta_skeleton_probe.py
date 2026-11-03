"""Source-only negative tests; hosted meta integration runs in separate CI."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trainingday import qwen35_meta_skeleton_probe as probe


class MetaSkeletonSourceTests(unittest.TestCase):
    @staticmethod
    def sample_config():
        return {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text",
                            "num_hidden_layers": 32,
                            "layer_types": ["linear_attention"] * 24
                                           + ["full_attention"] * 8},
        }

    def test_absent_or_forged_config_fails_before_heavy_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            with self.assertRaisesRegex(probe.MetaSkeletonProbeError, "inaccessible"):
                probe.run_meta_probe(path)
            path.write_text(json.dumps(self.sample_config()), encoding="utf-8")
            with self.assertRaisesRegex(probe.MetaSkeletonProbeError, "official SHA256"):
                probe.run_meta_probe(path)

    def test_synthetic_self_hash_is_not_official_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            raw = json.dumps(self.sample_config()).encode()
            path.write_bytes(raw)
            self.assertNotEqual(hashlib.sha256(raw).hexdigest(),
                                probe.OFFICIAL_CONFIG_SHA256)
            with patch.object(probe, "OFFICIAL_CONFIG_SHA256",
                              hashlib.sha256(raw).hexdigest()):
                self.assertEqual(probe.verified_config(path)["model_type"],
                                 "qwen3_5_text")
            with self.assertRaises(probe.MetaSkeletonProbeError):
                probe.verified_config(path)

    def test_wrong_identity_even_with_self_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            value = self.sample_config()
            value["model_type"] = "qwen3_5_moe"
            raw = json.dumps(value).encode()
            path.write_bytes(raw)
            with patch.object(probe, "OFFICIAL_CONFIG_SHA256",
                              hashlib.sha256(raw).hexdigest()):
                with self.assertRaisesRegex(probe.MetaSkeletonProbeError, "text config"):
                    probe.verified_config(path)

    def test_symlink_or_oversized_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "config.json"
            target.write_bytes(b"{}")
            alias = Path(folder) / "alias.json"
            alias.symlink_to(target)
            with self.assertRaisesRegex(probe.MetaSkeletonProbeError, "symlink"):
                probe.verified_config(alias)
            target.write_bytes(b"x" * (probe.MAX_CONFIG_BYTES + 1))
            with self.assertRaisesRegex(probe.MetaSkeletonProbeError, "bounded"):
                probe.verified_config(target)

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"),
                         "POSIX FIFO support only")
    def test_fifo_metadata_rejected_without_blocking_open(self):
        with tempfile.TemporaryDirectory() as folder:
            pipe = Path(folder) / "config.fifo"
            os.mkfifo(pipe)
            with self.assertRaises(probe.MetaSkeletonProbeError):
                probe.verified_config(pipe)

    def test_cli_requires_config_and_does_not_import_torch(self):
        command = subprocess.run(
            [sys.executable, "-m", "trainingday.qwen35_meta_skeleton_probe"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(command.returncode, 2)
        self.assertEqual(command.stdout, "")
        fresh = subprocess.run(
            [sys.executable, "-c",
             "import sys; import trainingday.qwen35_meta_skeleton_probe; "
             "assert 'torch' not in sys.modules and 'peft' not in sys.modules"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(fresh.returncode, 0, fresh.stderr)

    def test_source_never_loads_pretrained_or_calls_optimizer(self):
        source = inspect.getsource(probe.run_meta_probe)
        self.assertNotIn(".from_pretrained(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn(".cuda(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertIn("init_empty_weights(include_buffers=True)", source)
        self.assertIn("audit_loaded_minimal_lora_model(model", source)


if __name__ == "__main__":
    unittest.main()
