"""Offline hash-only source checks; no weights, network, GPU or rights grants."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trainingday.pinned_base import (
    BASE_ID, BASE_REVISION, BaseBindingError, require_official_base,
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


class PinnedBaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.shard = "model.safetensors-00001-of-00001.safetensors"
        self.files = {
            "config.json": b"{}", "tokenizer_config.json": b"{}",
            "tokenizer.json": b"{}", "model.safetensors.index.json": b"{}",
            self.shard: b"synthetic-only-weight-placeholder",
        }
        self.manifest = {
            "schema": "TRAININGDAY_PINNED_BASE_V1", "model_id": BASE_ID,
            "revision": BASE_REVISION, "loader_class": "Qwen3_5ForCausalLM",
            "quantization": "NF4_DOUBLE_BF16",
            "snapshot_path": str(self.snapshot),
            "files": {name: sha(data) for name, data in self.files.items()},
        }
        for name, contents in self.files.items():
            (self.snapshot / name).write_bytes(contents)
        self.binding = self.root / "binding.json"
        self.write_binding()

    def write_binding(self):
        self.binding.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_hashes_are_checked_but_no_authority_is_issued(self):
        binding, snapshot = require_official_base(self.binding)
        self.assertEqual(binding, self.manifest)
        self.assertEqual(snapshot, self.snapshot)
        self.assertNotIn("training_authorized", binding)

    def test_wrong_model_family_or_revision_is_rejected(self):
        for field, value in (("model_id", "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"),
                             ("revision", "d61dd146c8fd44c9a49cdb7f59f34e17b61902d8")):
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = value
                self.write_binding()
                with self.assertRaises(BaseBindingError):
                    require_official_base(self.binding)
                self.manifest[field] = original
        self.write_binding()

    def test_rejects_corrupt_or_unbound_weights(self):
        (self.snapshot / self.shard).write_bytes(b"tampered")
        with self.assertRaises(BaseBindingError):
            require_official_base(self.binding)
        (self.snapshot / self.shard).write_bytes(self.files[self.shard])
        (self.snapshot / "model.safetensors-00002-of-00002.safetensors").write_bytes(b"foreign")
        with self.assertRaises(BaseBindingError):
            require_official_base(self.binding)

    def test_rejects_duplicate_identity_keys_and_nonfinite(self):
        original = self.binding.read_text(encoding="utf-8")
        for raw in (original.replace('"model_id":', '"model_id":"forged", "model_id":', 1),
                    original[:-1] + ',"unexpected":NaN}'):
            with self.subTest(raw=raw[:80]):
                self.binding.write_text(raw, encoding="utf-8")
                with self.assertRaises(BaseBindingError):
                    require_official_base(self.binding)

    def test_rejects_symlink_binding_snapshot_or_shard(self):
        alias = self.root / "alias.json"
        alias.symlink_to(self.binding)
        with self.assertRaises(BaseBindingError):
            require_official_base(alias)
        shard = self.snapshot / self.shard
        shard.unlink()
        shard.symlink_to(self.snapshot / "config.json")
        with self.assertRaises(BaseBindingError):
            require_official_base(self.binding)


if __name__ == "__main__":
    unittest.main()
