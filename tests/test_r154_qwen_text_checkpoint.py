"""Offline hostile tests for the R18 multimodal-to-text checkpoint projection.

No Transformers, torch, model files, network, or physical authority is needed.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trainingday.qwen_text_checkpoint import (
    TextCheckpointProjectionError, index_binding_pin, text_key_mapping,
    verify_text_load,
)


class QwenTextProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.snapshot = Path(self.temp.name)
        self.index = self.snapshot / "model.safetensors.index.json"
        self.shard = "model.safetensors-00001-of-00002.safetensors"
        self.weight_map = {
            "model.language_model.embed_tokens.weight": self.shard,
            "model.language_model.layers.0.mlp.down_proj.weight": self.shard,
            "model.visual.patch_embed.proj.weight": self.shard,
            "mtp.fc.weight": self.shard,
        }
        self.write_index()

    def write_index(self):
        self.index.write_text(json.dumps({
            "metadata": {"total_size": 1234}, "weight_map": self.weight_map,
        }), encoding="utf-8")

    def mapping(self):
        return text_key_mapping(
            self.snapshot,
            expected_index_sha256=hashlib.sha256(self.index.read_bytes()).hexdigest(),
            expected_shards={self.shard},
        )

    def test_maps_text_weights_excludes_visual_and_mtp(self):
        self.assertEqual(self.mapping(), {
            "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.language_model.layers.0.mlp.down_proj.weight":
                "model.layers.0.mlp.down_proj.weight",
        })

    def test_missing_text_embedding_is_rejected(self):
        self.weight_map.pop("model.language_model.embed_tokens.weight")
        self.write_index()
        with self.assertRaisesRegex(TextCheckpointProjectionError, "embeddings"):
            self.mapping()

    def test_unknown_head_or_namespace_is_rejected(self):
        for name in ("lm_head.weight", "model.hidden.weight",
                     "model.language_modelx.weight", "model.language_model."):
            with self.subTest(name=name):
                self.weight_map[name] = self.shard
                self.write_index()
                with self.assertRaises(TextCheckpointProjectionError):
                    self.mapping()
                del self.weight_map[name]

    def test_untrusted_shard_paths_and_types_are_rejected(self):
        for shard in ("../escape.safetensors", "other.safetensors", "",
                      42, "model.safetensors-00001-of-00002.safetensors/../x"):
            with self.subTest(shard=shard):
                self.weight_map["model.language_model.embed_tokens.weight"] = shard
                self.write_index()
                with self.assertRaises(TextCheckpointProjectionError):
                    self.mapping()

    def test_duplicate_nonfinite_invalid_json_and_oversized_rejected(self):
        for value in (
            '{"weight_map":{},"weight_map":{},"metadata":{}}',
            '{"weight_map":{},"metadata":{"total_size":NaN}}',
            '{"weight_map":{},"metadata":{}}',
            "x" * (1024 * 1024 + 1),
        ):
            with self.subTest(value=value[:24]):
                self.index.write_text(value, encoding="utf-8")
                with self.assertRaises(TextCheckpointProjectionError):
                    self.mapping()

    def test_missing_and_symlinked_index_rejected(self):
        self.index.unlink()
        with self.assertRaises(TextCheckpointProjectionError):
            text_key_mapping(self.snapshot, expected_index_sha256="0" * 64,
                             expected_shards={self.shard})
        original = self.snapshot / "outside.json"
        original.write_text("{}", encoding="utf-8")
        self.index.symlink_to(original)
        with self.assertRaises(TextCheckpointProjectionError):
            text_key_mapping(self.snapshot, expected_index_sha256="0" * 64,
                             expected_shards={self.shard})

    def test_index_cannot_control_weights_without_binding_digest(self):
        digest = hashlib.sha256(self.index.read_bytes()).hexdigest()
        binding = {"files": {"config.json": "0" * 64,
                             "model.safetensors.index.json": digest,
                             self.shard: "1" * 64}}
        sha, shards = index_binding_pin(binding)
        self.assertEqual(text_key_mapping(self.snapshot, expected_index_sha256=sha,
                                          expected_shards=shards), self.mapping())
        for invalid in ({"files": {self.shard: "1" * 64}},
                        {"files": {"model.safetensors.index.json": "2" * 64}},
                        {"files": {"model.safetensors.index.json": digest,
                                   "unsafe.safetensors": "1" * 64}}):
            with self.subTest(invalid=invalid), self.assertRaises(TextCheckpointProjectionError):
                index_binding_pin(invalid)
        with self.assertRaises(TextCheckpointProjectionError):
            text_key_mapping(self.snapshot, expected_index_sha256="0" * 64,
                             expected_shards=shards)

    def test_index_shard_set_must_match_bound_inventory(self):
        with self.assertRaisesRegex(TextCheckpointProjectionError, "shard set"):
            text_key_mapping(self.snapshot,
                             expected_index_sha256=hashlib.sha256(
                                 self.index.read_bytes()).hexdigest(),
                             expected_shards={self.shard,
                                              "model.safetensors-00002-of-00002.safetensors"})

    def test_weight_load_evidence_is_fail_closed(self):
        class FakeModel:
            def __init__(self, tied=True):
                self.input = object()
                self.output = self.input if tied else object()

            def get_input_embeddings(self):
                return type("Embedding", (), {"weight": self.input})()

            def get_output_embeddings(self):
                return type("Embedding", (), {"weight": self.output})()

        valid = {"missing_keys": set(), "unexpected_keys": [],
                 "mismatched_keys": set(), "error_msgs": []}
        verify_text_load(FakeModel(), valid)
        verify_text_load(FakeModel(), {**valid, "unexpected_keys": [
            "model.visual.conv.weight", "mtp.fc.weight"]})
        for bad in ({**valid, "missing_keys": {"model.layers.1.weight"}},
                    {**valid, "mismatched_keys": {"model.embed_tokens.weight"}},
                    {**valid, "unexpected_keys": ["model.layers.1.unexpected"]},
                    {**valid, "error_msgs": ["checkpoint read failed"]},
                    {"missing_keys": []}):
            with self.subTest(bad=bad), self.assertRaises(TextCheckpointProjectionError):
                verify_text_load(FakeModel(), bad)
        with self.assertRaisesRegex(TextCheckpointProjectionError, "tied"):
            verify_text_load(FakeModel(tied=False), valid)


if __name__ == "__main__":
    unittest.main()
