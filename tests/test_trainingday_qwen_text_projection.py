"""Offline multimodal text-key projection tests; no real weights or CUDA."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trainingday.qwen_text_checkpoint import (
    TextCheckpointProjectionError, index_binding_pin, text_key_mapping,
    verify_text_load,
)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.snapshot = Path(self.temp.name)
        self.index = self.snapshot / "model.safetensors.index.json"
        self.shard = "model.safetensors-00001-of-00001.safetensors"
        self.entries = {
            "model.language_model.embed_tokens.weight": self.shard,
            "model.language_model.layers.0.mlp.down_proj.weight": self.shard,
            "model.visual.patch_embed.weight": self.shard,
            "mtp.fc.weight": self.shard,
        }
        self.write_index()

    def write_index(self):
        self.index.write_text(json.dumps({"metadata": {"total_size": 1337},
                                          "weight_map": self.entries}), encoding="utf-8")

    def mapping(self):
        digest = hashlib.sha256(self.index.read_bytes()).hexdigest()
        index_sha, shards = index_binding_pin({"files": {
            "model.safetensors.index.json": digest, self.shard: "a" * 64,
        }})
        return text_key_mapping(self.snapshot,
                                expected_index_sha256=index_sha,
                                expected_shards=shards)

    def test_maps_only_pinned_language_weights(self):
        self.assertEqual(self.mapping(), {
            "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.language_model.layers.0.mlp.down_proj.weight":
                "model.layers.0.mlp.down_proj.weight",
        })

    def test_rejects_missing_embedding_or_unknown_tensor(self):
        self.entries.pop("model.language_model.embed_tokens.weight")
        self.write_index()
        with self.assertRaises(TextCheckpointProjectionError):
            self.mapping()
        self.entries["model.language_model.embed_tokens.weight"] = self.shard
        self.entries["lm_head.weight"] = self.shard
        self.write_index()
        with self.assertRaises(TextCheckpointProjectionError):
            self.mapping()

    def test_rejects_index_tamper_duplicate_json_and_unbound_shard(self):
        index_sha = hashlib.sha256(self.index.read_bytes()).hexdigest()
        with self.assertRaises(TextCheckpointProjectionError):
            text_key_mapping(self.snapshot, expected_index_sha256="f" * 64,
                             expected_shards={self.shard})
        self.index.write_text('{"metadata":{},"metadata":{},"weight_map":{}}',
                              encoding="utf-8")
        with self.assertRaises(TextCheckpointProjectionError):
            text_key_mapping(self.snapshot, expected_index_sha256=hashlib.sha256(
                self.index.read_bytes()).hexdigest(), expected_shards={self.shard})
        with self.assertRaises(TextCheckpointProjectionError):
            index_binding_pin({"files": {"model.safetensors.index.json": index_sha,
                                         "wrong.safetensors": "f" * 64}})

    def test_rejects_partial_or_untied_weight_load(self):
        class FakeModel:
            def __init__(self, tied):
                self.input = object()
                self.output = self.input if tied else object()

            def get_input_embeddings(self):
                return type("Embedding", (), {"weight": self.input})()

            def get_output_embeddings(self):
                return type("Embedding", (), {"weight": self.output})()

        good = {"missing_keys": [], "unexpected_keys": [],
                "mismatched_keys": [], "error_msgs": []}
        verify_text_load(FakeModel(True), good)
        with self.assertRaises(TextCheckpointProjectionError):
            verify_text_load(FakeModel(False), good)
        with self.assertRaises(TextCheckpointProjectionError):
            verify_text_load(FakeModel(True), {**good, "missing_keys": ["x"]})


if __name__ == "__main__":
    unittest.main()
