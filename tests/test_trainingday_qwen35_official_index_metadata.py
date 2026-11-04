"""Negative controls for official pretrained checkpoint *index metadata only*."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday import qwen35_official_index_metadata as source


class OfficialIndexMetadataTests(unittest.TestCase):
    def setUp(self):
        self.weights = {"model.language_model.embed_tokens.weight": "s1"}
        self.weights.update({f"model.language_model.layers.{i}.weight": "s1"
                             for i in range(425)})
        self.weights.update({f"model.visual.blocks.{i}.weight": "s2"
                             for i in range(297)})
        self.weights.update({f"mtp.blocks.{i}.weight": "s2"
                             for i in range(15)})
        self.payload = {"metadata": {"total_size": 9_319_737_856},
                        "weight_map": self.weights}
        self.shards = frozenset({"s1", "s2"})
        self.counts = {"language": 426, "vision": 297, "mtp": 15}

    def raw(self):
        return json.dumps(self.payload, separators=(",", ":"),
                          sort_keys=True).encode("utf-8")

    def validate(self, raw):
        return source._parse_index(
            raw, sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw),
            payload_bytes=9_319_737_856, namespace_counts=self.counts,
            shard_names=self.shards)

    def test_pins_and_report_stay_source_only(self):
        report = self.validate(self.raw())
        self.assertEqual(report["indexed_tensor_count"], 738)
        self.assertEqual(report["indexed_namespace_counts"], self.counts)
        self.assertEqual(report["published_tensor_payload_bytes"], 9_319_737_856)
        self.assertEqual(report["referenced_safetensors_shards"], ["s1", "s2"])
        self.assertFalse(report["safetensors_shards_downloaded"])
        self.assertFalse(report["physical_training_authorized"])
        self.assertEqual(report["gpu_fit"], "UNKNOWN")

    def test_sha_and_size_mismatch_rejected(self):
        raw = self.raw()
        with self.assertRaises(source.OfficialIndexMetadataError):
            source._parse_index(raw + b" ", sha256=hashlib.sha256(raw).hexdigest(),
                                byte_size=len(raw), payload_bytes=9_319_737_856,
                                namespace_counts=self.counts, shard_names=self.shards)

    def test_duplicate_json_key_rejected(self):
        raw = self.raw().replace(b'"metadata":', b'"metadata":{},"metadata":', 1)
        with self.assertRaises(source.OfficialIndexMetadataError):
            self.validate(raw)

    def test_wrong_payload_size_rejected(self):
        self.payload["metadata"]["total_size"] -= 1
        with self.assertRaises(source.OfficialIndexMetadataError):
            self.validate(self.raw())

    def test_unbound_third_shard_or_path_rejected(self):
        first = "model.language_model.embed_tokens.weight"
        for unsafe in ("other.safetensors", "../../untrusted.safetensors"):
            with self.subTest(unsafe=unsafe):
                self.weights[first] = unsafe
                with self.assertRaises(source.OfficialIndexMetadataError):
                    self.validate(self.raw())

    def test_missing_text_or_namespace_rejected(self):
        first = "model.language_model.embed_tokens.weight"
        del self.weights[first]
        self.weights["other.embed_tokens.weight"] = "s1"
        with self.assertRaises(source.OfficialIndexMetadataError):
            self.validate(self.raw())

    def test_wrong_namespace_counts_rejected(self):
        old = self.weights.pop("model.visual.blocks.0.weight")
        self.weights["mtp.blocks.99.weight"] = old
        with self.assertRaises(source.OfficialIndexMetadataError):
            self.validate(self.raw())

    def test_public_wrapper_rejects_symlink_and_pin_drift(self):
        raw = self.raw()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_bytes(raw)
            with self.assertRaises(source.OfficialIndexMetadataError):
                source.inspect_official_index(path)  # Not official pinned bytes.
            link = Path(directory) / "link.json"
            try:
                link.symlink_to(path)
            except (OSError, NotImplementedError):
                pass  # Windows may disallow symlink creation without privileges.
            else:
                with self.assertRaises(source.OfficialIndexMetadataError):
                    source.inspect_official_index(link)

            with (patch.object(source, "INDEX_SHA256", hashlib.sha256(raw).hexdigest()),
                  patch.object(source, "INDEX_BYTE_SIZE", len(raw)),
                  patch.object(source, "EXPECTED_NAMESPACE_COUNTS", self.counts),
                  patch.object(source, "EXPECTED_SHARDS", self.shards)):
                self.assertEqual(source.inspect_official_index(path)["indexed_tensor_count"],
                                 738)


if __name__ == "__main__":
    unittest.main()
