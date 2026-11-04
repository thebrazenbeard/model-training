"""Offline source-only fixed official pins; never download weights or use GPU."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from trainingday.official_public_base import (
    INDEX_TENSOR_BYTES, OFFICIAL_ID, OFFICIAL_REVISION, PHYSICAL_SHARD_BYTES,
    OfficialBaseSourceError, expected_source_record, read_official_source_record,
    validate_official_source_record,
)


class OfficialPublicBaseSourceTests(unittest.TestCase):
    def test_repository_manifest_matches_hard_pins(self):
        manifest = (Path(__file__).resolve().parents[1] / "docs" / "artifacts"
                    / "qwen35-4b-OFFICIAL-trainingday-source-20261104.json")
        self.assertEqual(read_official_source_record(manifest), expected_source_record())
        self.assertEqual(OFFICIAL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(OFFICIAL_REVISION, "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(PHYSICAL_SHARD_BYTES - INDEX_TENSOR_BYTES, 90240)

    def test_rejects_third_party_and_shape_compatible_substitutions(self):
        attempts = (
            ("identity", "repository", "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"),
            ("identity", "revision", "d61dd146c8fd44c9a49cdb7f59f34e17b61902d8"),
            ("identity", "repository", "Qwen/Qwen3.5-4B-Base"),
            ("identity", "checkpoint_architecture", "Qwen3_5ForCausalLM"),
            ("safetensors", "physical_shard_bytes", PHYSICAL_SHARD_BYTES + 64),
        )
        for parent, key, replacement in attempts:
            with self.subTest(parent=parent, key=key, replacement=replacement):
                record = copy.deepcopy(expected_source_record())
                record[parent][key] = replacement
                with self.assertRaises(OfficialBaseSourceError):
                    validate_official_source_record(record)

    def test_rejects_different_shard_hash_same_dimensions(self):
        for field, change in (("size_bytes", 64), ("sha256_lfs_oid", "0" * 64)):
            with self.subTest(field=field):
                record = copy.deepcopy(expected_source_record())
                shard = record["safetensors"]["shards"][0]
                shard[field] = shard[field] + change if field == "size_bytes" else change
                with self.assertRaises(OfficialBaseSourceError):
                    validate_official_source_record(record)

    def test_rejects_inflated_authority_or_signature(self):
        for key, replacement in (("status", "TRUSTED"), ("signature", "self-signed"),
                                 ("physical_launch_allowed", True),
                                 ("protected_training_data_allowed", True)):
            with self.subTest(key=key):
                record = copy.deepcopy(expected_source_record())
                record["admission"][key] = replacement
                with self.assertRaises(OfficialBaseSourceError):
                    validate_official_source_record(record)

    def test_rejects_missing_or_extra_keys(self):
        for change in ("extra", "missing"):
            record = copy.deepcopy(expected_source_record())
            if change == "extra":
                record["admission"]["owner_approved"] = True
            else:
                del record["admission"]["status"]
            with self.assertRaises(OfficialBaseSourceError):
                validate_official_source_record(record)

    def test_rejects_duplicate_json_keys_or_nonfinite_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.json"
            raw = json.dumps(expected_source_record())
            for string in (
                raw.replace('"schema":', '"schema": "fake", "schema":', 1),
                raw[:-1] + ', "unexpected": NaN}',
            ):
                path.write_text(string, encoding="utf-8")
                with self.assertRaises(OfficialBaseSourceError):
                    read_official_source_record(path)
            path.write_text(raw, encoding="utf-8")
            alias = root / "alias.json"
            alias.symlink_to(path)
            with self.assertRaises(OfficialBaseSourceError):
                read_official_source_record(alias)


if __name__ == "__main__":
    unittest.main()
