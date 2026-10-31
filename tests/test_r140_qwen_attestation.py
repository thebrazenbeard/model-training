"""CPU fixtures: R140 proves bounded offline inventory consistency only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

from trainingday.r140_qwen_attestation import (
    SnapshotAttestationError, audit_snapshot,
)


PIN = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: object) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def shard(tensors: dict[str, tuple[str, list[int], bytes]], *, gap: int = 0) -> bytes:
    header: dict[str, dict] = {}
    payload = b""
    for name, (dtype, shape, raw) in tensors.items():
        offset = len(payload) + gap
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        payload += raw
    header_data = canonical(header)
    return struct.pack("<Q", len(header_data)) + header_data + payload


class QwenSnapshotAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.binding = self.root / "binding.json"
        config = {"architectures": ["Qwen3_5ForConditionalGeneration"],
                  "model_type": "qwen3_5",
                  "text_config": {"model_type": "qwen3_5_text"},
                  "vision_config": {"model_type": "qwen3_5"}}
        self._write("config.json", canonical(config))
        self._write("tokenizer_config.json", canonical({
            "chat_template": "{{ messages }}", "eos_token": "<|im_end|>",
            "tokenizer_class": "Qwen2Tokenizer",
        }))
        self._write("tokenizer.json", canonical({"model": {"vocab": {"a": 1}}}))
        self._write("vocab.json", canonical({"a": 1}))
        self._write("merges.txt", b"#version: 0.2\na b\n")
        self._write("chat_template.jinja", b"{{ messages }}\n")
        self._write("model.safetensors-00001-of-00002.safetensors",
                    shard({"text.weight": ("BF16", [2], b"ABCD")}))
        self._write("model.safetensors-00002-of-00002.safetensors",
                    shard({"text.bias": ("F32", [1], b"EFGH")}))
        self._index()
        self._binding()

    def _write(self, name: str, value: bytes) -> None:
        (self.snapshot / name).write_bytes(value)

    def _index(self, *, names: dict[str, str] | None = None,
               total_size: int = 8) -> None:
        if names is None:
            names = {"text.weight": "model.safetensors-00001-of-00002.safetensors",
                     "text.bias": "model.safetensors-00002-of-00002.safetensors"}
        self._write("model.safetensors.index.json", canonical({
            "metadata": {"total_size": total_size}, "weight_map": names,
        }))

    def _binding(self, *, changes: dict | None = None) -> None:
        files = {path.name: digest(path.read_bytes())
                 for path in self.snapshot.iterdir() if path.is_file()}
        bound = {"schema": "TRAININGDAY_PINNED_BASE_V1",
                 "model_id": "Qwen/Qwen3.5-4B", "revision": PIN,
                 "loader_class": "Qwen3_5ForCausalLM",
                 "quantization": "NF4_DOUBLE_BF16",
                 "snapshot_path": str(self.snapshot), "files": files}
        if changes:
            bound.update(changes)
        self.binding.write_bytes(canonical(bound))

    def _audit(self, **overrides: str) -> dict:
        defaults = {"expected_binding_sha256": digest(self.binding.read_bytes()),
                    "expected_config_sha256": digest((self.snapshot / "config.json").read_bytes()),
                    "expected_tokenizer_sha256": digest(
                        (self.snapshot / "tokenizer_config.json").read_bytes()),
                    "source_root": Path(__file__).resolve().parents[1]}
        defaults.update(overrides)
        return audit_snapshot(self.binding, **defaults)

    def test_valid_mock_inventory_remains_runtime_and_training_hold(self) -> None:
        result = self._audit()
        self.assertEqual(result["status"], "SOURCE_SNAPSHOT_INVENTORY_VALID_RUNTIME_HOLD")
        self.assertEqual(result["revision"], PIN)
        self.assertEqual(result["safetensors_shard_count"], 2)
        self.assertEqual(result["safetensors_tensor_count"], 2)
        self.assertEqual(result["safetensors_indexed_tensor_bytes"], 8)
        self.assertEqual(set(result["source_sha256"]), {"r18_mezo", "r18_qlora"})
        for name in ("physical_tokenizer_verified", "runtime_loader_verified",
                     "actual_model_tensors_materialized", "finite_forward_loss_verified",
                     "external_pin_provenance_verified", "host_immutable_custody_verified",
                     "exclusive_cuda_lease_verified", "authorizes_training"):
            self.assertIs(result[name], False, name)

    def test_independent_binding_pin_is_required_not_derivable_from_binding(self) -> None:
        for invalid in ("A" * 64, "a" * 63, None, True, "f" * 64):
            with self.subTest(invalid=invalid), self.assertRaises(SnapshotAttestationError):
                self._audit(expected_binding_sha256=invalid)

    def test_wrong_tokenizer_external_pin_rejected_even_when_binding_matches(self) -> None:
        with self.assertRaisesRegex(SnapshotAttestationError, "independent config/tokenizer"):
            self._audit(expected_tokenizer_sha256="a" * 64)

    def test_wrong_revision_loader_model_and_quantization_rejected(self) -> None:
        for key, value in (("revision", "e" * 40), ("model_id", "other/base"),
                           ("loader_class", "AutoModel"), ("quantization", "BF16")):
            with self.subTest(key=key):
                self._binding(changes={key: value})
                with self.assertRaisesRegex(SnapshotAttestationError, "base model"):
                    self._audit()

    def test_wrong_actual_file_bytes_rejected_after_digest_mismatch(self) -> None:
        with (self.snapshot / "model.safetensors-00002-of-00002.safetensors").open("ab") as stream:
            stream.write(b"X")
        with self.assertRaisesRegex(SnapshotAttestationError, "digest differs"):
            self._audit()

    def test_index_shard_name_and_actual_header_must_agree(self) -> None:
        self._index(names={"text.weight": "model.safetensors-00001-of-00002.safetensors",
                           "phantom.weight": "model.safetensors-00002-of-00002.safetensors"})
        self._binding()
        with self.assertRaisesRegex(SnapshotAttestationError, "tensor names differ"):
            self._audit()

    def test_index_total_size_rechecked_against_tensor_data(self) -> None:
        self._index(total_size=7)
        self._binding()
        with self.assertRaisesRegex(SnapshotAttestationError, "indexed total tensor size"):
            self._audit()

    def test_invalid_tensor_dtype_shape_offsets_gap_and_header_bounds(self) -> None:
        variants = (
            shard({"text.weight": ("E999", [4], b"ABCD")}),
            shard({"text.weight": ("BF16", [4], b"ABCD")}),
            shard({"text.weight": ("BF16", [2], b"ABCD")}, gap=1),
            struct.pack("<Q", 2**32) + b"{}ABCD",
            struct.pack("<Q", 10) + b"{}",
        )
        for data in variants:
            with self.subTest(prefix=data[:16]):
                self._write("model.safetensors-00001-of-00002.safetensors", data)
                self._binding()
                with self.assertRaises(SnapshotAttestationError):
                    self._audit()

    def test_unpinned_extra_safetensors_rejected(self) -> None:
        self._write("extra.safetensors", shard({"bad": ("BF16", [2], b"ABCD")}))
        with self.assertRaisesRegex(SnapshotAttestationError, "extra, missing or unbound"):
            self._audit()

    def test_missing_tokenizer_chat_contract_rejected_even_with_fresh_hashes(self) -> None:
        self._write("tokenizer_config.json", canonical({
            "eos_token": "<|im_end|>", "tokenizer_class": "Qwen2Tokenizer"}))
        self._binding()
        with self.assertRaisesRegex(SnapshotAttestationError, "chat template"):
            self._audit()

    def test_omitted_pinned_tokenizer_asset_rejected(self) -> None:
        for omitted in ("tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja"):
            with self.subTest(omitted=omitted):
                bound = json.loads(self.binding.read_text(encoding="utf-8"))
                bound["files"].pop(omitted)
                self.binding.write_bytes(canonical(bound))
                with self.assertRaisesRegex(SnapshotAttestationError, f"required {re.escape(omitted)}"):
                    self._audit()
                self._binding()

    def test_unbound_optional_tokenizer_override_rejected(self) -> None:
        self._write("special_tokens_map.json", canonical({"eos_token": "surprise"}))
        with self.assertRaisesRegex(SnapshotAttestationError, "unbound optional"):
            self._audit()

    def test_symlinked_bound_file_is_rejected(self) -> None:
        original = self.root / "real-tokenizer.json"
        candidate = self.snapshot / "tokenizer_config.json"
        candidate.rename(original)
        try:
            candidate.symlink_to(original)
        except OSError:
            self.skipTest("symlink creation unsupported")
        with self.assertRaisesRegex(SnapshotAttestationError, "symlinked"):
            self._audit()

    def test_reject_duplicate_json_fields_even_with_matching_sha(self) -> None:
        binding = json.loads(self.binding.read_text(encoding="utf-8"))
        raw = b'{"schema":"TRAININGDAY_PINNED_BASE_V1","schema":"WRONG"}'
        self.binding.write_bytes(raw)
        with self.assertRaisesRegex(SnapshotAttestationError, "invalid base binding"):
            self._audit()
        self.binding.write_bytes(canonical(binding))

    def test_cli_failure_redacts_snapshot_path(self) -> None:
        result = subprocess.run([
            sys.executable, "-m", "trainingday.r140_qwen_attestation",
            "--base-binding", str(self.binding),
            "--expected-binding-sha256", "f" * 64,
            "--expected-config-sha256", "f" * 64,
            "--expected-tokenizer-sha256", "f" * 64,
        ], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(str(self.snapshot), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
