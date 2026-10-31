"""R140 read-only Qwen3.5-4B offline *inventory* qualification.

Complements R75's source AST/config preflight with bounded file digests,
tokenizer declaration checks and a complete safetensors-index/header join.
Neither parsing weights metadata nor receiving an operator digest pin proves
that Transformers can load the base, compute a loss, or resume an adapter.

This module deliberately never imports torch/transformers, opens an adapter,
loads model tensors into memory, downloads an artifact or starts training.
File names and hashes are self-reported by the base binding; the caller must
independently authenticate the supplied binding/config/tokenizer digest pins.
Filesystem path checks are NOT a host-owned immutable custody transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct

from trainingday.r75_qwen_loader_preflight import (
    MODEL_ID, PINNED_REVISION, RUNNER_LOADER, LoaderPreflightError,
    _json_dict, audit as source_preflight,
)


SCHEMA = "VERA_TRAININGDAY_R140_QWEN_OFFLINE_SNAPSHOT_ATTESTATION_V1"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
SHARD_NAME = re.compile(r"model\.safetensors-([0-9]{5})-of-([0-9]{5})\.safetensors\Z")
SAFE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
MAX_BINDING_BYTES = 131072
MAX_SMALL_FILE_BYTES = 32 * 1024 * 1024
MAX_WEIGHT_BYTES = 12 * 1024**3
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_WEIGHT_MAP_ENTRIES = 16000
MAX_SHARDS = 16
MAX_DECLARED_FILES = 64
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
               "I64": 8, "U64": 8, "I32": 4, "U32": 4,
               "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}


class SnapshotAttestationError(ValueError):
    """A source-only snapshot inventory was incomplete or contradicted."""


def _require(ok: bool, description: str) -> None:
    if not ok:
        raise SnapshotAttestationError(description)


def _is_sha(value: object) -> bool:
    return type(value) is str and SHA256.fullmatch(value) is not None


def _strict_map(raw: bytes, name: str) -> dict:
    try:
        return _json_dict(raw, name)
    except LoaderPreflightError as exc:
        raise SnapshotAttestationError(f"invalid {name}") from exc


def _read_small(path: Path, name: str, limit: int = MAX_SMALL_FILE_BYTES) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{name} missing or symlinked")
    try:
        with path.open("rb") as reader:
            file_stat = os.fstat(reader.fileno())
            _require(stat.S_ISREG(file_stat.st_mode) and file_stat.st_size <= limit,
                     f"{name} is not a bounded regular file")
            data = reader.read(limit + 1)
    except OSError as exc:
        raise SnapshotAttestationError(f"{name} unreadable") from exc
    _require(len(data) <= limit, f"{name} exceeds bounded size")
    return data


def _hash_regular(path: Path, name: str, *, limit: int) -> str:
    _require(path.is_file() and not path.is_symlink(), f"{name} missing or symlinked")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as reader:
            info = os.fstat(reader.fileno())
            _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit,
                     f"{name} is not a nonempty bounded regular file")
            consumed = 0
            while block := reader.read(1024 * 1024):
                consumed += len(block)
                _require(consumed <= limit, f"{name} grew beyond bound")
                digest.update(block)
            _require(consumed == info.st_size, f"{name} size changed while hashing")
    except OSError as exc:
        raise SnapshotAttestationError(f"{name} unreadable") from exc
    return digest.hexdigest()


def _safetensors_header(path: Path, name: str) -> tuple[dict[str, dict], int]:
    """Parse only metadata; fail on holes/overlaps/incorrect element sizes."""
    _require(path.is_file() and not path.is_symlink(), f"{name} missing or symlinked")
    try:
        with path.open("rb") as reader:
            info = os.fstat(reader.fileno())
            _require(stat.S_ISREG(info.st_mode) and 8 < info.st_size <= MAX_WEIGHT_BYTES,
                     f"{name} not a bounded safetensors regular file")
            prefix = reader.read(8)
            _require(len(prefix) == 8, f"{name} incomplete header prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            _require(2 <= header_size <= MAX_HEADER_BYTES and 8 + header_size < info.st_size,
                     f"{name} malformed header length")
            raw = reader.read(header_size)
            _require(len(raw) == header_size, f"{name} truncated header")
    except OSError as exc:
        raise SnapshotAttestationError(f"{name} cannot be inspected") from exc
    head = _strict_map(raw, f"{name} header")
    meta = head.pop("__metadata__", None)
    if meta is not None:
        _require(type(meta) is dict and all(type(k) is str and type(v) is str
                                           for k, v in meta.items()),
                 f"{name} invalid metadata")
    _require(bool(head) and len(head) <= MAX_WEIGHT_MAP_ENTRIES,
             f"{name} empty or excessive tensor inventory")
    intervals: list[tuple[int, int]] = []
    payload_length = info.st_size - 8 - header_size
    tensor_bytes = 0
    for tensor_name, tensor in head.items():
        _require(type(tensor_name) is str and tensor_name and len(tensor_name) <= 512,
                 f"{name} invalid tensor name")
        _require(type(tensor) is dict and set(tensor) == {"dtype", "shape", "data_offsets"},
                 f"{name} invalid tensor metadata")
        shape, bounds, dtype = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
        _require(type(shape) is list and len(shape) <= 12
                 and all(type(dim) is int and 0 <= dim <= 2**32 for dim in shape),
                 f"{name} invalid tensor shape")
        _require(type(bounds) is list and len(bounds) == 2
                 and all(type(v) is int and 0 <= v <= payload_length for v in bounds),
                 f"{name} invalid tensor offsets")
        _require(type(dtype) is str and dtype in DTYPE_BYTES,
                 f"{name} unsupported tensor dtype")
        start, end = bounds
        _require(start <= end, f"{name} reversed tensor offsets")
        count = 1
        for dim in shape:
            count *= dim
            _require(count <= MAX_WEIGHT_BYTES, f"{name} tensor element count exceeds bound")
        size = count * DTYPE_BYTES[dtype]
        _require(size == end - start, f"{name} tensor shape/dtype byte count contradicts offsets")
        tensor_bytes += size
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        _require(start == cursor, f"{name} tensor offsets overlap or contain holes")
        cursor = end
    _require(cursor == payload_length and tensor_bytes == payload_length,
             f"{name} tensor offsets do not cover entire payload")
    return head, tensor_bytes


def audit_snapshot(base_binding_path: Path, *, expected_binding_sha256: str,
                   expected_config_sha256: str, expected_tokenizer_sha256: str,
                   source_root: Path) -> dict:
    """Verify offline byte inventory, returning HOLD for all runtime claims.

    The caller-provided pins are NOT authenticated by this procedure. A host
    authority must separately prove provenance and immutable storage custody.
    """
    for label, value in (("binding", expected_binding_sha256),
                         ("config", expected_config_sha256),
                         ("tokenizer", expected_tokenizer_sha256)):
        _require(_is_sha(value), f"{label} independent SHA256 pin missing")
    binding_data = _read_small(Path(base_binding_path), "base binding", MAX_BINDING_BYTES)
    _require(hashlib.sha256(binding_data).hexdigest() == expected_binding_sha256,
             "independently pinned binding digest mismatch")
    binding = _strict_map(binding_data, "base binding")
    _require(binding.get("schema") == "TRAININGDAY_PINNED_BASE_V1"
             and binding.get("model_id") == MODEL_ID
             and binding.get("revision") == PINNED_REVISION
             and binding.get("loader_class") == RUNNER_LOADER
             and binding.get("quantization") == "NF4_DOUBLE_BF16",
             "base model, revision, loader or quantization changed")
    snapshot_text = binding.get("snapshot_path")
    _require(type(snapshot_text) is str and bool(snapshot_text),
             "base snapshot path missing")
    snapshot = Path(snapshot_text)
    _require(snapshot.is_absolute() and snapshot.is_dir() and not snapshot.is_symlink(),
             "base snapshot must be absolute nonsymlink directory")
    files = binding.get("files")
    _require(type(files) is dict and 4 <= len(files) <= MAX_DECLARED_FILES,
             "base inventory missing or excessive")
    _require(all(type(k) is str and SAFE_FILE.fullmatch(k) is not None
                 and k not in {".", ".."} and _is_sha(v)
                 for k, v in files.items()), "invalid file inventory declaration")
    for required in ("config.json", "tokenizer_config.json", "tokenizer.json",
                     "vocab.json", "merges.txt", "chat_template.jinja",
                     "model.safetensors.index.json"):
        _require(required in files, f"required {required} not pinned")
    for possible_override in ("special_tokens_map.json", "added_tokens.json",
                              "generation_config.json", "tokenizer.model"):
        override_path = snapshot / possible_override
        _require(not (override_path.exists() or override_path.is_symlink())
                 or possible_override in files,
                 f"unbound optional tokenizer/generation override: {possible_override}")
    _require(files["config.json"] == expected_config_sha256
             and files["tokenizer_config.json"] == expected_tokenizer_sha256,
             "independent config/tokenizer pins disagree with binding")

    observed: dict[str, str] = {}
    for name, sha in sorted(files.items()):
        file_limit = MAX_WEIGHT_BYTES if name.endswith(".safetensors") else MAX_SMALL_FILE_BYTES
        observed[name] = _hash_regular(snapshot / name, name, limit=file_limit)
        _require(observed[name] == sha, f"{name} digest differs from pinned binding")

    # R75 validates the official multimodal config declaration and statically
    # guarded text-only loader selection in BOTH experimental training arms.
    source = source_preflight(Path(base_binding_path),
                              expected_config_sha256=expected_config_sha256,
                              source_root=Path(source_root))
    _require(source["status"] == "DECLARED_MISMATCH_UNKNOWN_RUNTIME",
             "R75 loader source decision unexpectedly changed")
    tokenizer = _strict_map(_read_small(snapshot / "tokenizer_config.json",
                                        "tokenizer_config.json"), "tokenizer_config")
    # The official pinned repository publishes a Qwen2Tokenizer declaration
    # with a tokenizer.json plus separate vocab/merges. A valid chat template
    # alone is not evidence the token-to-ID mapping assets exist.
    _require(tokenizer.get("tokenizer_class") == "Qwen2Tokenizer",
             "pinned tokenizer class differs from official declaration")
    template = tokenizer.get("chat_template")
    eos = tokenizer.get("eos_token")
    _require(type(template) is str and bool(template.strip()),
             "pinned tokenizer has no declared chat template")
    _require((type(eos) is str and bool(eos))
             or (type(eos) is dict and type(eos.get("content")) is str
                 and bool(eos["content"])), "pinned tokenizer EOS declaration missing")
    _require(bool(_strict_map(_read_small(snapshot / "tokenizer.json", "tokenizer.json"),
                              "tokenizer model JSON")), "pinned tokenizer model is empty")
    _require(bool(_strict_map(_read_small(snapshot / "vocab.json", "vocab.json"),
                              "tokenizer vocab JSON")), "pinned tokenizer vocab is empty")
    _require(bool(_read_small(snapshot / "merges.txt", "merges.txt").strip()),
             "pinned BPE merges file empty")
    try:
        template_source = _read_small(snapshot / "chat_template.jinja", "chat_template.jinja")
        _require(bool(template_source.decode("utf-8").strip()),
                 "pinned standalone chat template empty")
    except UnicodeError as exc:
        raise SnapshotAttestationError("pinned standalone chat template is not UTF-8") from exc

    index = _strict_map(_read_small(snapshot / "model.safetensors.index.json",
                                    "model.safetensors.index.json"), "safetensors index")
    _require(set(index) == {"metadata", "weight_map"}
             and type(index["metadata"]) is dict
             and type(index["metadata"].get("total_size")) is int
             and 0 < index["metadata"]["total_size"] <= 12 * 1024**3,
             "safetensors index size metadata incomplete")
    weight_map = index["weight_map"]
    _require(type(weight_map) is dict and 0 < len(weight_map) <= MAX_WEIGHT_MAP_ENTRIES,
             "safetensors weight map missing or excessive")
    _require(all(type(name) is str and bool(name) and len(name) <= 512
                 and type(shard) is str and SHARD_NAME.fullmatch(shard) is not None
                 for name, shard in weight_map.items()),
             "safetensors weight map has invalid tensor/shard names")
    declared_shards = {name for name in files if name.endswith(".safetensors")}
    indexed_shards = set(weight_map.values())
    _require(len(indexed_shards) == 2 and declared_shards == indexed_shards,
             "Qwen3.5-4B expected exactly two indexed and pinned weight shards")
    _require(indexed_shards == {f"model.safetensors-{i:05d}-of-00002.safetensors" for i in (1, 2)},
             "Qwen3.5-4B shards are not the expected numbered pair")
    found_names: set[str] = set()
    aggregate_bytes = 0
    tensor_count = 0
    for shard in sorted(indexed_shards):
        metadata, byte_count = _safetensors_header(snapshot / shard, shard)
        expected_names = {name for name, mapped_shard in weight_map.items()
                          if mapped_shard == shard}
        _require(set(metadata) == expected_names,
                 "safetensors index and actual shard tensor names differ")
        _require(not found_names.intersection(metadata),
                 "duplicate tensor names across safetensors shards")
        found_names.update(metadata)
        tensor_count += len(metadata)
        aggregate_bytes += byte_count
    _require(tensor_count == len(weight_map)
             and aggregate_bytes == index["metadata"]["total_size"],
             "safetensors indexed total tensor size contradicts headers")
    # `glob` is a second inventory observation; neither this nor earlier
    # hashing constitutes a TOCTOU-proof host custody operation.
    _require({p.name for p in snapshot.glob("*.safetensors")} == declared_shards,
             "extra, missing or unbound model weight shards")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_SNAPSHOT_INVENTORY_VALID_RUNTIME_HOLD",
        "evidence_class": "LOCAL_SOURCE_AND_FILE_INVENTORY_ONLY",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_declared_architecture": "Qwen3_5ForConditionalGeneration",
        "candidate_text_loader": RUNNER_LOADER,
        "text_only_compatibility_status": "DOCUMENTED_POSSIBILITY_NOT_PHYSICALLY_TESTED",
        "binding_sha256": expected_binding_sha256,
        "config_sha256": expected_config_sha256,
        "tokenizer_sha256": expected_tokenizer_sha256,
        "source_sha256": source["source_sha256"],
        "observed_file_sha256": observed,
        "safetensors_shard_count": len(declared_shards),
        "safetensors_tensor_count": tensor_count,
        "safetensors_indexed_tensor_bytes": aggregate_bytes,
        "tokenizer_chat_template_declared": True,
        "tokenizer_eos_declared": True,
        "physical_tokenizer_verified": False,
        "runtime_loader_verified": False,
        "actual_model_tensors_materialized": False,
        "finite_forward_loss_verified": False,
        "external_pin_provenance_verified": False,
        "host_immutable_custody_verified": False,
        "exclusive_cuda_lease_verified": False,
        "authorizes_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R140 offline Qwen snapshot source inspection")
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        result = audit_snapshot(args.base_binding,
                                expected_binding_sha256=args.expected_binding_sha256,
                                expected_config_sha256=args.expected_config_sha256,
                                expected_tokenizer_sha256=args.expected_tokenizer_sha256,
                                source_root=args.source_root)
    except (SnapshotAttestationError, LoaderPreflightError, OSError) as exc:
        parser.exit(2, f"HOLD: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
