"""Offline byte-pin verifier for the exact official Qwen3.5-4B R18 base.

This checks local files and a caller-supplied hash declaration. It is NOT a
source-rights grant, trusted signer, host custody proof, or GPU launch lease.
The distinct public Aggressive/uncensored model manifest is not this base.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from trainingday.official_public_base import (
    INDEX_TENSOR_BYTES, METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS,
    SHARDS as OFFICIAL_SHARDS,
)


BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SCHEMA = "TRAININGDAY_PINNED_BASE_V1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9_.-]{1,120}\Z")
SHARD = re.compile(r"model\.safetensors-[0-9]{5}-of-[0-9]{5}\.safetensors\Z")
MAX_BINDING_BYTES = 1024 * 1024
MAX_FILES = 128


class BaseBindingError(ValueError):
    """An untrusted declaration or local model snapshot failed validation."""


def _require(condition: bool):
    if not condition:
        raise BaseBindingError("official R18 base binding rejected")


def _unique_pairs(pairs):
    record = {}
    for key, value in pairs:
        _require(key not in record)
        record[key] = value
    return record


def _bad_constant(_value):
    raise BaseBindingError("official R18 base binding rejected")


def _read_regular_nofollow(path: Path, *, limit: int | None = None) -> bytes:
    """Use a single FD for read/hash; prevent final-component symlink swap."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
        try:
            meta = os.fstat(fd)
            _require(stat.S_ISREG(meta.st_mode))
            if limit is not None:
                _require(meta.st_size <= limit)
            chunks = []
            size = 0
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                size += len(block)
                if limit is not None:
                    _require(size <= limit)
                chunks.append(block)
            return b"".join(chunks)
        finally:
            os.close(fd)
    except OSError as exc:
        raise BaseBindingError("official R18 base file unavailable") from exc


def _sha_file(path: Path) -> str:
    """Stream potentially multi-GiB shards with bounded host RAM."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
        try:
            _require(stat.S_ISREG(os.fstat(fd).st_mode))
            digest = hashlib.sha256()
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            return digest.hexdigest()
        finally:
            os.close(fd)
    except OSError as exc:
        raise BaseBindingError("official R18 base file unavailable") from exc


def inspect_untrusted_base_binding_for_native_test_only(
    binding_path: str | Path,
) -> tuple[dict, Path]:
    """Inspect caller-declared file hashes, never official source authenticity.

    Synthetic tests use this seam. A successful match is NOT official model
    admission, trusted metadata custody, or permission to run a model.
    """
    _require(isinstance(binding_path, (str, Path)))
    binding = Path(binding_path)
    if not binding.is_absolute():
        binding = Path.cwd() / binding
    _require(binding.is_file() and not binding.is_symlink())
    try:
        raw = _read_regular_nofollow(binding, limit=MAX_BINDING_BYTES)
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_bad_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise BaseBindingError("official R18 base strict JSON invalid") from exc
    _require(type(value) is dict and value.get("schema") == SCHEMA
             and value.get("model_id") == BASE_ID
             and value.get("revision") == BASE_REVISION
             and value.get("loader_class") == "Qwen3_5ForCausalLM"
             and value.get("quantization") == "NF4_DOUBLE_BF16")
    snapshot_raw = value.get("snapshot_path")
    _require(type(snapshot_raw) is str and snapshot_raw)
    snapshot = Path(snapshot_raw)
    _require(snapshot.is_absolute() and snapshot.is_dir()
             and all(not part.is_symlink() for part in (snapshot, *snapshot.parents)))
    files = value.get("files")
    _require(type(files) is dict and 5 <= len(files) <= MAX_FILES)
    for required in ("config.json", "tokenizer_config.json", "tokenizer.json",
                     "model.safetensors.index.json"):
        _require(required in files)
    bound_shards = set()
    for name, expected_sha in files.items():
        _require(type(name) is str and NAME.fullmatch(name) is not None
                 and name not in {".", ".."} and type(expected_sha) is str
                 and HEX64.fullmatch(expected_sha) is not None)
        if name.endswith(".safetensors"):
            _require(SHARD.fullmatch(name) is not None)
            bound_shards.add(name)
        current = snapshot / name
        _require(current.is_file() and not current.is_symlink()
                 and _sha_file(current) == expected_sha)
    _require(bound_shards and {p.name for p in snapshot.glob("*.safetensors")}
             == bound_shards)
    return value, snapshot


# These file identities are present at official revision 851bf6e, but their
# trusted full-byte SHA256 values have NOT yet been independently admitted.
# Filling this tuple is a separately reviewed source change, never caller input.
# Empty by design: require_official_base MUST remain fail-closed.
TRUSTED_OFFICIAL_METADATA_PIN_ITEMS: tuple[tuple[str, str], ...] = ()


def require_official_base(binding_path: str | Path) -> tuple[dict, Path]:
    """Require complete official shard identity AND trusted metadata byte pins.

    This production-facing source gate currently has no enrolled metadata
    trust pins and thus cannot admit ANY checkpoint. It also confers no GPU
    lease, owner signature, host path custody, rights, or launch authority.
    Tests of tiny synthetic fixtures must call the explicitly untrusted
    inspector instead.
    """
    declared, snapshot = inspect_untrusted_base_binding_for_native_test_only(
        binding_path)
    declared_files = declared["files"]
    shard_names = {name for name, _, _ in OFFICIAL_SHARDS}
    required_metadata = set(METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS)
    _require({name for name in declared_files if name.endswith(".safetensors")}
             == shard_names)
    _require(required_metadata.issubset(declared_files))
    _require(set(declared_files) == shard_names | required_metadata)

    # The declared values are untrusted; compare the ACTUAL hashed local bytes
    # (checked above) with immutable independently reviewed official shard
    # byte pins, including physical size. No LFS/Xet pointer is a weight file.
    for name, official_size, official_digest in OFFICIAL_SHARDS:
        _require(declared_files[name] == official_digest)
        try:
            _require((snapshot / name).stat().st_size == official_size)
        except OSError as exc:
            raise BaseBindingError("official shard size unavailable") from exc

    # Check that the named index really maps tensors into both official
    # shards, not a caller-crafted one-shard or dimension-only stand-in.
    try:
        raw_index = _read_regular_nofollow(
            snapshot / "model.safetensors.index.json", limit=1024 * 1024)
        index = json.loads(raw_index.decode("utf-8"),
                           object_pairs_hook=_unique_pairs,
                           parse_constant=_bad_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise BaseBindingError("official index unavailable or invalid") from exc
    _require(type(index) is dict and set(index) == {"metadata", "weight_map"}
             and type(index["metadata"]) is dict
             and type(index["metadata"].get("total_size")) is int
             and index["metadata"]["total_size"] == INDEX_TENSOR_BYTES
             and type(index["weight_map"]) is dict
             and bool(index["weight_map"])
             and all(type(name) is str and type(shard) is str
                     and shard in shard_names
                     for name, shard in index["weight_map"].items())
             and set(index["weight_map"].values()) == shard_names)

    # No trusted SHA256 pins yet for config/tokenizer/index/chat template/
    # BPE vocab+merges. A caller-supplied manifest cannot supply them.
    pins = dict(TRUSTED_OFFICIAL_METADATA_PIN_ITEMS)
    _require(len(pins) == len(TRUSTED_OFFICIAL_METADATA_PIN_ITEMS)
             and set(pins) == required_metadata
             and all(type(digest) is str and HEX64.fullmatch(digest)
                     and declared_files[name] == digest
                     for name, digest in pins.items()))
    return declared, snapshot
