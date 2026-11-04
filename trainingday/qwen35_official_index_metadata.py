"""Inspect the official Qwen3.5-4B shard index without downloading model weights.

This is an independently byte-pinned, metadata-only storage preflight. It is
not shard authentication, checkpoint loading, training, GPU qualification, or
an authorization to download or execute the protected training job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
INDEX_SHA256 = "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
INDEX_BYTE_SIZE = 76_196
PUBLISHED_TENSOR_PAYLOAD_BYTES = 9_319_737_856
EXPECTED_NAMESPACE_COUNTS = {"language": 426, "vision": 297, "mtp": 15}
EXPECTED_SHARDS = frozenset({
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
})
MAX_INDEX_BYTES = 128 * 1024
_TENSOR_NAME = re.compile(r"[A-Za-z0-9_.]+\Z")


class OfficialIndexMetadataError(ValueError):
    """The supposedly official checkpoint index did not match its source pin."""


def _unique_pairs(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise OfficialIndexMetadataError("duplicate index key")
        record[key] = value
    return record


def _reject_constant(_value):
    raise OfficialIndexMetadataError("nonfinite index value")


def _namespace(key: str) -> str:
    if key.startswith("model.language_model."):
        return "language"
    if key.startswith("model.visual."):
        return "vision"
    if key.startswith("mtp."):
        return "mtp"
    raise OfficialIndexMetadataError("unexpected tensor namespace")


def _parse_index(raw: bytes, *, sha256: str, byte_size: int,
                 payload_bytes: int, namespace_counts: dict,
                 shard_names: frozenset[str]) -> dict:
    """Pure validation seam; only ``inspect_official_index`` supplies trust pins."""
    if (type(raw) is not bytes or len(raw) != byte_size
            or hashlib.sha256(raw).hexdigest() != sha256):
        raise OfficialIndexMetadataError("raw index bytes differ from pinned source")
    try:
        document = json.loads(raw.decode("utf-8"),
                              object_pairs_hook=_unique_pairs,
                              parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise OfficialIndexMetadataError("invalid index JSON") from exc
    if (type(document) is not dict or set(document) != {"metadata", "weight_map"}
            or type(document["metadata"]) is not dict
            or document["metadata"] != {"total_size": payload_bytes}
            or type(document["weight_map"]) is not dict
            or len(document["weight_map"]) != sum(namespace_counts.values())):
        raise OfficialIndexMetadataError("official index inventory drifted")

    counts = {namespace: 0 for namespace in namespace_counts}
    used_shards = set()
    for key, shard in document["weight_map"].items():
        if (type(key) is not str or not _TENSOR_NAME.fullmatch(key)
                or type(shard) is not str or shard not in shard_names):
            raise OfficialIndexMetadataError("foreign tensor or shard entry")
        counts[_namespace(key)] += 1
        used_shards.add(shard)
    if counts != namespace_counts or used_shards != shard_names:
        raise OfficialIndexMetadataError("official tensor namespaces or shard use drifted")
    if "model.language_model.embed_tokens.weight" not in document["weight_map"]:
        raise OfficialIndexMetadataError("official text embeddings absent")

    return {
        "status": "OFFICIAL_QWEN35_WEIGHT_INDEX_METADATA_ONLY_HOLD",
        "official_model": "Qwen/Qwen3.5-4B",
        "official_revision": OFFICIAL_REVISION,
        "index_sha256": sha256,
        "index_bytes_verified": byte_size,
        "indexed_tensor_count": len(document["weight_map"]),
        "indexed_namespace_counts": counts,
        "referenced_safetensors_shards": sorted(used_shards),
        "published_tensor_payload_bytes": payload_bytes,
        "published_payload_gib_approx": round(payload_bytes / (1024 ** 3), 3),
        "checkpoint_storage_requirement": "AT_LEAST_DECLARED_PAYLOAD_PLUS_HEADERS_AND_STAGING",
        "safetensors_shard_bytes_authenticated": False,
        "safetensors_shards_downloaded": False,
        "pretrained_weights_loaded": False,
        "physical_training_authorized": False,
        "gpu_fit": "UNKNOWN",
        "host_memory_fit": "UNKNOWN",
    }


def inspect_official_index(index_path: Path) -> dict:
    """Read exact official index bytes through one bounded regular-file FD."""
    index_path = Path(index_path)
    if index_path.is_symlink():
        raise OfficialIndexMetadataError("index cannot be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(index_path, flags)
        try:
            metadata = os.fstat(fd)
            if (not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size != INDEX_BYTE_SIZE
                    or metadata.st_size > MAX_INDEX_BYTES):
                raise OfficialIndexMetadataError("index is not the expected bounded file")
            raw = os.read(fd, MAX_INDEX_BYTES + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise OfficialIndexMetadataError("index metadata inaccessible") from exc
    return _parse_index(raw, sha256=INDEX_SHA256, byte_size=INDEX_BYTE_SIZE,
                        payload_bytes=PUBLISHED_TENSOR_PAYLOAD_BYTES,
                        namespace_counts=EXPECTED_NAMESPACE_COUNTS,
                        shard_names=EXPECTED_SHARDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned official Qwen3.5 index metadata only")
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_official_index(args.index), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
