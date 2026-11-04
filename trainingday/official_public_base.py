"""Source-only identity contract for the official Qwen3.5-4B checkpoint.

This module compares a repository manifest with pinned upstream LFS pointer
metadata. It never downloads, opens weight shards, grants training rights,
signs an attestation, or authorizes a physical launch. A broker must require
independent trusted signer, host, path, local byte, workload and lease proofs.
"""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA = "TRAININGDAY_OFFICIAL_QWEN35_4B_SOURCE_V1"
OFFICIAL_ID = "Qwen/Qwen3.5-4B"
OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SHARDS = (
    (
        "model.safetensors-00001-of-00002.safetensors",
        5329398688,
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    ),
    (
        "model.safetensors-00002-of-00002.safetensors",
        3990429408,
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
    ),
)
INDEX_TENSOR_BYTES = 9319737856
PHYSICAL_SHARD_BYTES = sum(item[1] for item in SHARDS)
MAX_MANIFEST_BYTES = 16384
METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS = (
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "model.safetensors.index.json",
)


class OfficialBaseSourceError(ValueError):
    """Manifest is untrusted or is not the exact official upstream identity."""


def _unique_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise OfficialBaseSourceError("duplicate manifest key")
        output[key] = value
    return output


def _reject_constant(_value):
    raise OfficialBaseSourceError("nonfinite manifest number")


def expected_source_record() -> dict:
    """Return the exact source record; *no* signer or launch authorization."""
    return {
        "schema": SCHEMA,
        "kind": "source_only_official_public_checkpoint_metadata",
        "identity": {
            "provider": "Hugging Face",
            "repository": OFFICIAL_ID,
            "revision": OFFICIAL_REVISION,
            "license_declared": "apache-2.0",
            "checkpoint_architecture": "Qwen3_5ForConditionalGeneration",
            "text_projection": "model.language_model.* -> model.*",
            "text_loader": "Qwen3_5ForCausalLM",
        },
        "safetensors": {
            "index_metadata_tensor_bytes": INDEX_TENSOR_BYTES,
            "physical_shard_bytes": PHYSICAL_SHARD_BYTES,
            "shards": [
                {"path": name, "size_bytes": size, "sha256_lfs_oid": digest}
                for name, size, digest in SHARDS
            ],
        },
        "evidence": {
            "class": "UPSTREAM_LFS_POINTER_METADATA_NOT_LOCAL_WEIGHT_VERIFICATION",
            "pinned_revision_url": (
                "https://huggingface.co/Qwen/Qwen3.5-4B/tree/"
                + OFFICIAL_REVISION
            ),
            "index_url": (
                "https://huggingface.co/Qwen/Qwen3.5-4B/raw/"
                + OFFICIAL_REVISION + "/model.safetensors.index.json"
            ),
        },
        "admission": {
            "status": "SOURCE_ONLY_NOT_SIGNED_NOT_BROKER_ADMITTED",
            "missing_trusted_byte_pins": list(METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS),
            "signature": None,
            "physical_launch_allowed": False,
            "protected_training_data_allowed": False,
        },
    }


def validate_official_source_record(document: object) -> dict:
    """Reject modified/shape-compatible/third-party source records, exactly."""
    if type(document) is not dict or document != expected_source_record():
        raise OfficialBaseSourceError("not the pinned official public source record")
    return document


def read_official_source_record(path: str | Path) -> dict:
    """Parse bounded strict JSON and validate against separate code constants.

    Passing establishes exact metadata equality only. Neither the manifest nor
    caller-supplied keys can confer trusted-signer or broker custody status.
    """
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise OfficialBaseSourceError("manifest path not a regular file")
    try:
        with candidate.open("rb") as source:
            raw = source.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise OfficialBaseSourceError("manifest too large")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                              parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise OfficialBaseSourceError("invalid official source manifest") from exc
    return validate_official_source_record(document)
