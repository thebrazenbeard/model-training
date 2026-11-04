"""Offline architecture and optimistic memory accounting for Qwen3.5-4B.

The evidence is caller-supplied *metadata bytes* and caller-supplied digests,
not independently authenticated official model custody. This module never
loads model weights, imports torch, probes CUDA, or grants physical training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from trainingday.pinned_base import BASE_ID, BASE_REVISION


SCHEMA = "TRAININGDAY_QWEN35_METADATA_MEMORY_ADMISSION_V1"
OFFICIAL_REVISION_URL = (
    "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/"
)
# SHA256 over raw bytes fetched from the pinned *official* Hugging Face
# revision; these are not Git object IDs or Xet identifiers.
OFFICIAL_CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
OFFICIAL_INDEX_SHA256 = "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
OFFICIAL_SHARD_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors":
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors":
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
BF16_CHECKPOINT_BYTES = 9_319_737_856
DECODER_LAYERS = 32
GDN_LAYERS = 24
ATTENTION_LAYERS = 8
VISION_LAYERS = 24
VOCAB_SIZE = 248_320
HIDDEN_SIZE = 2_560
SHARD_NAMES = frozenset({
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_CONFIG_BYTES = 64 * 1024
MAX_INDEX_BYTES = 1024 * 1024


class MetadataAdmissionError(ValueError):
    """Untrusted bytes or declarations cannot establish the source contract."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise MetadataAdmissionError(reason)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate metadata JSON field")
        result[key] = value
    return result


def _reject_constant(_value):
    raise MetadataAdmissionError("non-finite metadata JSON value")


def _strict_json(raw: bytes) -> dict:
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MetadataAdmissionError("malformed metadata JSON") from exc
    _require(type(obj) is dict, "metadata JSON root must be object")
    return obj


def _read_bounded_file(path: Path, max_bytes: int) -> bytes:
    path = Path(path)
    _require(not path.is_symlink(), "metadata path must not be symlink")
    # A FIFO can block indefinitely in os.open() *before* fstat rejects it.
    # O_NONBLOCK makes nonregular path rejection bounded on POSIX; regular
    # file reads are unaffected. No GPU or model weights are touched.
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        fd = os.open(path, flags)
        try:
            st = os.fstat(fd)
            _require(stat.S_ISREG(st.st_mode) and st.st_size <= max_bytes,
                     "metadata must be a bounded regular file")
            data = bytearray()
            while True:
                chunk = os.read(fd, min(65_536, max_bytes + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                _require(len(data) <= max_bytes, "metadata file too large")
            return bytes(data)
        finally:
            os.close(fd)
    except OSError as exc:
        raise MetadataAdmissionError("metadata file inaccessible") from exc


def _check_digest(raw: bytes, expected: str, label: str) -> None:
    _require(type(expected) is str and SHA256_RE.fullmatch(expected) is not None,
             f"{label} SHA256 missing or malformed")
    _require(hashlib.sha256(raw).hexdigest() == expected,
             f"{label} bytes differ from claimed SHA256")


def missing_source_report() -> dict:
    """Explicit non-admission when authentic metadata bytes are absent."""
    return {
        "schema": SCHEMA,
        "status": "OFFICIAL_METADATA_BYTES_MISSING_HARD_HOLD",
        "base_model_claim": BASE_ID,
        "base_revision_claim": BASE_REVISION,
        "metadata_source_url": OFFICIAL_REVISION_URL,
        "architecture_shape_checked": False,
        "source_bytes_match_declared_digests": False,
        "official_origin_authenticated": False,
        "official_weight_shards_authenticated": False,
        "pinned_official_metadata_bytes_verified": False,
        "hardware_fit": "UNKNOWN",
        "training_authorized": False,
        "optimizer_steps_observed": 0,
    }


def audit_metadata(config_bytes: bytes, index_bytes: bytes, *,
                   config_sha256: str, index_sha256: str) -> dict:
    """Audit declared SHA-bound config/index structure without reading weights.

    The SHA claims come from the caller: byte integrity is checked, but their
    authoritative origin is NOT. A positive report cannot qualify a model or
    satisfy physical, rights, evaluator or Q6 gates.
    """
    _require(type(config_bytes) is bytes and len(config_bytes) <= MAX_CONFIG_BYTES,
             "invalid/oversized model configuration")
    _require(type(index_bytes) is bytes and len(index_bytes) <= MAX_INDEX_BYTES,
             "invalid/oversized checkpoint index")
    _check_digest(config_bytes, config_sha256, "config")
    _check_digest(index_bytes, index_sha256, "index")
    config = _strict_json(config_bytes)
    index = _strict_json(index_bytes)

    text = config.get("text_config")
    vision = config.get("vision_config")
    _require(type(text) is dict and type(vision) is dict,
             "nested Qwen3.5 text/vision configs missing")
    _require(config.get("model_type") == "qwen3_5"
             and config.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
             and text.get("model_type") == "qwen3_5_text"
             and text.get("tie_word_embeddings") is True
             and text.get("num_hidden_layers") == DECODER_LAYERS
             and text.get("vocab_size") == VOCAB_SIZE
             and text.get("hidden_size") == HIDDEN_SIZE
             and text.get("mtp_num_hidden_layers") == 1
             and vision.get("depth") == VISION_LAYERS,
             "pinned Qwen3.5 architecture contract mismatch")
    types = text.get("layer_types")
    _require(type(types) is list and len(types) == DECODER_LAYERS
             and all(type(t) is str for t in types)
             and Counter(types) == {"linear_attention": GDN_LAYERS,
                                    "full_attention": ATTENTION_LAYERS},
             "hybrid decoder layer inventory mismatch")

    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    _require(type(metadata) is dict and type(weight_map) is dict
             and type(metadata.get("total_size")) is int
             and metadata["total_size"] == BF16_CHECKPOINT_BYTES
             and 100 <= len(weight_map) <= 16_000,
             "pinned BF16 index size/inventory mismatch")
    _require(all(type(k) is str and type(v) is str and v in SHARD_NAMES
                 for k, v in weight_map.items())
             and set(weight_map.values()) == SHARD_NAMES,
             "checkpoint index contains unexpected shard names")
    namespaces = {
        "text": sum(name.startswith("model.language_model.") for name in weight_map),
        "vision": sum(name.startswith("model.visual.") for name in weight_map),
        "mtp": sum(name.startswith("mtp.") for name in weight_map),
    }
    _require(all(namespaces.values()) and sum(namespaces.values()) == len(weight_map),
             "checkpoint has missing/foreign tensor namespace")
    _require("model.language_model.embed_tokens.weight" in weight_map,
             "language embedding absent from official index")
    for layer, layer_type in enumerate(types):
        prefix = f"model.language_model.layers.{layer}."
        suffixes = (
            ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
             "linear_attn.out_proj.weight")
            if layer_type == "linear_attention" else
            ("self_attn.q_proj.weight", "self_attn.v_proj.weight")
        )
        _require(all(prefix + suffix in weight_map
                     for suffix in (*suffixes, "mlp.down_proj.weight")),
                 f"decoder layer {layer} missing expected attention projection")

    embedding_bytes = VOCAB_SIZE * HIDDEN_SIZE * 2
    # Deliberately optimistic ALL-checkpoint estimate, not a text-only or
    # offload memory lower bound. Quantization scales and runtime omitted.
    idealized = (BF16_CHECKPOINT_BYTES - embedding_bytes) // 4 + embedding_bytes
    return {
        "schema": SCHEMA,
        "status": "SOURCE_METADATA_CONSISTENT_HARDWARE_FIT_UNKNOWN",
        "base_model_claim": BASE_ID,
        "base_revision_claim": BASE_REVISION,
        "metadata_source_url": OFFICIAL_REVISION_URL,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "architecture_shape_checked": True,
        "source_bytes_match_declared_digests": True,
        "official_origin_authenticated": False,
        "official_weight_shards_authenticated": False,
        "pinned_official_metadata_bytes_verified": False,
        "official_shard_sha256_remote_metadata_only": OFFICIAL_SHARD_SHA256,
        "checkpoint_namespaces": namespaces,
        "decoder_layers": DECODER_LAYERS,
        "gated_deltanet_layers": GDN_LAYERS,
        "full_attention_layers": ATTENTION_LAYERS,
        "vision_layers": VISION_LAYERS,
        "mtp_layers": 1,
        "minimal_lora_targets": ["q_proj", "v_proj"],
        "minimal_lora_scope": "trainingday.lowmem_experiment.nf4_minimal_gpu_ONLY",
        "minimal_lora_full_attention_layers": ATTENTION_LAYERS,
        "minimal_lora_gdn_layers": 0,
        "gdn_lora_targets_if_expanded": ["linear_attn.in_proj_qkv",
                                         "linear_attn.in_proj_z", "linear_attn.out_proj"],
        "lora_coverage": "LOWMEM_PILOT_ONLY_EIGHT_FULL_ATTENTION_LAYERS",
        "bf16_checkpoint_bytes": BF16_CHECKPOINT_BYTES,
        "bf16_tied_embedding_bytes": embedding_bytes,
        "idealized_full_checkpoint_nf4_plus_bf16_embedding_bytes": idealized,
        "estimate_excludes": ["quantization metadata", "other unquantized tensors",
                              "adapter and optimizer state", "activations",
                              "CUDA runtime/other processes"],
        "text_only_projection_may_drop": ["model.visual.*", "mtp.*"],
        "hardware_fit": "UNKNOWN",
        "training_authorized": False,
        "optimizer_steps_observed": 0,
    }


def admit_pinned_official_metadata(config_bytes: bytes, index_bytes: bytes) -> dict:
    """Require raw official-revision metadata digests, not caller-self hashes.

    This authenticates *byte equality to source-pinned metadata* only. No
    weights are downloaded/read, and official model-shard custody remains
    unverified until separate host-owned readback against the pinned digests.
    """
    result = audit_metadata(
        config_bytes, index_bytes,
        config_sha256=OFFICIAL_CONFIG_SHA256,
        index_sha256=OFFICIAL_INDEX_SHA256,
    )
    result["status"] = "PINNED_OFFICIAL_METADATA_BYTES_VERIFIED_HARDWARE_FIT_UNKNOWN"
    result["pinned_official_metadata_bytes_verified"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if not all((args.config, args.index)):
        print(json.dumps(missing_source_report(), sort_keys=True))
        return 2
    try:
        report = admit_pinned_official_metadata(
            _read_bounded_file(args.config, MAX_CONFIG_BYTES),
            _read_bounded_file(args.index, MAX_INDEX_BYTES))
    except MetadataAdmissionError as exc:
        print(json.dumps({**missing_source_report(),
                          "status": "SOURCE_METADATA_REJECTED_HARD_HOLD",
                          "reason": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
