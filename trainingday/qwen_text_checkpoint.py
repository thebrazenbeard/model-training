"""Fail-closed text projection for the pinned official Qwen3.5 multimodal base.

The R18 language trainer targets Qwen3_5ForCausalLM while its official
checkpoint stores language weights beneath ``model.language_model.*``. A
mapping must be proved complete before loading any real weights. This code
does not fetch files, allocate CUDA or grant protected training authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


INDEX_NAME = "model.safetensors.index.json"
TEXT_PREFIX = "model.language_model."
IGNORED_PREFIXES = ("model.visual.", "mtp.")
MAX_INDEX_BYTES = 1024 * 1024
MAX_KEYS = 16000
SHARD_RE = re.compile(r"model\.safetensors-[0-9]{5}-of-[0-9]{5}\.safetensors\Z")


class TextCheckpointProjectionError(ValueError):
    """An incompatible or unbound multimodal checkpoint cannot become text."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TextCheckpointProjectionError("duplicate checkpoint index key")
        result[key] = value
    return result


def _bad_constant(_value):
    raise TextCheckpointProjectionError("nonfinite checkpoint index value")


def index_binding_pin(base: dict) -> tuple[str, set[str]]:
    """Require a separately supplied digest for the index and every shard."""
    files = base.get("files") if type(base) is dict else None
    if type(files) is not dict:
        raise TextCheckpointProjectionError("checkpoint file binding absent")
    index_pin = files.get(INDEX_NAME)
    if (type(index_pin) is not str or
            re.fullmatch(r"[0-9a-f]{64}", index_pin) is None):
        raise TextCheckpointProjectionError("checkpoint index is not SHA-pinned")
    shards = {name for name in files if name.endswith(".safetensors")}
    if not shards or any(
        not SHARD_RE.fullmatch(name)
        or type(files[name]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", files[name]) is None
        for name in shards
    ):
        raise TextCheckpointProjectionError("checkpoint shard binding invalid")
    return index_pin, shards


def text_key_mapping(snapshot: Path, *, expected_index_sha256: str,
                     expected_shards: set[str]) -> dict[str, str]:
    """Bind indexed language tensors to the text-only state-dict namespace."""
    if (type(expected_index_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_index_sha256) is None
            or type(expected_shards) is not set or not expected_shards
            or any(type(name) is not str or not SHARD_RE.fullmatch(name)
                   for name in expected_shards)):
        raise TextCheckpointProjectionError("independent index/shard pins missing")
    index = Path(snapshot) / INDEX_NAME
    if index.is_symlink() or not index.is_file() or index.stat().st_size > MAX_INDEX_BYTES:
        raise TextCheckpointProjectionError("missing, symlinked or oversized checkpoint index")
    try:
        raw = index.read_bytes()
        if len(raw) > MAX_INDEX_BYTES:
            raise TextCheckpointProjectionError("checkpoint index exceeded bounded read")
        if hashlib.sha256(raw).hexdigest() != expected_index_sha256:
            raise TextCheckpointProjectionError("checkpoint index SHA differs from binding")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                              parse_constant=_bad_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise TextCheckpointProjectionError("invalid checkpoint index") from exc
    if (type(document) is not dict or set(document) != {"metadata", "weight_map"}
            or type(document["metadata"]) is not dict
            or type(document["metadata"].get("total_size")) is not int
            or not 0 < document["metadata"]["total_size"] <= 12 * 1024**3
            or type(document["weight_map"]) is not dict
            or not 1 <= len(document["weight_map"]) <= MAX_KEYS):
        raise TextCheckpointProjectionError("invalid checkpoint inventory")

    mapping = {}
    targets = set()
    referenced_shards = set()
    for name, shard in document["weight_map"].items():
        if (type(name) is not str or type(shard) is not str
                or not SHARD_RE.fullmatch(shard)):
            raise TextCheckpointProjectionError("invalid checkpoint tensor or shard")
        referenced_shards.add(shard)
        if name.startswith(TEXT_PREFIX):
            target = "model." + name[len(TEXT_PREFIX):]
            if not name[len(TEXT_PREFIX):] or target in targets:
                raise TextCheckpointProjectionError("ambiguous language-model projection")
            mapping[name] = target
            targets.add(target)
        elif not name.startswith(IGNORED_PREFIXES):
            raise TextCheckpointProjectionError("unexpected multimodal tensor namespace")
    if "model.language_model.embed_tokens.weight" not in mapping:
        raise TextCheckpointProjectionError("checkpoint text embeddings missing")
    if referenced_shards != expected_shards:
        raise TextCheckpointProjectionError("index shard set differs from bound checkpoint shards")
    return mapping


def prepare_text_loader(snapshot: Path, *, expected_index_sha256: str,
                        expected_shards: set[str]):
    """Build a meta-only language-model inventory before real tensor loading."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    full = AutoConfig.from_pretrained(snapshot, local_files_only=True,
                                      trust_remote_code=False)
    if (type(full).__name__ != "Qwen3_5Config"
            or full.model_type != "qwen3_5"
            or full.architectures != ["Qwen3_5ForConditionalGeneration"]
            or type(full.text_config).__name__ != "Qwen3_5TextConfig"
            or full.text_config.model_type != "qwen3_5_text"
            or full.text_config.tie_word_embeddings is not True):
        raise TextCheckpointProjectionError("official nested text config or tied head absent")
    mapping = text_key_mapping(snapshot, expected_index_sha256=expected_index_sha256,
                               expected_shards=expected_shards)
    with init_empty_weights():
        skeleton = Qwen3_5ForCausalLM(full.text_config)
    skeleton.tie_weights()
    if skeleton.get_output_embeddings().weight is not skeleton.get_input_embeddings().weight:
        raise TextCheckpointProjectionError("text output head is not tied to input embeddings")
    expected = set(skeleton.state_dict()) - {"lm_head.weight"}
    if set(mapping.values()) != expected:
        raise TextCheckpointProjectionError("indexed text weights do not cover model parameters")
    return full.text_config, mapping


def verify_text_load(model, loading_info: dict) -> None:
    """Reject silent missing, mismatched or unexpected language weights."""
    if type(loading_info) is not dict:
        raise TextCheckpointProjectionError("model load returned no loading information")
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = loading_info.get(field)
        if type(value) not in (list, set, tuple):
            raise TextCheckpointProjectionError(f"model load omitted {field} evidence")
    if (loading_info["missing_keys"] or loading_info["mismatched_keys"]
            or loading_info["error_msgs"]):
        raise TextCheckpointProjectionError("model load left missing/mismatched text weights")
    if any(type(name) is not str or not name.startswith(IGNORED_PREFIXES)
           for name in loading_info["unexpected_keys"]):
        raise TextCheckpointProjectionError("model load reported unexpected text/foreign weights")
    if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise TextCheckpointProjectionError("checkpoint output head is not tied to input embeddings")
