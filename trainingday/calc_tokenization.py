"""Source-only TRAIN tokenization for exact-integer Calc curriculum records.

This adapter does not import or load Transformers, Torch, Qwen weights, or any
protected DEV row. It converts a *caller-supplied* public TRAIN candidate into
answer-only causal labels, rejecting truncation, token/character mismatches,
model substitution, and positive authority claims. A matching SHA-256 manifest
is structural consistency, not authenticated Calc provenance or a TRAIN grant.
"""

from __future__ import annotations

import hashlib
import json
import re

from trainingday.pinned_base import BASE_ID, BASE_REVISION


SCHEMA = "TRAININGDAY_CALC_TOKENIZATION_SOURCE_ONLY_V1"
CALC_REPO = "thebrazenbeard/calc"
CALC_REVISION = "17e12d0f659b42732fb98e5730732aba9ff1b488"
CALC_MANIFEST_SCHEMA = "VERA_TRAININGDAY_R201_CALC_POLYNOMIAL_V1"
CALC_ORACLE_FAMILY = "CALC_SYMBOLIC_BOUNDED_INTEGER_POLYNOMIAL_DERIVATIVE_AT_POINT"
CALC_SYMBOLIC_SHA256 = "aba705f0d4370afa35538280f3db00ecbe36da1b3f3ca083ee29ebc292113b0b"
FROZEN_TRAIN_SHA256 = "fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b"
FROZEN_CORPUS_ID = "R201_NEW_CALC_POLYNOMIAL_TRAIN"
TRAIN_COUNT = 1024
MAX_SEQUENCE_LENGTH = 64
_ROW_ID = re.compile(r"r201-calc-train-[0-9]{4}\Z")
_EXACT_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_MANIFEST_KEYS = frozenset({
    "schema", "status", "calc_repository", "calc_revision",
    "calc_symbolic_sha256", "oracle_family", "corpus_id",
    "r185_corpus_replacement", "train_count", "train_sha256",
    "public_train_rows_returned", "private_dev_included", "private_dev_commitment",
    "rights_holder_authenticated", "data_rights_granted",
    "protected_evaluator_custody_verified", "gpu_lease_granted",
    "authorizes_training",
})


class CalcTokenizationError(ValueError):
    """Untrusted source, tokenizer or sequence violated a strict TRAIN seam."""


def _require(value: bool) -> None:
    if not value:
        raise CalcTokenizationError("Calc TRAIN source-only tokenization rejected")


def _train_hash(rows: list[dict]) -> str:
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )
    return hashlib.sha256(raw).hexdigest()


def _validate_frozen_manifest(manifest: dict) -> None:
    """Validate one frozen source-candidate identity, never caller authority."""
    _require(type(manifest) is dict and set(manifest) == _MANIFEST_KEYS)
    _require(type(manifest.get("schema")) is str
             and manifest["schema"] == CALC_MANIFEST_SCHEMA
             and type(manifest.get("status")) is str
             and manifest["status"] == "SOURCE_ONLY_UNAUTHENTICATED_HOLD"
             and all(type(manifest.get(name)) is str for name in
                     ("calc_repository", "calc_revision", "oracle_family",
                      "calc_symbolic_sha256", "train_sha256", "corpus_id"))
             and manifest["calc_repository"] == CALC_REPO
             and manifest["calc_revision"] == CALC_REVISION
             and manifest["oracle_family"] == CALC_ORACLE_FAMILY
             and manifest["calc_symbolic_sha256"] == CALC_SYMBOLIC_SHA256
             and manifest["train_sha256"] == FROZEN_TRAIN_SHA256
             and manifest["corpus_id"] == FROZEN_CORPUS_ID
             and type(manifest.get("train_count")) is int
             and manifest["train_count"] == TRAIN_COUNT)
    for flag in ("r185_corpus_replacement", "public_train_rows_returned",
                 "private_dev_included", "rights_holder_authenticated",
                 "data_rights_granted", "protected_evaluator_custody_verified",
                 "gpu_lease_granted", "authorizes_training"):
        _require(manifest.get(flag) is False)
    _require(manifest.get("private_dev_commitment") is None)


def _tokens(tokenizer, text: str) -> list[int]:
    try:
        result = tokenizer.encode(text, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CalcTokenizationError("tokenizer encode unavailable") from exc
    _require(type(result) is list and bool(result)
             and all(type(token) is int and 0 <= token < (1 << 31)
                     for token in result))
    return result


def _tokenize_one(row: dict, tokenizer, max_sequence_length: int) -> dict:
    _require(type(row) is dict and set(row) == {"id", "prompt", "answer", "split"})
    identifier, prompt, answer = row["id"], row["prompt"], row["answer"]
    _require(type(identifier) is str and _ROW_ID.fullmatch(identifier) is not None
             and type(row["split"]) is str and row["split"] == "train"
             and type(prompt) is str and 1 <= len(prompt) <= 180
             and all(char not in prompt for char in ("\r", "\n", "\x00"))
             and type(answer) is str and 1 <= len(answer) <= 32
             and _EXACT_INTEGER.fullmatch(answer) is not None)
    prefix = f"Q: {prompt}\nA: "
    prefix_ids = _tokens(tokenizer, prefix)
    answer_ids = _tokens(tokenizer, answer)
    eos = getattr(tokenizer, "eos_token_id", None)
    _require(type(eos) is int and 0 <= eos < (1 << 31)
             and eos not in prefix_ids and eos not in answer_ids)
    joined_ids = prefix_ids + answer_ids
    _require(len(joined_ids) + 1 <= max_sequence_length)
    # Separate encodes are intentional: a tokenizer must preserve the exact
    # prompt+answer bytes when token chunks are concatenated. Never silently
    # shift the answer-label boundary after a context-sensitive BPE merge.
    try:
        decoded = tokenizer.decode(
            joined_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CalcTokenizationError("tokenizer decode unavailable") from exc
    _require(type(decoded) is str and decoded == prefix + answer)
    input_ids = joined_ids + [eos]
    return {
        "id": identifier,
        "input_ids": input_ids,
        "labels": [-100] * len(prefix_ids) + answer_ids + [eos],
        "attention_mask": [1] * len(input_ids),
    }


def compile_calc_train_tokens(rows, manifest, tokenizer, *, model_id,
                              model_revision, max_sequence_length=64):
    """Compile a pinned public 1,024-row candidate; authorize nothing.

    `tokenizer` is supplied by a caller. Exact Qwen tokenizer bytes and their
    host custody must be independently verified before a protected process
    consumes this output. The returned token IDs are not training evidence.
    """
    _require(type(rows) is list and len(rows) == TRAIN_COUNT
             and type(max_sequence_length) is int
             and 8 <= max_sequence_length <= MAX_SEQUENCE_LENGTH)
    _require(type(model_id) is str and model_id == BASE_ID
             and type(model_revision) is str and model_revision == BASE_REVISION)
    _validate_frozen_manifest(manifest)
    # Validate every row before hashing (including 1,024 unique identifiers).
    _require(all(type(row) is dict and set(row) ==
                 {"id", "prompt", "answer", "split"} for row in rows))
    identifiers = [row["id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    _require(all(type(value) is str for value in identifiers + prompts)
             and len(set(identifiers)) == TRAIN_COUNT
             and len(set(prompts)) == TRAIN_COUNT)
    digest = _train_hash(rows)
    _require(digest == FROZEN_TRAIN_SHA256)
    examples = [_tokenize_one(row, tokenizer, max_sequence_length) for row in rows]
    return {
        "schema": SCHEMA,
        "status": "TOKENIZATION_SOURCE_ONLY_UNAUTHENTICATED_HOLD",
        "calc_repository": CALC_REPO,
        "calc_revision": CALC_REVISION,
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "train_count": TRAIN_COUNT,
        "train_sha256": digest,
        "max_sequence_length": max_sequence_length,
        "examples": examples,
        "tokenizer_authenticated": False,
        "private_dev_access": False,
        "authorizes_training": False,
        "optimizer_steps_observed": 0,
    }
