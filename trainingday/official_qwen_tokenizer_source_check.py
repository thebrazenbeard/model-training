"""Offline source-only check of all public Calc rows with pinned Qwen3.5 BPE.

This is deliberately not a model loader, training launcher, authenticated
weights receipt, data-rights decision, or protected-evaluator admission.
Only the public tokenizer JSON and public Calc TRAIN source are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import stat
import sys

from trainingday import calc_curriculum, calc_tokenization
from trainingday.pinned_base import BASE_ID, BASE_REVISION


SCHEMA = "TRAININGDAY_OFFICIAL_QWEN_CALC_TOKENIZER_SOURCE_CHECK_V1"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZERS_VERSION = "0.22.1"
EOS_TOKEN = "<|im_end|>"
EOS_TOKEN_ID = 248046
MAX_TOKENIZER_BYTES = 32 * 1024 * 1024


class TokenizerSourceCheckError(RuntimeError):
    """A pinned source or the full public TRAIN tokenization failed."""


def _require(condition: bool, why: str) -> None:
    if not condition:
        raise TokenizerSourceCheckError(why)


def _read_verified_tokenizer(path: str | Path) -> bytes:
    """Hash the same regular-file bytes later parsed by tokenizers."""
    target = Path(path)
    _require(not target.is_symlink(), "tokenizer symlink rejected")
    try:
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(fd)
            _require(stat.S_ISREG(metadata.st_mode), "tokenizer is not regular file")
            _require(0 < metadata.st_size <= MAX_TOKENIZER_BYTES,
                     "tokenizer size outside bounded source check")
            data = bytearray()
            while len(data) <= MAX_TOKENIZER_BYTES:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                data.extend(block)
            _require(0 < len(data) <= MAX_TOKENIZER_BYTES,
                     "tokenizer source exceeds size limit")
        finally:
            os.close(fd)
    except OSError as exc:
        raise TokenizerSourceCheckError("tokenizer source not readable") from exc
    raw = bytes(data)
    _require(hashlib.sha256(raw).hexdigest() == TOKENIZER_SHA256,
             "official tokenizer SHA256 mismatch")
    return raw


class _RawTokenizerAdapter:
    """Minimal facade required by the no-Transformers Calc V2 compiler."""

    def __init__(self, raw):
        self.raw = raw
        self.eos_token_id = raw.token_to_id(EOS_TOKEN)
        _require(type(self.eos_token_id) is int
                 and self.eos_token_id == EOS_TOKEN_ID,
                 "official Qwen EOS mapping mismatch")

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return self.raw.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool,
               clean_up_tokenization_spaces: bool) -> str:
        _require(clean_up_tokenization_spaces is False,
                 "tokenizer decode cleanup must be disabled")
        return self.raw.decode(ids, skip_special_tokens=skip_special_tokens)


def _load_tokenizer(raw_bytes: bytes) -> _RawTokenizerAdapter:
    """Lazy dependency: importing this source-only checker never loads ML."""
    try:
        installed = version("tokenizers")
    except PackageNotFoundError as exc:
        raise TokenizerSourceCheckError("tokenizers==0.22.1 not installed") from exc
    _require(installed == TOKENIZERS_VERSION, "tokenizers package version mismatch")
    try:
        from tokenizers import Tokenizer

        raw = Tokenizer.from_str(raw_bytes.decode("utf-8"))
    except (ImportError, ValueError, RuntimeError, UnicodeError) as exc:
        raise TokenizerSourceCheckError("pinned tokenizer JSON could not be loaded") from exc
    return _RawTokenizerAdapter(raw)


def _audit_compiled(rows: list[dict], compiled: dict,
                    tokenizer: _RawTokenizerAdapter) -> dict:
    """Independently compare compiler outputs against canonical full encodes."""
    _require(type(compiled) is dict, "compiled result is not a dictionary")
    _require(compiled.get("schema") == calc_tokenization.SCHEMA
             and calc_tokenization.SCHEMA == "TRAININGDAY_CALC_TOKENIZATION_SOURCE_ONLY_V2",
             "Calc V2 compiler schema mismatch")
    _require(compiled.get("status") == "TOKENIZATION_SOURCE_ONLY_UNAUTHENTICATED_HOLD"
             and compiled.get("base_model") == BASE_ID
             and compiled.get("base_revision") == BASE_REVISION
             and compiled.get("train_count") == calc_curriculum.TRAIN_COUNT
             and compiled.get("train_sha256") == calc_tokenization.FROZEN_TRAIN_SHA256
             and compiled.get("max_sequence_length") == 64,
             "compiled source identity mismatch")
    for flag in ("tokenizer_authenticated", "private_dev_access", "authorizes_training"):
        _require(compiled.get(flag) is False, f"protected flag {flag} was elevated")
    _require(type(compiled.get("optimizer_steps_observed")) is int
             and compiled["optimizer_steps_observed"] == 0,
             "source check may not claim optimizer steps")
    examples = compiled.get("examples")
    _require(type(examples) is list and len(examples) == len(rows)
             and len(rows) == calc_curriculum.TRAIN_COUNT,
             "public TRAIN coverage not exactly 1,024")
    lengths = []
    negative_answers = 0
    for index, (row, example) in enumerate(zip(rows, examples)):
        _require(type(row) is dict and type(example) is dict
                 and example.get("id") == f"r201-calc-train-{index:04d}"
                 and row.get("id") == example["id"]
                 and row.get("split") == "train"
                 and set(example) == {"id", "input_ids", "labels", "attention_mask"},
                 "noncanonical public TRAIN row or example")
        prefix = f"Q: {row['prompt']}\nA:"
        completion = " " + row["answer"]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        canonical = tokenizer.encode(prefix + completion, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        expected_ids = canonical + [tokenizer.eos_token_id]
        expected_labels = [-100] * len(prefix_ids) + completion_ids + [tokenizer.eos_token_id]
        _require(canonical == prefix_ids + completion_ids
                 and example["input_ids"] == expected_ids
                 and example["labels"] == expected_labels
                 and example["attention_mask"] == [1] * len(expected_ids)
                 and len(expected_ids) <= 64
                 and any(label != -100 for label in expected_labels),
                 f"canonical tokenization/mask/EOS mismatch at TRAIN row {index}")
        lengths.append(len(expected_ids))
        if row["answer"].startswith("-"):
            negative_answers += 1
    _require(negative_answers > 0, "no negative-answer token boundary exercised")
    return {"max_observed_tokens": max(lengths),
            "min_observed_tokens": min(lengths),
            "negative_answer_rows": negative_answers}


def run_source_check(tokenizer_path: str | Path) -> dict:
    tokenizer = _load_tokenizer(_read_verified_tokenizer(tokenizer_path))
    rows = calc_curriculum.train_rows()
    manifest = calc_curriculum.source_commitment()
    compiled = calc_tokenization.compile_calc_train_tokens(
        rows, manifest, tokenizer, model_id=BASE_ID, model_revision=BASE_REVISION,
        max_sequence_length=64,
    )
    counts = _audit_compiled(rows, compiled, tokenizer)
    return {
        "schema": SCHEMA,
        "status": "REAL_TOKENIZER_SOURCE_ONLY_PASS",
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizers_version": TOKENIZERS_VERSION,
        "calc_revision": calc_curriculum.CALC_REVISION,
        "train_sha256": calc_tokenization.FROZEN_TRAIN_SHA256,
        "train_count": calc_curriculum.TRAIN_COUNT,
        "max_sequence_length": 64,
        "source_tokenizer_bytes_verified": True,
        "tokenizer_authenticated": False,
        "authorizes_training": False,
        "optimizer_steps_observed": 0,
        **counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path,
                        help="locally downloaded pinned public tokenizer.json")
    options = parser.parse_args(argv)
    try:
        report = run_source_check(options.tokenizer)
    except (TokenizerSourceCheckError, calc_curriculum.CalcCurriculumError,
            calc_tokenization.CalcTokenizationError) as exc:
        print(f"official-tokenizer source HOLD: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
