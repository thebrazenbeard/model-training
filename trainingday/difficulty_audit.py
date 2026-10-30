"""Aggregate-only R22 audit of *already admitted* integer-chain training rows.

This does not select data, load a model/tokenizer, or access any heldout bank.
The caller-supplied token lengths are declarations, not a tokenizer attestation.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from trainingday.r18_benchmark import (
    BenchmarkError, _training_index, canonical_sha256, selected_id_digest,
)


class AuditError(ValueError):
    """Training-only source or aggregate evidence is not admissible."""


_STATUS = "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE"
_PROMPT = re.compile(
    r"Start with (-?(?:0|[1-9][0-9]*))\. "
    r"((?:(?:add|subtract|multiply) -?(?:0|[1-9][0-9]*)\. ){2,5})"
    r"What is the final integer\?"
)
_OP = re.compile(r"(add|subtract|multiply) (-?(?:0|[1-9][0-9]*))\. ")
_ROW_KEYS = frozenset({"id", "family", "prompt", "answer"})


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _chain(row: Mapping) -> tuple[str, ...]:
    match = _PROMPT.fullmatch(row["prompt"])
    _check(match is not None, "training question is not a canonical integer chain")
    value = int(match.group(1))
    commands = _OP.findall(match.group(2))
    _check(2 <= len(commands) <= 5, "operation count outside admitted train corridor")
    names = []
    for name, operand_text in commands:
        operand = int(operand_text)
        _check(-15 <= operand <= 15, "operation operand outside admitted corridor")
        if name == "add":
            value += operand
        elif name == "subtract":
            value -= operand
        else:
            value *= operand
        names.append(name)
    answer = str(value)
    _check(row["answer"] == answer, "training answer differs from arithmetic replay")
    source = f"Question: {row['prompt']}\nAnswer: {answer}".encode("utf-8")
    _check(row["id"] == "reasoning-" + hashlib.sha256(source).hexdigest(),
           "training case ID differs from replayed original content")
    return tuple(names)


def audit_admitted_training(
    rows: Sequence[Mapping], *, selected_id_sha256: str,
    source_manifest_sha256: str, tokenizer_binding_sha256: str,
    token_lengths: Mapping[str, int], expected_count: int = 1024,
    source_admission_status: str = _STATUS,
) -> dict:
    """Return no raw questions, answers, IDs or individual tokenizer lengths.

    `token_lengths` must be independently measured with the pinned tokenizer by
    the caller; this function checks their shape and hashes the declarations but
    cannot establish that the stated tokenizer was actually used.
    """
    _check(type(expected_count) is int and 1 <= expected_count <= 1024,
           "invalid expected selected training count")
    _check(isinstance(rows, (list, tuple)) and len(rows) == expected_count,
           "selected training count mismatch")
    _check(source_admission_status == _STATUS,
           "only replay-admitted Labyrinth training source is auditable")
    _check(all(_hex64(x) for x in (
        selected_id_sha256, source_manifest_sha256, tokenizer_binding_sha256)),
        "source/tokenizer/selection binding absent")
    _check(isinstance(token_lengths, Mapping), "token length observations absent")
    for row in rows:
        _check(isinstance(row, Mapping) and set(row) == _ROW_KEYS
               and row.get("family") == "verified_integer_chains"
               and all(type(row.get(key)) is str for key in _ROW_KEYS),
               "noncanonical or contaminated training record")
    try:
        _check(selected_id_digest(rows) == selected_id_sha256,
               "ordered selected-ID hash mismatch")
        training_index = _training_index(rows, source_admission_status)
        ids = {row["id"] for row in rows}
        _check(set(token_lengths) == ids, "token length coverage mismatch")
        _check(all(type(value) is int and 1 <= value <= 4096
                   for value in token_lengths.values()),
               "invalid tokenizer target length declaration")
        pattern_counts: Counter[str] = Counter()
        op_counts: Counter[str] = Counter()
        token_counts: Counter[str] = Counter()
        for row in rows:
            names = _chain(row)
            pattern_counts[">".join(names)] += 1
            op_counts[str(len(names))] += 1
            token_counts[str(token_lengths[row["id"]])] += 1
        summary = {
            "schema": "VERA_TRAININGDAY_R22_AGGREGATE_ONLY_V1",
            "status": "PUBLIC_TRAIN_SOURCE_AGGREGATED_NO_WEIGHT_UPDATE",
            "sample_size": len(rows),
            "selected_id_sha256": selected_id_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "tokenizer_binding_sha256": tokenizer_binding_sha256,
            "train_content_sha256": canonical_sha256(training_index),
            "token_length_declaration_sha256": canonical_sha256(
                {key: token_lengths[key] for key in sorted(ids)}),
            "operation_patterns": dict(sorted(pattern_counts.items())),
            "operation_count_distribution": dict(sorted(op_counts.items())),
            "target_token_length_distribution": dict(sorted(token_counts.items())),
            "structural_duplicate_count": sum(n - 1 for n in pattern_counts.values()),
            "exact_duplicate_count": 0,
            "tokenizer_physically_verified": False,
            "neural_improvement_proven": False,
            "qualified_for_neural_training": False,
            "private_evaluation_accessed": False,
        }
        return {**summary, "aggregate_sha256": canonical_sha256(summary)}
    except (BenchmarkError, UnicodeError, TypeError, KeyError, ValueError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError("admitted training aggregate failed structural verification") from exc
