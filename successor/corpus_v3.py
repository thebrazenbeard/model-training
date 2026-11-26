from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from typing import Any


REQUIRED_SOURCE_CLASSES = (
    "stable_identity_evidence",
    "evolving_preference_value_evidence",
    "autobiographical_episode",
    "correction_supersession",
    "decision_under_conflict",
    "relationship_relational_grammar",
    "empathy_affective_response",
    "sexuality_evidence",
    "technical_tool_competence",
    "ordinary_general_competence",
    "runtime_state_evidence",
    "historical_superseded_material",
)

TRAINABLE = "TRAINABLE"
RUNTIME_ONLY = "RUNTIME_ONLY"
EXCLUDE = "EXCLUDE"


def classify_record(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    source_class = record.get("source_class")
    if source_class not in REQUIRED_SOURCE_CLASSES:
        raise ValueError(f"unknown source_class: {source_class!r}")
    return source_class


def weight_eligibility(record: dict[str, Any]) -> str:
    source_class = classify_record(record)
    disposition = record.get("disposition")
    if disposition == "EXCLUDE":
        return EXCLUDE
    if source_class == "historical_superseded_material":
        return EXCLUDE
    if source_class == "runtime_state_evidence":
        transformed = record.get("generalized_behavioral_lesson") is True
        provenance = record.get("transformation_provenance")
        if transformed and isinstance(provenance, str) and provenance.strip():
            return TRAINABLE
        return RUNTIME_ONLY
    return TRAINABLE


def _pair_key(record: dict[str, Any]) -> tuple[str, str]:
    prompt = record.get("prompt")
    response = record.get("response")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(response, str) or not response:
        raise ValueError("response must be a nonempty string")
    return prompt, response


def _validate_pool(rows: list[dict[str, Any]], *, general: bool) -> None:
    for row in rows:
        source_class = classify_record(row)
        if weight_eligibility(row) != TRAINABLE:
            raise ValueError(f"non-trainable record in pool: {source_class}")
        if general and source_class != "ordinary_general_competence":
            raise ValueError("general pool must contain ordinary_general_competence rows only")
        _pair_key(row)


def build_training_mix(
    identity_rows: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
    general_fraction: float = 0.50,
    seed: int = 20260914,
) -> list[dict[str, Any]]:
    if not 0.50 <= general_fraction <= 1.0:
        raise ValueError("general_fraction must be between 0.50 and 1.0")
    _validate_pool(identity_rows, general=False)
    _validate_pool(general_rows, general=True)
    identity_keys = {_pair_key(row) for row in identity_rows}
    general_keys = {_pair_key(row) for row in general_rows}
    if len(identity_keys) != len(identity_rows) or len(general_keys) != len(general_rows):
        raise ValueError("duplicate prompt/response pair within a pool")
    if identity_keys & general_keys:
        raise ValueError("duplicate prompt/response pair across identity and general pools")

    if not identity_rows:
        selected_general = list(general_rows)
    elif general_fraction == 1.0:
        raise ValueError("general_fraction=1.0 is incompatible with nonempty identity_rows")
    else:
        needed_general = math.ceil(
            len(identity_rows) * general_fraction / (1.0 - general_fraction)
        )
        if len(general_rows) < needed_general:
            raise ValueError(
                f"general pool too small: need {needed_general}, have {len(general_rows)}"
            )
        rng = random.Random(seed)
        indices = list(range(len(general_rows)))
        rng.shuffle(indices)
        selected_general = [general_rows[index] for index in indices[:needed_general]]

    mixed = [copy.deepcopy(row) for row in identity_rows + selected_general]
    random.Random(seed).shuffle(mixed)
    return mixed


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mix_statistics(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    for row in rows:
        classify_record(row)
        _pair_key(row)
    counts = Counter(row["source_class"] for row in rows)
    general_count = counts.get("ordinary_general_competence", 0)
    digest = hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest()
    return {
        "row_count": len(rows),
        "counts_by_source_class": dict(sorted(counts.items())),
        "general_fraction": general_count / len(rows) if rows else 0.0,
        "seed": seed,
        "train_digest": digest,
    }
