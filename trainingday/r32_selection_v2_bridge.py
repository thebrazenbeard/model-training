"""Public CPU bridge between production admission V2 and legacy R18 plans.

This module does not modify the historical R18 plan schema or either neural
runner. All evidence is supplied by one caller and remains untrusted; the
bridge merely exposes whether a separate declared V2 pin matches selected
source bytes while legacy consumers are left intact (and unqualified).
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from trainingday.admission import SCHEMA as ADMISSION_SCHEMA, SELECTED_ID_V2_SCHEMA
from trainingday.r18_benchmark import (
    BenchmarkError, _training_index, canonical_sha256, selected_id_digest,
    validate_plan,
)


class SelectionV2BridgeError(ValueError):
    """Malformed or unbounded public declaration, never a recovery bypass."""


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LEGACY_STATUS = "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE"
_MAX_ROWS = 1024
_MAX_NODES = 32768
_MAX_DEPTH = 16


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _plain_snapshot(value: object, *, depth: int = 0,
                    budget: list[int]) -> object:
    """Clone only exact builtin JSON-like values without invoking user hooks."""
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_DEPTH:
        raise SelectionV2BridgeError("declaration too deep or too large")
    cls = type(value)
    if cls is str:
        if len(value) > 4096:
            raise SelectionV2BridgeError("string declaration too large")
        return value
    if cls is int:
        if abs(value) > 2**63:
            raise SelectionV2BridgeError("integer declaration out of bounds")
        return value
    if cls is float:
        if not math.isfinite(value):
            raise SelectionV2BridgeError("nonfinite declaration")
        return value
    if cls in (bool, type(None)):
        return value
    if cls is list:
        if len(value) > 2048:
            raise SelectionV2BridgeError("list declaration too large")
        return [_plain_snapshot(item, depth=depth + 1, budget=budget)
                for item in value]
    if cls is tuple:
        if len(value) > 2048:
            raise SelectionV2BridgeError("tuple declaration too large")
        return tuple(_plain_snapshot(item, depth=depth + 1, budget=budget)
                     for item in value)
    if cls is dict:
        if len(value) > 2048:
            raise SelectionV2BridgeError("dict declaration too large")
        clone = {}
        for key, item in value.items():
            if type(key) is not str or len(key) > 256:
                raise SelectionV2BridgeError("non-builtin or oversized dict key")
            clone[key] = _plain_snapshot(item, depth=depth + 1, budget=budget)
        return clone
    raise SelectionV2BridgeError("non-builtin public declaration")


def _v2_selected_id_sha256(ids: list[str]) -> str:
    return hashlib.sha256(
        SELECTED_ID_V2_SCHEMA.encode("ascii") + b"\x00"
        + json.dumps(ids, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assess_v2_selection_bridge(*, selected_rows: object,
                               admission_receipt: object, frozen_plan: object,
                               frozen_v2_selection_sha256: object) -> dict:
    """Assess explicit source-only V2 selection consistency; authorize nothing.

    A future independently held plan needs to freeze the V2 digest *before*
    training. Merely supplying a matching pin here cannot authenticate custody
    and does not upgrade the old R18 MeZO/QLoRA/evaluator consumers.
    """
    if type(selected_rows) not in (list, tuple) or not 1 <= len(selected_rows) <= _MAX_ROWS:
        raise SelectionV2BridgeError("selected rows malformed or unbounded")
    if type(admission_receipt) is not dict or type(frozen_plan) is not dict:
        raise SelectionV2BridgeError("exact builtin public receipt and plan required")
    budget = [_MAX_NODES]
    rows = _plain_snapshot(selected_rows, budget=budget)
    summary = _plain_snapshot(admission_receipt, budget=budget)
    plan = _plain_snapshot(frozen_plan, budget=budget)
    if any(type(row) is not dict or type(row.get("id")) is not str
           or not row["id"].strip() or len(row["id"]) > 256 for row in rows):
        raise SelectionV2BridgeError("selected source IDs malformed")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SelectionV2BridgeError("duplicate selected source ID")
    try:
        plan_sha = validate_plan(plan)
        content_sha = canonical_sha256(_training_index(rows, plan["source_admission_status"]))
        legacy_sha = selected_id_digest(ids)
    except (BenchmarkError, TypeError, KeyError, ValueError) as exc:
        raise SelectionV2BridgeError("invalid legacy R18 source/plan declaration") from exc
    v2_sha = _v2_selected_id_sha256(ids)
    blockers: list[str] = []

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    if summary.get("schema") != ADMISSION_SCHEMA:
        block("PRODUCTION_ADMISSION_SCHEMA_MISMATCH")
    if summary.get("status") != "ADMITTED_SOURCE_ONLY_NOT_TRAINED":
        block("UNSUPPORTED_V2_ADMISSION_SOURCE")
    if type(summary.get("selected_count")) is not int or (
            summary["selected_count"] != len(ids)
            or plan["selected_count"] != len(ids)):
        block("SELECTED_COUNT_MISMATCH")
    if summary.get("manifest_sha256") != plan["corpus_manifest_sha256"]:
        block("MANIFEST_PLAN_MISMATCH")
    if (summary.get("selected_id_sha256") != legacy_sha
            or plan["selected_id_sha256"] != legacy_sha):
        block("LEGACY_SELECTED_IDS_MISMATCH")
    if plan["train_content_sha256"] != content_sha:
        block("TRAIN_CONTENT_PLAN_MISMATCH")
    if summary.get("selected_id_sha256_status") != _LEGACY_STATUS:
        block("LEGACY_AMBIGUITY_DISCLOSURE_MISSING")
    if ("selected_id_v2_schema" not in summary
            or "selected_id_v2_sha256" not in summary):
        block("V2_SELECTION_RECEIPT_MISSING")
    else:
        if summary["selected_id_v2_schema"] != SELECTED_ID_V2_SCHEMA:
            block("V2_SELECTION_SCHEMA_MISMATCH")
        if (not _sha(summary["selected_id_v2_sha256"])
                or summary["selected_id_v2_sha256"] != v2_sha):
            block("V2_RECEIPT_SELECTED_IDS_MISMATCH")
    if frozen_v2_selection_sha256 is None:
        block("FROZEN_V2_SELECTION_PIN_MISSING")
    elif not _sha(frozen_v2_selection_sha256):
        raise SelectionV2BridgeError("frozen V2 digest malformed")
    elif frozen_v2_selection_sha256 != v2_sha:
        block("FROZEN_V2_SELECTION_MISMATCH")
    return {
        "schema": "VERA_TRAININGDAY_R32_SELECTED_ID_BRIDGE_V1",
        "status": "HOLD", "blockers": blockers,
        "synthetic_v2_source_consistent": not blockers,
        "public_selected_id_v2_sha256": v2_sha,
        "declared_frozen_plan_sha256": plan_sha,
        "r18_runners_v2_migrated": False,
        "independent_custody_verified": False,
        "protected_eval_unseal_authorized": False,
        "gpu_start_authorized": False,
        "neural_training_authorized": False,
        "model_promotion_authorized": False,
    }
