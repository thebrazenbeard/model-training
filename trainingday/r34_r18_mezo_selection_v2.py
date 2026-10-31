"""Source-only frozen V2 selected-ID binding for the real R18 MeZO preflight.

The historical R18 plan V1 and arm receipts retain their legacy digest for
readback. A NEW MeZO run needs a separately frozen, byte-pinned V2 sidecar.
This module cannot authenticate who froze that sidecar or authorize CUDA.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from trainingday.admission import SCHEMA as PRODUCTION_ADMISSION_SCHEMA, SELECTED_ID_V2_SCHEMA


BINDING_SCHEMA = "VERA_TRAININGDAY_R34_R18_MEZO_SELECTED_V2_FROZEN_BINDING_V1"
LEGACY_STATUS = "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_STATUSES = frozenset({
    "ADMITTED_SOURCE_ONLY_NOT_TRAINED",
    "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
})
_MAX_ROWS = 1024
_MAX_BYTES = 4096
_KEYS = frozenset({
    "schema", "plan_sha256", "corpus_manifest_sha256", "selected_count",
    "source_admission_status", "selected_id_v2_schema", "selected_id_v2_sha256",
    "selected_id_sha256", "selected_id_sha256_status",
})


class FrozenSelectionV2Error(ValueError):
    """A declared source selection differs from a frozen V2 commitment."""


def _check(value: bool, message: str) -> None:
    if not value:
        raise FrozenSelectionV2Error(message)


def _sha(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _snapshot_source(selected_rows: object, summary: object) -> tuple[tuple[dict, ...], dict]:
    """Detach only the primitive fields relevant to this source-only V2 pin.

    Exact builtin containers and string keys are checked before any lookup so
    attacker-defined Python hash/equality methods cannot stand in for JSON.
    The copies are not an atomic host-owned or authenticated source snapshot.
    """
    _check(type(summary) is dict and all(type(key) is str for key in summary),
           "source admission summary must use exact JSON string keys")
    _check(type(selected_rows) in (list, tuple)
           and 1 <= len(selected_rows) <= _MAX_ROWS,
           "selected source rows absent or unbounded")
    fields = ("id", "evaluation_holdout", "split", "training_use")
    detached: list[dict] = []
    for row in selected_rows:
        _check(type(row) is dict and all(type(key) is str for key in row),
               "selected source rows must use exact JSON string keys")
        detached.append({field: row[field] for field in fields if field in row})
    summary_fields = (
        "status", "selected_count", "manifest_sha256", "selected_id_sha256",
        "schema", "selected_id_v2_schema", "selected_id_v2_sha256",
        "selected_id_sha256_status",
    )
    return tuple(detached), {
        field: summary[field] for field in summary_fields if field in summary
    }


def _selected_ids(selected_rows: object, summary: object) -> list[str]:
    _check(type(summary) is dict, "source admission summary must be a JSON object")
    _check(type(selected_rows) in (list, tuple)
           and 1 <= len(selected_rows) <= _MAX_ROWS,
           "selected source rows absent or unbounded")
    _check(type(summary.get("status")) is str
           and summary["status"] in _SOURCE_STATUSES,
           "source admission status unsupported")
    _check(type(summary.get("selected_count")) is int
           and summary["selected_count"] == len(selected_rows),
           "selected source count differs from admission")
    _check(_sha(summary.get("manifest_sha256"))
           and _sha(summary.get("selected_id_sha256")),
           "source admission manifest/legacy digest missing")
    ids: list[str] = []
    for row in selected_rows:
        _check(type(row) is dict and type(row.get("id")) is str
               and 0 < len(row["id"]) <= 256 and row["id"].strip(),
               "selected source ID malformed")
        if summary["status"] == "ADMITTED_SOURCE_ONLY_NOT_TRAINED":
            _check(row.get("evaluation_holdout") is False,
                   "production evaluation holdout must be explicitly false")
            _check(type(row.get("split")) is str
                   and row["split"] == "train"
                   and type(row.get("training_use")) is str
                   and row["training_use"] == "ADMITTED",
                   "only admitted train split may enter MeZO")
        else:
            _check("evaluation_holdout" not in row,
                   "Labyrinth replay cannot carry a production holdout marker")
            _check("split" not in row and "training_use" not in row,
                   "Labyrinth selected source must not carry a split override")
        ids.append(row["id"])
    _check(len(set(ids)) == len(ids), "duplicate selected training IDs")
    legacy = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    _check(legacy == summary["selected_id_sha256"],
           "selected IDs disagree with legacy admission checksum")
    if summary["status"] == "ADMITTED_SOURCE_ONLY_NOT_TRAINED":
        _check(type(summary.get("schema")) is str
               and summary["schema"] == PRODUCTION_ADMISSION_SCHEMA
               and type(summary.get("selected_id_v2_schema")) is str
               and summary["selected_id_v2_schema"] == SELECTED_ID_V2_SCHEMA
               and _sha(summary.get("selected_id_v2_sha256"))
               and summary["selected_id_v2_sha256"] == _v2_sha256(ids)
               and type(summary.get("selected_id_sha256_status")) is str
               and summary["selected_id_sha256_status"] == LEGACY_STATUS,
               "production V2 admission receipt disagrees with selected IDs")
    return ids


def _v2_sha256(ids: list[str]) -> str:
    return hashlib.sha256(
        SELECTED_ID_V2_SCHEMA.encode("ascii") + b"\x00"
        + json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def make_frozen_selection_v2(*, plan_sha256: str, summary: object,
                             selected_rows: object) -> dict:
    """Create declared source evidence, to be separately frozen by custodian.

    It does NOT itself make the declaration immutable, authenticated, or
    sufficient to start CUDA training.
    """
    _check(_sha(plan_sha256), "frozen plan SHA256 absent")
    rows_copy, summary_copy = _snapshot_source(selected_rows, summary)
    ids = _selected_ids(rows_copy, summary_copy)
    return {
        "schema": BINDING_SCHEMA,
        "plan_sha256": plan_sha256,
        "corpus_manifest_sha256": summary_copy["manifest_sha256"],
        "selected_count": len(ids),
        "source_admission_status": summary_copy["status"],
        "selected_id_v2_schema": SELECTED_ID_V2_SCHEMA,
        "selected_id_v2_sha256": _v2_sha256(ids),
        "selected_id_sha256": summary_copy["selected_id_sha256"],
        "selected_id_sha256_status": LEGACY_STATUS,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise FrozenSelectionV2Error("duplicate frozen V2 JSON key")
        result[key] = value
    return result


def verify_frozen_selection_v2(*, binding_path: Path | None, plan_sha256: str,
                               summary: object, selected_rows: object) -> dict:
    """Gate one real MeZO source preflight on a separately pinned V2 sidecar."""
    _check(isinstance(binding_path, Path) and binding_path.is_file()
           and not binding_path.is_symlink(), "frozen V2 binding missing/unsafe")
    with binding_path.open("rb") as stream:
        content = stream.read(_MAX_BYTES + 1)
    _check(0 < len(content) <= _MAX_BYTES, "frozen V2 binding file unbounded")
    try:
        pin = json.loads(content.decode("utf-8"),
                         object_pairs_hook=_reject_duplicate_keys,
                         parse_constant=lambda unused: (_ for _ in ()).throw(
                             FrozenSelectionV2Error("nonfinite frozen V2 JSON value")))
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise FrozenSelectionV2Error("malformed frozen V2 JSON") from exc
    _check(type(pin) is dict and set(pin) == _KEYS,
           "frozen V2 binding schema/keys missing")
    _check(pin.get("schema") == BINDING_SCHEMA
           and pin.get("selected_id_v2_schema") == SELECTED_ID_V2_SCHEMA
           and pin.get("selected_id_sha256_status") == LEGACY_STATUS,
           "frozen V2 binding contract changed")
    _check(_sha(pin.get("selected_id_v2_sha256")) and _sha(pin.get("plan_sha256")),
           "frozen V2 selected IDs or plan SHA malformed")
    _check(type(pin.get("selected_count")) is int
           and 1 <= pin["selected_count"] <= _MAX_ROWS
           and _sha(pin.get("corpus_manifest_sha256"))
           and _sha(pin.get("selected_id_sha256"))
           and type(pin.get("source_admission_status")) is str,
           "frozen V2 source fields malformed")
    expected = make_frozen_selection_v2(
        plan_sha256=plan_sha256, summary=summary, selected_rows=selected_rows)
    _check(pin["plan_sha256"] == expected["plan_sha256"],
           "frozen V2 plan differs from validated preregistration")
    _check(pin["selected_id_v2_sha256"] == expected["selected_id_v2_sha256"],
           "frozen V2 selected IDs differ from admitted training rows")
    _check(pin == expected, "frozen V2 source/manifest/count/legacy pin mismatch")
    return {
        "status": "SOURCE_ONLY_V2_SELECTION_MATCH",
        "selected_id_v2_schema": SELECTED_ID_V2_SCHEMA,
        "selected_id_v2_sha256": expected["selected_id_v2_sha256"],
        "frozen_v2_binding_file_sha256": hashlib.sha256(content).hexdigest(),
        "independent_custody_verified": False,
        "neural_training_authorized": False,
    }
