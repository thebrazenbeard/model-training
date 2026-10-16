"""Deterministic development-only scoring of machine-typed semantic transfers.

This scorer consumes model-produced structured outputs. Its frozen, public
development vectors are NOT independent held-out evidence, qualification, or
proof of cross-model latent transfer. No inference/network side effects here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Mapping


SCHEMA = "VERA_SEMANTIC_TRANSFER_DEVSET_V1"
PREDICTION_SCHEMA = "VERA_SEMANTIC_TRANSFER_PREDICTIONS_V1"
REPORT_SCHEMA = "VERA_SEMANTIC_TRANSFER_DEV_REPORT_V1"
CATEGORIES = ("referent", "correction", "scope", "authority", "recovery")
MAX_INPUT_BYTES = 128_000


class TransferEvidenceError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    def check(v: Any, depth: int = 0):
        if depth > 15:
            raise TransferEvidenceError("Maximum semantic nesting exceeded")
        if v is None or type(v) in (bool, str):
            return
        if type(v) is int and -9007199254740991 <= v <= 9007199254740991:
            return
        if type(v) is list and len(v) <= 256:
            for x in v:
                check(x, depth + 1)
            return
        if type(v) is dict and len(v) <= 256:
            if any(type(k) is not str or not k.isascii() or not k.isprintable() for k in v):
                raise TransferEvidenceError("Invalid canonical key")
            for x in v.values():
                check(x, depth + 1)
            return
        raise TransferEvidenceError("Invalid canonical semantic value")
    check(value)
    raw = json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,
                     allow_nan=False).encode("utf8")
    if len(raw) > MAX_INPUT_BYTES:
        raise TransferEvidenceError("Canonical payload too large")
    return raw


def digest(value: Any) -> str:
    return sha256(canonical(value)).hexdigest()


def load_cases(path: str | Path) -> tuple[dict, ...]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise TransferEvidenceError("Case file missing or oversized")
    try:
        raw = json.loads(path.read_text(encoding="utf8"))
    except (UnicodeError, ValueError) as exc:
        raise TransferEvidenceError("Case set is not valid JSON") from exc
    if type(raw) is not dict or set(raw) != {"schema", "cases"} or raw["schema"] != SCHEMA:
        raise TransferEvidenceError("Unknown development set schema")
    cases = raw["cases"]
    if type(cases) is not list or len(cases) < 25 or len(cases) > 256:
        raise TransferEvidenceError("Development set must contain at least 25 bounded cases")
    ids = set()
    category_counts = {k:0 for k in CATEGORIES}
    for c in cases:
        if type(c) is not dict or set(c) != {"id", "category", "state", "query", "expected"}:
            raise TransferEvidenceError("Malformed test case")
        if (type(c["id"]) is not str or not c["id"] or len(c["id"]) > 80
                or not c["id"].isascii() or c["id"] in ids):
            raise TransferEvidenceError("Case ID collision or invalid ID")
        ids.add(c["id"])
        if c["category"] not in CATEGORIES:
            raise TransferEvidenceError("Unrecognized semantic category")
        category_counts[c["category"]]+=1
        if type(c["state"]) is not dict or type(c["query"]) is not dict:
            raise TransferEvidenceError("Machine input must be typed object")
        if (type(c["expected"]) is not dict
                or set(c["expected"]) != {"decision", "value"}
                or type(c["expected"]["decision"]) is not str):
            raise TransferEvidenceError("Invalid expected response shape")
        canonical(c)
    if any(n < 5 for n in category_counts.values()):
        raise TransferEvidenceError("At least five cases per semantic category required")
    canonical(raw)
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class DevelopmentScore:
    dataset_sha256: str
    prediction_sha256: str
    scores: tuple[tuple[str,int,int], ...]
    failed_ids: tuple[str, ...]
    run_id: str
    submission_kind: str

    def to_mapping(self):
        return {
            "schema":REPORT_SCHEMA,
            "dataset_sha256":self.dataset_sha256,
            "prediction_sha256":self.prediction_sha256,
            "scores":[{"category":k,"passed":passed,"total":total}
                      for k,passed,total in self.scores],
            "failed_ids":list(self.failed_ids),
            "run_id":self.run_id,
            "submission_kind":self.submission_kind,
            "qualification":"UNQUALIFIED_DEVELOPMENT_DIAGNOSTIC",
        }


def score_predictions(
    cases: tuple[dict, ...],
    predictions: Mapping[str, Any],
    *,
    run_id: str,
    submission_kind: str = "EXTERNAL_CANDIDATE",
) -> DevelopmentScore:
    """Missing, non-object, malformed, or mismatching responses are failures.

    predictions must be a mapping of exact case IDs to candidate values. This
    function never substitutes an expected answer or treats a self-declared
    confidence, score, or success field as evidence.
    """
    if type(run_id) is not str or not run_id or len(run_id) > 120:
        raise TransferEvidenceError("Run must have a bounded ID")
    if submission_kind not in ("EXTERNAL_CANDIDATE","FIXTURE_POSITIVE",
                               "FIXTURE_NEGATIVE","LOCAL_MODEL_DEVELOPMENT"):
        raise TransferEvidenceError("Unknown submission kind")
    if type(predictions) is not dict or len(predictions) > 256:
        raise TransferEvidenceError("Invalid predictions object")
    allowed = {c["id"] for c in cases}
    if set(predictions) - allowed:
        raise TransferEvidenceError("Unknown predictions cannot be quietly ignored")
    canonical(predictions)
    counts = {c:[0,0] for c in CATEGORIES}
    failed=[]
    for case in cases:
        category=case["category"]
        counts[category][1]+=1
        prediction=predictions.get(case["id"])
        if type(prediction) is dict and set(prediction)=={"decision","value"} and (
            canonical(prediction)==canonical(case["expected"])
        ):
            counts[category][0]+=1
        else:
            failed.append(case["id"])
    return DevelopmentScore(
        dataset_sha256=digest({"schema":SCHEMA, "cases":list(cases)}),
        prediction_sha256=digest(predictions),
        scores=tuple((k,*counts[k]) for k in CATEGORIES),
        failed_ids=tuple(failed),
        run_id=run_id,
        submission_kind=submission_kind,
    )


def load_predictions(path: str | Path) -> dict:
    path=Path(path)
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise TransferEvidenceError("Prediction file absent or oversized")
    data=json.loads(path.read_text(encoding="utf8"))
    if (type(data) is not dict or set(data)!={"schema","predictions"} or
        data["schema"]!=PREDICTION_SCHEMA):
        raise TransferEvidenceError("Incorrect prediction submission schema")
    return data["predictions"]
