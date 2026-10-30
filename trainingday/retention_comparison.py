"""Compare declared public canary judgments without upgrading them to model proof.

R20 source-only utility. All four arms must use the same pinned *nonsealed*
public cases and frozen base/initial adapter binding. This code never invokes
models, reads protected evaluations, or confers neural qualification.
"""

from __future__ import annotations

import hashlib
import re

from trainingday.admission import AdmissionError
from trainingday.retention_canary import (
    ARMS, IDS, RELATIVE, SCHEMA, SOURCE_SHA256, WINDOWS_CRLF_SHA256,
)


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_METHOD = "DECLARED_RUBRIC_JUDGMENTS_NOT_ATTESTED"
_PARTITION = "PUBLIC_NONSEALED_RETENTION_ONLY"
_FIELDS = frozenset({
    "arm", "source_sha256", "case_id_sha256", "trial_binding_sha256",
    "base_binding_sha256", "trainable_before_sha256", "trainable_after_sha256",
    "data_partition", "private_eval_access", "canary_used_for_training",
    "method", "rows",
})


def _pinned_digest(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def compare_retention_receipts(preflight: dict, receipts: list[dict]) -> dict:
    """Return paired declared canary changes, never a promotion decision.

    Per-case ``rubric_pass`` is externally supplied and *not* independently
    verified by this function. An aggregate pass-count gain can coexist with
    regressions on cases passed by the frozen arm.
    """
    expected_ids_sha = hashlib.sha256("\n".join(IDS).encode()).hexdigest()
    if (type(preflight) is not dict or preflight.get("schema") != SCHEMA
            or preflight.get("state") != "SOURCE_PREFLIGHT_ONLY_NOT_MODEL_SCORED"
            or preflight.get("source_sha256") != SOURCE_SHA256
            or preflight.get("checkout_sha256") not in (
                SOURCE_SHA256, WINDOWS_CRLF_SHA256)
            or preflight.get("checkout_line_endings") != (
                "LF" if preflight.get("checkout_sha256") == SOURCE_SHA256 else "CRLF")
            or preflight.get("source_relative_path") != RELATIVE.as_posix()
            or preflight.get("case_id_sha256") != expected_ids_sha
            or preflight.get("case_count") != len(IDS)
            or preflight.get("required_matched_arms") != list(ARMS)):
        raise AdmissionError("pinned public retention preflight required")
    if type(receipts) is not list or len(receipts) != len(ARMS):
        raise AdmissionError("all four matched public retention arms required")

    by_arm = {}
    bindings = set()
    initial_hashes = set()
    for entry in receipts:
        if type(entry) is not dict or set(entry) != _FIELDS:
            raise AdmissionError("retention receipt fields unrecognized or incomplete")
        arm = entry["arm"]
        if type(arm) is not str or arm not in ARMS or arm in by_arm:
            raise AdmissionError("duplicate or unknown public retention arm")
        if (entry["source_sha256"] != SOURCE_SHA256
                or entry["case_id_sha256"] != expected_ids_sha
                or entry["data_partition"] != _PARTITION
                or entry["private_eval_access"] is not False
                or entry["canary_used_for_training"] is not False
                or entry["method"] != _METHOD):
            raise AdmissionError("retention source or contamination boundary violated")
        if not all(_pinned_digest(entry[key]) for key in (
            "trial_binding_sha256", "base_binding_sha256",
            "trainable_before_sha256", "trainable_after_sha256",
        )):
            raise AdmissionError("retention model binding is not SHA-256 pinned")
        bindings.add((entry["trial_binding_sha256"], entry["base_binding_sha256"]))
        initial_hashes.add(entry["trainable_before_sha256"])
        if (arm in ("frozen", "zero")
                and entry["trainable_before_sha256"] != entry["trainable_after_sha256"]):
            raise AdmissionError("frozen/zero control claims a tensor update")
        if (arm in ("correct", "corrupt")
                and entry["trainable_before_sha256"] == entry["trainable_after_sha256"]):
            raise AdmissionError("trained control has no declared tensor update")
        rows = entry["rows"]
        if (type(rows) is not list or len(rows) != len(IDS)
                or any(type(row) is not dict or set(row) != {"id", "rubric_pass"}
                       or row.get("id") != case_id
                       or type(row.get("rubric_pass")) is not bool
                       for row, case_id in zip(rows, IDS))):
            raise AdmissionError("retention case IDs/order or rubric fields mismatched")
        by_arm[arm] = tuple(row["rubric_pass"] for row in rows)

    if set(by_arm) != set(ARMS) or len(bindings) != 1 or len(initial_hashes) != 1:
        raise AdmissionError("retention arms do not share pinned trial/base/initial state")
    frozen, zero, correct = (by_arm[name] for name in ("frozen", "zero", "correct"))
    regressions = sum(before and not after for before, after in zip(frozen, correct))
    recoveries = sum(not before and after for before, after in zip(frozen, correct))
    sham_disagreements = sum(a != b for a, b in zip(frozen, zero))
    if not any(frozen):
        status = "FROZEN_COMPETENCE_UNPROVEN"
    elif sham_disagreements:
        status = "ZERO_CONTROL_DISAGREEMENT"
    elif regressions:
        status = "DECLARED_RETENTION_REGRESSION"
    else:
        status = "NO_DECLARED_REGRESSION_UNVERIFIED"

    return {
        "schema": "VERA_TRAININGDAY_R20_PUBLIC_RETENTION_COMPARISON_V1",
        "status": status,
        "source_sha256": SOURCE_SHA256,
        "case_id_sha256": expected_ids_sha,
        "case_count": len(IDS),
        "matched_arms": list(ARMS),
        "pass_counts": {arm: sum(by_arm[arm]) for arm in ARMS},
        "correct_vs_frozen": {"regressions": regressions, "recoveries": recoveries},
        "frozen_zero_case_disagreements": sham_disagreements,
        "assessment_evidence": _METHOD,
        "model_improvement_proven": False,
        "promotion_allowed": False,
        "qualification": "PUBLIC_NONSEALED_DECLARATIONS_NOT_NEURAL_OR_INDEPENDENT_EVIDENCE",
    }
