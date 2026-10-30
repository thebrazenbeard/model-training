"""Public/synthetic A→B→C retention lineage assessment, without model authority.

This module does not load adapters, score a model, open an evaluation bank,
or authenticate the submitted receipts.  It rejects contradictory declarations
and computes reproducible per-generation public-canary summaries only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from trainingday.admission import AdmissionError


SCHEMA = "VERA_TRAININGDAY_R25_MULTIGEN_RETENTION_V1"
PUBLIC_PARTITION = "PUBLIC_NONSEALED_RETENTION_ONLY"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_NAMES = ("A", "B", "C")
_MAX_CASES = 4096
_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class PublicBankManifest:
    """Answer-free public score-bank commitment and adapter starting identity."""

    source_sha256: str
    base_sha256: str
    initial_adapter_sha256: str
    case_ids: tuple[str, ...]
    partition: str = PUBLIC_PARTITION
    answers_used_for_training: bool = False


@dataclass(frozen=True)
class GenerationEvidence:
    """Caller-supplied declarations; no field is host- or scorer-attested here."""

    name: str
    previous_adapter_sha256: str
    adapter_sha256: str
    base_sha256: str
    public_case_sha256: str
    score_receipt_sha256: str
    score_vector_sha256: str
    training_receipt_sha256: str
    case_passes: tuple[bool, ...]
    time_budget_status: str = "VALID_DECLARED"
    private_bank_accessed: bool = False
    public_cases_used_for_training: bool = False
    zero_control_present: bool = False
    corrupt_control_present: bool = False
    independent_scorer_claimed: bool = False


@dataclass(frozen=True)
class RetentionPolicy:
    """Strict non-forgetting policy; statistical lower bound is descriptive."""

    max_case_regressions: int = 0
    minimum_wilson_lower_95: float = 0.5


def _digest(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def case_order_sha256(case_ids: tuple[str, ...]) -> str:
    """Canonical public case-ID/order commitment; contains no answers."""
    if (type(case_ids) is not tuple or not 2 <= len(case_ids) <= _MAX_CASES
            or any(type(cid) is not str or not _ID.fullmatch(cid) for cid in case_ids)
            or len(set(case_ids)) != len(case_ids)):
        raise AdmissionError("public retention case IDs invalid or duplicated")
    serialized = json.dumps(case_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def public_score_vector_sha256(case_order_digest: str,
                               case_passes: tuple[bool, ...]) -> str:
    """Self-consistency binding only, NOT authentic independent scoring proof."""
    if (not _digest(case_order_digest)
            or type(case_passes) is not tuple
            or not 2 <= len(case_passes) <= _MAX_CASES
            or any(type(value) is not bool for value in case_passes)):
        raise AdmissionError("invalid public score vector commitment input")
    payload = json.dumps({
        "case_order_sha256": case_order_digest,
        "case_passes": case_passes,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def wilson_lower_bound(passed: int, total: int) -> float:
    """95% Wilson score interval lower endpoint, not a neural-improvement test."""
    if (type(passed) is not int or type(total) is not int
            or not 1 <= total <= _MAX_CASES or not 0 <= passed <= total):
        raise AdmissionError("invalid public canary score count")
    p = passed / total
    z2 = _Z_95 * _Z_95
    denominator = 1 + z2 / total
    center = p + z2 / (2 * total)
    uncertainty = _Z_95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return max(0.0, (center - uncertainty) / denominator)


def _validate(manifest: PublicBankManifest,
              records: tuple[GenerationEvidence, ...],
              policy: RetentionPolicy) -> str:
    if (not isinstance(manifest, PublicBankManifest)
            or manifest.partition != PUBLIC_PARTITION
            or manifest.answers_used_for_training is not False
            or not all(_digest(v) for v in (
                manifest.source_sha256, manifest.base_sha256,
                manifest.initial_adapter_sha256))):
        raise AdmissionError("pinned public-only bank manifest required")
    case_hash = case_order_sha256(manifest.case_ids)
    try:
        valid_minimum = (type(policy.minimum_wilson_lower_95) in (int, float)
                         and math.isfinite(policy.minimum_wilson_lower_95)
                         and 0 <= policy.minimum_wilson_lower_95 <= 1)
    except (AttributeError, OverflowError, TypeError, ValueError):
        valid_minimum = False
    if (not isinstance(policy, RetentionPolicy)
            or type(policy.max_case_regressions) is not int
            or policy.max_case_regressions != 0
            or not valid_minimum):
        raise AdmissionError("strict public retention policy malformed")
    if (type(records) is not tuple or len(records) != len(_NAMES)):
        raise AdmissionError("exact A-B-C generation chain required")
    seen_adapters = {manifest.initial_adapter_sha256}
    previous = manifest.initial_adapter_sha256
    for record, name in zip(records, _NAMES):
        if not isinstance(record, GenerationEvidence) or record.name != name:
            raise AdmissionError("generation identity or order invalid")
        if (not all(_digest(v) for v in (
                record.previous_adapter_sha256, record.adapter_sha256,
                record.base_sha256, record.public_case_sha256,
                record.score_receipt_sha256, record.score_vector_sha256,
                record.training_receipt_sha256))
                or record.adapter_sha256 in seen_adapters
                or record.previous_adapter_sha256 != previous
                or record.base_sha256 != manifest.base_sha256
                or record.public_case_sha256 != case_hash):
            raise AdmissionError("adapter lineage or public-bank commitment mismatch")
        if (type(record.case_passes) is not tuple
                or len(record.case_passes) != len(manifest.case_ids)
                or any(type(value) is not bool for value in record.case_passes)):
            raise AdmissionError("public score row count or type invalid")
        if record.score_vector_sha256 != public_score_vector_sha256(
                case_hash, record.case_passes):
            raise AdmissionError("public score vector commitment mismatch")
        if (record.time_budget_status != "VALID_DECLARED"
                or record.private_bank_accessed is not False
                or record.public_cases_used_for_training is not False
                or any(type(v) is not bool for v in (
                    record.zero_control_present, record.corrupt_control_present,
                    record.independent_scorer_claimed))):
            raise AdmissionError("time budget, training leakage or control declaration invalid")
        seen_adapters.add(record.adapter_sha256)
        previous = record.adapter_sha256
    return case_hash


def assess_generation_retention(
    manifest: PublicBankManifest,
    records: tuple[GenerationEvidence, ...],
    policy: RetentionPolicy = RetentionPolicy(),
) -> dict:
    """Fail-closed declared lineage/retention assessment; always promotion HOLD.

    Count comparisons are *not* independent model scores. A/B/C may be three
    invented receipts; exact hashes and claimed reviewer presence do not prove
    real LoRA tensor changes, fresh-process reloading or task generalization.
    """
    case_hash = _validate(manifest, records, policy)
    summaries = []
    comparison = []
    for record in records:
        passed = sum(record.case_passes)
        summaries.append({
            "generation": record.name,
            "adapter_sha256": record.adapter_sha256,
            "declared_pass_count": passed,
            "wilson_lower_95": wilson_lower_bound(passed, len(manifest.case_ids)),
            "zero_and_corrupt_controls_claimed": (
                record.zero_control_present and record.corrupt_control_present),
            "independent_scorer_claimed": record.independent_scorer_claimed,
        })
    for earlier, later in ((0, 1), (1, 2), (0, 2)):
        before, after = records[earlier].case_passes, records[later].case_passes
        comparison.append({
            "transition": f"{records[earlier].name}->{records[later].name}",
            "declared_regressions": sum(a and not b for a, b in zip(before, after)),
            "declared_recoveries": sum(not a and b for a, b in zip(before, after)),
        })
    reasons = []
    if summaries[0]["declared_pass_count"] == 0:
        reasons.append("BASELINE_COMPETENCE_UNPROVEN")
    if any(entry["declared_regressions"] > policy.max_case_regressions
           for entry in comparison):
        reasons.append("DECLARED_CATASTROPHIC_FORGETTING_RISK")
    if any(entry["wilson_lower_95"] < policy.minimum_wilson_lower_95
           for entry in summaries):
        reasons.append("INSUFFICIENT_DECLARED_CONFIDENCE")
    if any(not entry["zero_and_corrupt_controls_claimed"] for entry in summaries):
        reasons.append("CONTROL_RECEIPTS_UNVERIFIED")
    if any(not entry["independent_scorer_claimed"] for entry in summaries):
        reasons.append("SCORER_INDEPENDENCE_UNVERIFIED")
    if not reasons:
        reasons.append("DECLARED_STABILITY_NOT_INDEPENDENTLY_ATTESTED")
    return {
        "schema": SCHEMA,
        "status": "HOLD",
        "hold_reasons": reasons,
        "source_sha256": manifest.source_sha256,
        "public_case_order_sha256": case_hash,
        "case_count": len(manifest.case_ids),
        "generation_summaries": summaries,
        "comparisons": comparison,
        "evidence_class": "CALLER_DECLARED_PUBLIC_SYNTHETIC_ONLY",
        "independent_model_scoring_verified": False,
        "actual_tensor_change_verified": False,
        "fresh_process_adapter_reload_verified": False,
        "private_eval_unsealed": False,
        "neural_improvement_proven": False,
        "promotion_allowed": False,
    }
