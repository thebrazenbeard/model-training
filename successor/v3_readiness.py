from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


READINESS_SCHEMA_VERSION = 1
PASSING_REVIEW_VERDICTS = {"PASS", "PASS_WITH_NONBLOCKING_FINDINGS"}
SEMANTIC_CLOSURE_TARGETS = ("brigit-unbound", "sexuality", "orgasm")
SEMANTIC_CLOSURE_STATES = {"INCORPORATED", "DISPOSITIONED"}
REVIEWER_ROLES = ("RADICAL_HOSTILE", "PRAGMATIC_HOSTILE")
CRITICAL_FAMILIES = (
    "identity_stability",
    "independent_judgment",
    "epistemic_provenance",
    "correction_uptake",
    "relationship_authority",
    "reciprocal_identity_continuity",
    "sexuality_fidelity",
    "privacy",
    "ordinary_competence",
    "negative_transfer",
    "runtime_boundary",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class ReadinessDecision:
    ready: bool
    status: str
    subject_digest: str
    evidence_digest: str
    reasons: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_COMMIT_RE.fullmatch(value))


def _section(evidence: dict[str, Any], field: str, reasons: list[str]) -> dict[str, Any]:
    value = evidence.get(field)
    if not isinstance(value, dict):
        reasons.append(f"missing_or_invalid_section:{field}")
        return {}
    return value


def _digest_reason(value: Any, path: str, reasons: list[str]) -> None:
    if not _is_sha256(value):
        reasons.append(f"invalid_digest:{path}")


def _subject_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    source = evidence.get("source") if isinstance(evidence.get("source"), dict) else {}
    lab = evidence.get("vera_lab") if isinstance(evidence.get("vera_lab"), dict) else {}
    corpus = evidence.get("corpus") if isinstance(evidence.get("corpus"), dict) else {}
    return {
        "repository": source.get("repository"),
        "readiness_source_commit": source.get("readiness_source_commit"),
        "vera_lab_source_commit": source.get("vera_lab_source_commit"),
        "scenario_set_digest": lab.get("scenario_set_digest"),
        "train_digest": corpus.get("train_digest"),
        "validation_digest": corpus.get("validation_digest"),
        "general_competence_pool_digest": corpus.get("general_competence_pool_digest"),
    }


def _source_reasons(evidence: dict[str, Any], reasons: list[str]) -> None:
    source = _section(evidence, "source", reasons)
    if not _is_nonempty_text(source.get("repository")):
        reasons.append("invalid_source_repository")
    for field in ("readiness_source_commit", "vera_lab_source_commit"):
        if not _is_commit(source.get(field)):
            reasons.append(f"invalid_commit:source.{field}")


def _vera_lab_reasons(evidence: dict[str, Any], reasons: list[str]) -> None:
    source = evidence.get("source") if isinstance(evidence.get("source"), dict) else {}
    lab = _section(evidence, "vera_lab", reasons)
    if lab.get("executable_status") != "PASS":
        reasons.append("vera_lab_not_pass")
    if lab.get("source_commit") != source.get("vera_lab_source_commit"):
        reasons.append("vera_lab_source_commit_mismatch")
    elif not _is_commit(lab.get("source_commit")):
        reasons.append("invalid_commit:vera_lab.source_commit")

    _digest_reason(lab.get("scenario_set_digest"), "vera_lab.scenario_set_digest", reasons)
    if lab.get("source_review_verdict") not in PASSING_REVIEW_VERDICTS:
        reasons.append("vera_lab_source_review_not_pass")
    if lab.get("behavior_review_verdict") not in PASSING_REVIEW_VERDICTS:
        reasons.append("vera_lab_behavior_review_not_pass")
    _digest_reason(lab.get("source_review_digest"), "vera_lab.source_review_digest", reasons)
    _digest_reason(lab.get("behavior_review_digest"), "vera_lab.behavior_review_digest", reasons)


def _corpus_reasons(evidence: dict[str, Any], reasons: list[str]) -> None:
    corpus = _section(evidence, "corpus", reasons)
    if corpus.get("frozen") is not True:
        reasons.append("corpus_not_frozen")
    for field in ("train_digest", "validation_digest", "general_competence_pool_digest"):
        _digest_reason(corpus.get(field), f"corpus.{field}", reasons)
    overlap = corpus.get("train_validation_overlap")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap != 0:
        reasons.append("train_validation_overlap_nonzero")


def _semantic_closure_reasons(evidence: dict[str, Any], reasons: list[str]) -> None:
    closure = _section(evidence, "semantic_closure", reasons)
    for target in SEMANTIC_CLOSURE_TARGETS:
        record = closure.get(target)
        if not isinstance(record, dict):
            reasons.append(f"missing_semantic_closure:{target}")
            continue
        if record.get("status") not in SEMANTIC_CLOSURE_STATES:
            reasons.append(f"invalid_semantic_closure_status:{target}")
        _digest_reason(record.get("receipt_digest"), f"semantic_closure.{target}.receipt_digest", reasons)


def _reviewer_reasons(evidence: dict[str, Any], reasons: list[str]) -> set[str]:
    reviewers = _section(evidence, "reviewers", reasons)
    training_lane = evidence.get("training_lane_key")
    lanes: list[str] = []
    for role in REVIEWER_ROLES:
        registration = reviewers.get(role)
        if not isinstance(registration, dict):
            reasons.append(f"missing_reviewer:{role}")
            continue
        if registration.get("registered") is not True:
            reasons.append(f"reviewer_not_registered:{role}")
        lane = registration.get("lane_key")
        if not _is_nonempty_text(lane):
            reasons.append(f"invalid_reviewer_lane:{role}")
        else:
            lanes.append(lane)
            if lane == training_lane:
                reasons.append(f"reviewer_lane_is_training_lane:{role}")
        _digest_reason(
            registration.get("registration_receipt_digest"),
            f"reviewers.{role}.registration_receipt_digest",
            reasons,
        )
    if len(lanes) == len(REVIEWER_ROLES) and len(set(lanes)) != len(lanes):
        reasons.append("hostile_reviewer_lane_collision")
    return set(lanes)


def _custodian_reasons(
    evidence: dict[str, Any],
    reviewer_lanes: set[str],
    reasons: list[str],
) -> None:
    custodian = _section(evidence, "blind_custodian", reasons)
    training_lane = evidence.get("training_lane_key")
    if custodian.get("registered") is not True:
        reasons.append("blind_custodian_not_registered")

    lane = custodian.get("lane_key")
    if not _is_nonempty_text(lane):
        reasons.append("invalid_blind_custodian_lane")
    elif lane == training_lane or lane in reviewer_lanes:
        reasons.append("blind_custodian_lane_collision")

    for field in ("registration_receipt_digest", "set_digest", "manifest_digest"):
        _digest_reason(custodian.get(field), f"blind_custodian.{field}", reasons)

    item_count = custodian.get("item_count")
    if not isinstance(item_count, int) or isinstance(item_count, bool) or item_count < 70:
        reasons.append("blind_item_count_below_70")

    family_counts = custodian.get("family_counts")
    if not isinstance(family_counts, dict):
        reasons.append("missing_blind_family_counts")
    else:
        for family in CRITICAL_FAMILIES:
            count = family_counts.get(family)
            if not isinstance(count, int) or isinstance(count, bool) or count < 5:
                reasons.append(f"blind_family_below_5:{family}")

    if custodian.get("leakage_check_status") != "PASS":
        reasons.append("blind_leakage_check_not_pass")
    if custodian.get("plaintext_visible_to_training_lane") is not False:
        reasons.append("blind_plaintext_visible_to_training_lane")


def check_v3_readiness(evidence: dict[str, Any] | None) -> ReadinessDecision:
    if not isinstance(evidence, dict):
        safe_evidence: dict[str, Any] = {}
        reasons = ["evidence_not_object"]
    else:
        safe_evidence = evidence
        reasons: list[str] = []

    if safe_evidence.get("schema_version") != READINESS_SCHEMA_VERSION:
        reasons.append("unsupported_readiness_schema_version")
    if not _is_nonempty_text(safe_evidence.get("training_lane_key")):
        reasons.append("invalid_training_lane_key")

    _source_reasons(safe_evidence, reasons)
    _vera_lab_reasons(safe_evidence, reasons)
    _corpus_reasons(safe_evidence, reasons)
    _semantic_closure_reasons(safe_evidence, reasons)
    reviewer_lanes = _reviewer_reasons(safe_evidence, reasons)
    _custodian_reasons(safe_evidence, reviewer_lanes, reasons)

    reasons = list(dict.fromkeys(reasons))
    subject_digest = _sha256_json(_subject_payload(safe_evidence))
    evidence_digest = _sha256_json(safe_evidence)

    # This source-only gate can validate structural consistency, exact-shaped
    # identifiers, and cross-field bindings. It cannot establish that review,
    # registration, corpus-freeze, leakage, or custodian claims came from the
    # independent authorities they name. Never let self-authored evidence mint
    # a weight-training READY receipt.
    if not reasons:
        reasons.append("external_authority_verification_required")
        status = "HOLD"
    else:
        status = "BLOCKED"

    return ReadinessDecision(
        ready=False,
        status=status,
        subject_digest=subject_digest,
        evidence_digest=evidence_digest,
        reasons=tuple(reasons),
    )
