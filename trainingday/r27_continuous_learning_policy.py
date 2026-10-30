"""R27 public/synthetic continuous-learning consent and freeze policy shadow.

This is a deterministic assessor for caller-supplied declarations, not an
authorization service. It neither authenticates Patrick's consent nor reads a
host, model checkpoint, protected examination, GPU, or deployed release. Every
outcome is HOLD; no state transition enables training, online updates, model
promotion, or an actual rollback. No model weights or evidence are deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


SCHEMA = "VERA_TRAININGDAY_R27_CONTINUOUS_LEARNING_POLICY_SHADOW_V1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_KINDS = frozenset({
    "REQUEST", "FREEZE", "CONSENT_CLAIM", "PUBLIC_REVIEW",
    "RIGHTS_REVOKED", "PRIVACY_TAINTED", "TRAINING_STARTED",
})
_RIGHTS_STATUSES = frozenset({
    "CLEAR_DECLARED", "REVIEW_REQUIRED", "REVOKED", "PRIVACY_TAINTED",
})
_MAX_EVENTS = 32
_MAX_UNIX = 2**53 - 1


class PolicyError(ValueError):
    """Malformed public declaration, never a permission to attempt recovery."""


@dataclass(frozen=True)
class FrozenBaseline:
    base_sha256: str
    adapter_sha256: str
    checkpoint_sha256: str
    evidence_sha256: str
    rights_sha256: str


@dataclass(frozen=True)
class CycleRequest:
    cycle_id: str
    requester_id: str
    objective_id: str
    source_sha256: str
    baseline_sha256: str
    proposed_adapter_sha256: str
    proposed_checkpoint_sha256: str


@dataclass(frozen=True)
class ConsentClaim:
    cycle_id: str
    issuer_id: str
    issuer_role: str
    decision: str
    baseline_sha256: str
    source_sha256: str
    objective_id: str
    proposed_adapter_sha256: str
    proposed_checkpoint_sha256: str
    issued_unix: int
    expires_unix: int
    nonce: str
    max_active_seconds: int
    max_cost_cents: int
    max_energy_mwh: int


@dataclass(frozen=True)
class CycleEvidence:
    cycle_id: str
    previous_adapter_sha256: str
    candidate_adapter_sha256: str
    candidate_checkpoint_sha256: str
    rights_sha256: str
    public_before_ppm: int
    public_after_ppm: int
    public_regressions: int
    unseen_regressions: int
    frozen_control_present: bool
    zero_control_present: bool
    corrupted_control_present: bool
    active_seconds: int
    cost_cents: int
    energy_mwh: int
    actual_tensor_bytes_independently_verified: bool
    rights_status: str = "CLEAR_DECLARED"


@dataclass(frozen=True)
class CycleEvent:
    seq: int
    kind: str
    actor_id: str
    unix: int


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _integer(value: object, lower: int, upper: int) -> bool:
    return type(value) is int and lower <= value <= upper


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"), sort_keys=True).encode("ascii")


def baseline_digest(baseline: FrozenBaseline) -> str:
    """A stable source-commitment hash; not a hash of authenticated bytes."""
    if type(baseline) is not FrozenBaseline or not all(_sha(value) for value in (
        baseline.base_sha256, baseline.adapter_sha256,
        baseline.checkpoint_sha256, baseline.evidence_sha256,
        baseline.rights_sha256,
    )):
        raise PolicyError("frozen baseline identity malformed")
    return hashlib.sha256(_canonical({
        "base": baseline.base_sha256,
        "adapter": baseline.adapter_sha256,
        "checkpoint": baseline.checkpoint_sha256,
        "evidence": baseline.evidence_sha256,
        "rights": baseline.rights_sha256,
    })).hexdigest()


def _validate(request: object, consent: object, evidence: object,
              events: object, now_unix: object) -> None:
    if (type(request) is not CycleRequest
            or not all(_id(x) for x in (
                request.cycle_id, request.requester_id, request.objective_id))
            or not all(_sha(x) for x in (
                request.source_sha256, request.baseline_sha256,
                request.proposed_adapter_sha256,
                request.proposed_checkpoint_sha256))):
        raise PolicyError("cycle request malformed")
    if (type(consent) is not ConsentClaim
            or not all(_id(x) for x in (
                consent.cycle_id, consent.issuer_id, consent.nonce))
            or type(consent.issuer_role) is not str
            or type(consent.decision) is not str
            or not _id(consent.objective_id)
            or not all(_sha(x) for x in (
                consent.baseline_sha256, consent.source_sha256,
                consent.proposed_adapter_sha256,
                consent.proposed_checkpoint_sha256))
            or not _integer(consent.issued_unix, 0, _MAX_UNIX)
            or not _integer(consent.expires_unix, 0, _MAX_UNIX)
            or not _integer(consent.max_active_seconds, 1, 3600)
            or not _integer(consent.max_cost_cents, 0, 100_000)
            or not _integer(consent.max_energy_mwh, 0, 100_000_000)):
        raise PolicyError("consent claim malformed or unbounded")
    if (type(evidence) is not CycleEvidence
            or not _id(evidence.cycle_id)
            or not all(_sha(x) for x in (
                evidence.previous_adapter_sha256,
                evidence.candidate_adapter_sha256,
                evidence.candidate_checkpoint_sha256, evidence.rights_sha256))
            or not all(_integer(x, 0, 1_000_000) for x in (
                evidence.public_before_ppm, evidence.public_after_ppm))
            or not all(_integer(x, 0, 4096) for x in (
                evidence.public_regressions, evidence.unseen_regressions))
            or not all(_integer(x, 0, 2**31 - 1) for x in (
                evidence.active_seconds, evidence.cost_cents, evidence.energy_mwh))
            or any(type(x) is not bool for x in (
                evidence.frozen_control_present, evidence.zero_control_present,
                evidence.corrupted_control_present,
                evidence.actual_tensor_bytes_independently_verified))
            or type(evidence.rights_status) is not str
            or evidence.rights_status not in _RIGHTS_STATUSES):
        raise PolicyError("public cycle evidence malformed")
    if not _integer(now_unix, 0, _MAX_UNIX):
        raise PolicyError("assessment time malformed")
    if (type(events) is not tuple or not 4 <= len(events) <= _MAX_EVENTS
            or any(type(e) is not CycleEvent or not _integer(e.seq, 1, _MAX_EVENTS)
                   or type(e.kind) is not str or not _id(e.actor_id)
                   or not _integer(e.unix, 0, _MAX_UNIX) for e in events)):
        raise PolicyError("cycle event declarations malformed or unbounded")
    seqs = [e.seq for e in events]
    if len(seqs) != len(set(seqs)):
        raise PolicyError("cycle event sequence replay")


def assess_cycle(*, baseline: FrozenBaseline,
                 request: CycleRequest, consent: ConsentClaim,
                 evidence: CycleEvidence,
                 events: tuple[CycleEvent, ...], now_unix: int) -> dict:
    """Check a future-cycle declaration. Never authorize effects.

    Even if all source fields are internally consistent, consent, host leases,
    rights and independent scorer receipts are untrusted caller assertions.
    """
    pinned = baseline_digest(baseline)
    _validate(request, consent, evidence, events, now_unix)
    blockers: list[str] = []

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    if request.baseline_sha256 != pinned:
        block("BASELINE_REPIN_MISMATCH")
    if (request.cycle_id != consent.cycle_id
            or request.baseline_sha256 != consent.baseline_sha256
            or request.source_sha256 != consent.source_sha256
            or request.objective_id != consent.objective_id
            or request.proposed_adapter_sha256 != consent.proposed_adapter_sha256
            or request.proposed_checkpoint_sha256 != consent.proposed_checkpoint_sha256):
        block("CONSENT_SCOPE_MISMATCH")
    if request.requester_id == consent.issuer_id:
        block("CONSENT_PRINCIPAL_NOT_SEPARATE")
    if (consent.issuer_role != "HUMAN_OWNER_CLAIMED"
            or consent.decision != "APPROVE_CLAIMED"):
        block("CONSENT_NOT_APPROVED")
    if consent.issued_unix > now_unix or consent.expires_unix <= now_unix:
        block("CONSENT_EXPIRED")
    if consent.expires_unix <= consent.issued_unix:
        block("CONSENT_INTERVAL_INVALID")
    if evidence.cycle_id != request.cycle_id:
        block("EVIDENCE_CYCLE_MISMATCH")
    if evidence.rights_status != "CLEAR_DECLARED":
        block("RIGHTS_NOT_CLEAR")
    if (evidence.previous_adapter_sha256 != baseline.adapter_sha256
            or evidence.candidate_adapter_sha256 != request.proposed_adapter_sha256
            or evidence.candidate_checkpoint_sha256 != request.proposed_checkpoint_sha256
            or evidence.rights_sha256 != baseline.rights_sha256):
        block("CANDIDATE_IDENTITY_MISMATCH")
    if (evidence.candidate_adapter_sha256 == baseline.adapter_sha256
            or request.proposed_adapter_sha256 == baseline.adapter_sha256):
        block("NO_DECLARED_ADAPTER_CHANGE")
    if evidence.public_regressions:
        block("PUBLIC_REGRESSION")
    if evidence.unseen_regressions:
        block("UNSEEN_REGRESSION")
    if evidence.public_after_ppm < evidence.public_before_ppm:
        block("PUBLIC_SCORE_DECREASE")
    if evidence.active_seconds == 0:
        block("NO_DECLARED_TRAINING_PROGRESS")
    if not (evidence.frozen_control_present and evidence.zero_control_present
            and evidence.corrupted_control_present):
        block("CONTROL_INCOMPLETE")
    if (evidence.active_seconds > consent.max_active_seconds
            or evidence.cost_cents > consent.max_cost_cents
            or evidence.energy_mwh > consent.max_energy_mwh):
        block("DECLARED_BUDGET_OVERRUN")

    expected_prefix = ("REQUEST", "FREEZE", "CONSENT_CLAIM", "PUBLIC_REVIEW")
    if (tuple(e.kind for e in events[:4]) != expected_prefix
            or any(e.kind not in ("RIGHTS_REVOKED", "PRIVACY_TAINTED",
                                  "TRAINING_STARTED") for e in events[4:])
            or any(e.seq != index for index, e in enumerate(events, start=1))
            or any(b.unix < a.unix for a, b in zip(events, events[1:]))
            or any(e.kind not in _KINDS for e in events)):
        block("INVALID_EVENT_ORDER")
    expected_actor = {
        "REQUEST": request.requester_id,
        "CONSENT_CLAIM": consent.issuer_id,
    }
    if any(e.kind in expected_actor and e.actor_id != expected_actor[e.kind]
           for e in events):
        block("EVENT_PRINCIPAL_MISMATCH")
    if (events[1].actor_id in (request.requester_id, consent.issuer_id)
            or events[3].actor_id in (request.requester_id, consent.issuer_id,
                                      events[1].actor_id)):
        block("CLAIMED_REVIEWER_NOT_SEPARATE")
    if events[2].unix < consent.issued_unix or events[2].unix > consent.expires_unix:
        block("CONSENT_NOT_CURRENT")
    if any(e.unix > now_unix for e in events):
        block("FUTURE_EVENT")

    saw_revocation = False
    for event in events:
        if saw_revocation and event.kind not in ("RIGHTS_REVOKED", "PRIVACY_TAINTED"):
            block("EVENT_AFTER_REVOCATION")
        if event.kind == "RIGHTS_REVOKED":
            saw_revocation = True
            block("RIGHTS_REVOKED")
        if event.kind == "PRIVACY_TAINTED":
            saw_revocation = True
            block("PRIVACY_TAINTED")
        if event.kind == "TRAINING_STARTED":
            block("UNAUTHORIZED_TRAINING_EVENT")

    return {
        "schema": SCHEMA,
        "status": "HOLD",
        "reason": "SOURCE_DECLARATION_ONLY" if not blockers else "POLICY_BLOCKED",
        "blockers": blockers,
        "original_baseline_sha256": pinned,
        "synthetic_source_conditions_satisfied": not blockers,
        "previous_snapshot_preserved": True,
        "consent_independently_authenticated": False,
        "host_lease_authenticated": False,
        "tensor_bytes_independently_verified": False,
        "independent_unseen_evaluator_qualified": False,
        "gpu_start_authorized": False,
        "continuous_weight_update_authorized": False,
        "model_promotion_authorized": False,
        "physical_rollback_proven": False,
        "unlearning_verified": False,
    }
