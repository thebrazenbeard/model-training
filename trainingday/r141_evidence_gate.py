"""Source-only TrainingDay qualification-evidence packet assessor.

The distinction between an evidence *claim* and an independently verified
physical result is deliberate. All records presented to this pure Python
function are caller controlled. No input field, including a purported host
signature, can confer training, resume, unseal, or graduation authority.

Protocol V2 allows bounded reversible source/testing work; it does not make
source tests equivalent to native-model qualification or override protected
effects, private-evaluation custody, exclusive host ownership, or cost gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_GIT_HEAD = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}\Z")

# Evidence classes are not ordered: a model review cannot upgrade a test, and
# neither is a substitute for a genuine physical probe.
EVIDENCE_CLASSES = frozenset({
    "DOC", "DEP_METADATA", "SOURCE_INSPECTION", "TEST", "REAL_SYSTEM_PROBE",
    "FORMAL_VERIFY", "MODEL_REVIEW", "HUMAN_REVIEW",
})

# All six requirements need an actual real-system probe report, not a source
# assertion or merely an independent opinion. Evidence itself remains
# *unverified* until a host-owned authority validates it out of band.
REQUIRED_PHYSICAL_PROBES = frozenset({
    "EXCLUSIVE_HOST_LEASE",
    "INDEPENDENT_MONOTONIC_WITNESS",
    "PINNED_QWEN_WEIGHT_LOAD",
    "CHECKPOINT_RELOAD",
    "OPTIMIZER_RNG_CONTINUITY",
    "SEALED_HELDOUT_EVALUATION",
})


@dataclass(frozen=True)
class EvidenceClaim:
    requirement: str
    evidence_class: str
    git_head: str
    run_id: str
    checkpoint_sha256: str
    issuer: str
    receipt_sha256: str
    passed: bool


@dataclass(frozen=True)
class QualificationPacket:
    git_head: str
    run_id: str
    checkpoint_sha256: str
    trainer_principal: str
    claims: tuple[EvidenceClaim, ...]


@dataclass(frozen=True)
class QualificationDecision:
    reason: str
    missing_physical_claims: tuple[str, ...]
    documentary_packet_claimed_complete: bool = False
    # NEVER change these from False in a source-only assessor. They are
    # downstream host-policy decisions requiring separately authenticated
    # custody, physical trials, and independent evaluation.
    training_authorized: bool = False
    checkpoint_resume_authorized: bool = False
    sealed_eval_unseal_authorized: bool = False
    model_learning_qualified: bool = False


def _good_id(value: object) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _good_hash(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _hold(reason: str, missing: set[str]) -> QualificationDecision:
    return QualificationDecision(reason, tuple(sorted(missing)))


def assess_qualification_packet(packet: QualificationPacket) -> QualificationDecision:
    """Check structure and self-consistency, NEVER trust or exercise authority.

    Even a ``documentary_packet_claimed_complete`` decision only means the
    *caller* supplied six well-shaped, mutually bound reports of purported
    real-system probes. It does not authenticate a provider, witness, probe,
    native model, sealed bank, or exclusive owner; it is not a qualification.
    """
    missing = set(REQUIRED_PHYSICAL_PROBES)
    if type(packet) is not QualificationPacket:
        return _hold("MALFORMED_PACKET", missing)
    if not _good_hash(packet.git_head, _GIT_HEAD):
        return _hold("EXACT_GIT_HEAD_REQUIRED", missing)
    if not _good_hash(packet.checkpoint_sha256, _SHA256):
        return _hold("CHECKPOINT_DIGEST_REQUIRED", missing)
    if not _good_id(packet.run_id) or not _good_id(packet.trainer_principal):
        return _hold("MALFORMED_IDENTITY_BINDING", missing)
    if type(packet.claims) is not tuple or len(packet.claims) > 128:
        return _hold("MALFORMED_CLAIM_SET", missing)

    seen: set[str] = set()
    seen_receipts: set[str] = set()
    claimed: set[str] = set()
    for claim in packet.claims:
        if type(claim) is not EvidenceClaim:
            return _hold("MALFORMED_EVIDENCE_CLAIM", missing)
        if type(claim.requirement) is not str:
            return _hold("MALFORMED_EVIDENCE_CLAIM", missing)
        if claim.requirement not in REQUIRED_PHYSICAL_PROBES:
            return _hold("UNKNOWN_REQUIREMENT", missing)
        if claim.requirement in seen:
            return _hold("DUPLICATE_OR_CONFLICTING_REQUIREMENT", missing)
        seen.add(claim.requirement)
        if (type(claim.evidence_class) is not str
                or claim.evidence_class not in EVIDENCE_CLASSES):
            return _hold("UNKNOWN_EVIDENCE_CLASS", missing)
        if (not _good_hash(claim.git_head, _GIT_HEAD)
                or not _good_id(claim.run_id)
                or not _good_hash(claim.checkpoint_sha256, _SHA256)):
            return _hold("MALFORMED_EVIDENCE_CLAIM", missing)
        if (claim.git_head != packet.git_head
                or claim.run_id != packet.run_id
                or claim.checkpoint_sha256 != packet.checkpoint_sha256):
            return _hold("EXACT_ARTIFACT_BINDING_MISMATCH", missing)
        if not _good_id(claim.issuer) or not _good_hash(claim.receipt_sha256, _SHA256):
            return _hold("MALFORMED_RECEIPT_IDENTITY", missing)
        if claim.receipt_sha256 in seen_receipts:
            return _hold("REPLAYED_RECEIPT_ACROSS_REQUIREMENTS", missing)
        seen_receipts.add(claim.receipt_sha256)
        if type(claim.passed) is not bool:
            return _hold("MALFORMED_PROBE_RESULT", missing)
        if (claim.requirement in {
                "EXCLUSIVE_HOST_LEASE", "INDEPENDENT_MONOTONIC_WITNESS",
                "SEALED_HELDOUT_EVALUATION",
            } and claim.issuer == packet.trainer_principal):
            return _hold("INDEPENDENT_ISSUER_REQUIRED", missing)
        if claim.evidence_class == "REAL_SYSTEM_PROBE" and claim.passed:
            claimed.add(claim.requirement)

    missing -= claimed
    if missing:
        return _hold("PHYSICAL_PROBE_REPORTS_MISSING_OR_WRONG_CLASS", missing)
    return QualificationDecision(
        reason="ALL_PHYSICAL_REPORTS_ASSERTED_BUT_UNVERIFIED_HOST_AUTHORITY_HOLD",
        missing_physical_claims=(),
        documentary_packet_claimed_complete=True,
    )
