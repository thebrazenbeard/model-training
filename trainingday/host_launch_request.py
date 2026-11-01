"""Immutable *declaration* of a proposed TrainingDay host launch.

This source-only envelope has no authentication, trusted clock, GPU lease,
dataset-rights authority, non-replay store, or host-controlled execution.
Matching its syntax or SHA-256 is NEVER permission to train, resume, read
protected data, or mutate files. The unconditional physical launch HARD HOLD
in physical_launch_authority must remain in place.

A future independently deployed host-owned broker must authenticate its caller,
re-establish every declared binding from trusted sources, acquire/hold the GPU
lease, atomically check the replay witness and expiry on host time, and launch
the contained process itself. The broker cannot trust a caller's assertion
that any of those checks has succeeded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re


SCHEMA = "TRAININGDAY_HOST_LAUNCH_DECLARATION_V1"
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GENESIS_WITNESS_HEAD = "0" * 64
_HEX40 = re.compile(r"[a-f0-9]{40}\Z")
_HEX64 = re.compile(r"[a-f0-9]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}\Z")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_MAX_COUNTER = (1 << 63) - 2


@dataclass(frozen=True, slots=True)
class SourceBinding:
    source_commit_sha1: str
    source_tree_sha1: str
    corpus_manifest_sha256: str
    corpus_producer_commit_sha1: str
    training_input_sha256: str
    base_binding_sha256: str
    base_model_id: str
    base_revision_sha1: str


@dataclass(frozen=True, slots=True)
class RightsDeclaration:
    """A claimed owner decision: neither signed nor verified by this module."""

    decision_id: str
    decision_sha256: str
    covered_manifest_sha256: str
    declared_disposition: str
    purported_owner: str


@dataclass(frozen=True, slots=True)
class RunPlan:
    run_id: str
    protocol: str
    mode: str
    optimizer_steps: int
    selected_cases: int
    seed: int
    # An integer part-per-billion rate avoids floating-point JSON ambiguity.
    learning_rate_ppb: int
    expected_zero_digest_sha256: str | None


@dataclass(frozen=True, slots=True)
class HostLeaseDeclaration:
    """An untrusted host/lease/replay expectation, not a grant or nonce proof."""

    host_instance: str
    caller_principal: str
    gpu_lease_id: str
    gpu_lease_epoch: int
    request_nonce: str
    issued_at_unix: int
    expires_at_unix: int
    expected_witness_epoch: int
    expected_witness_head_sha256: str


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    schema: str
    source: SourceBinding
    rights: RightsDeclaration
    plan: RunPlan
    lease: HostLeaseDeclaration


@dataclass(frozen=True, slots=True)
class LaunchRequestAssessment:
    status: str
    reasons: tuple[str, ...]
    request_sha256: str | None
    authorizes_training: bool = False
    authorizes_resume: bool = False
    host_identity_verified: bool = False
    dataset_rights_verified: bool = False
    gpu_lease_verified: bool = False
    replay_prevention_verified: bool = False
    physical_training_observed: bool = False


def _sha(value: object, regex: re.Pattern[str]) -> bool:
    return type(value) is str and regex.fullmatch(value) is not None


def _id(value: object) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _counter(value: object, minimum: int = 0) -> bool:
    # bool is an int subclass; a forged true cannot stand in for an epoch.
    return type(value) is int and minimum <= value <= _MAX_COUNTER


def _reasons(request: object) -> tuple[str, ...]:
    if type(request) is not LaunchRequest:
        return ("INVALID_REQUEST_TYPE",)
    if type(request.schema) is not str or request.schema != SCHEMA:
        return ("INVALID_SCHEMA",)
    if (type(request.source) is not SourceBinding
            or type(request.rights) is not RightsDeclaration
            or type(request.plan) is not RunPlan
            or type(request.lease) is not HostLeaseDeclaration):
        return ("INVALID_NESTED_TYPE",)

    errors: list[str] = []
    src = request.source
    for name in ("source_commit_sha1", "source_tree_sha1",
                 "corpus_producer_commit_sha1", "base_revision_sha1"):
        if not _sha(getattr(src, name), _HEX40):
            errors.append("INVALID_" + name.upper())
    for name in ("corpus_manifest_sha256", "training_input_sha256",
                 "base_binding_sha256"):
        if not _sha(getattr(src, name), _HEX64):
            errors.append("INVALID_" + name.upper())
    if (type(src.base_model_id) is not str or src.base_model_id != BASE_MODEL
            or type(src.base_revision_sha1) is not str
            or src.base_revision_sha1 != BASE_REVISION):
        errors.append("UNFROZEN_QWEN_BASE")

    rights = request.rights
    if not _id(rights.decision_id) or not _id(rights.purported_owner):
        errors.append("INVALID_RIGHTS_OWNER_OR_DECISION")
    if not _sha(rights.decision_sha256, _HEX64):
        errors.append("INVALID_RIGHTS_RECORD_DIGEST")
    if (not _sha(src.corpus_manifest_sha256, _HEX64)
            or not _sha(rights.covered_manifest_sha256, _HEX64)
            or rights.covered_manifest_sha256 != src.corpus_manifest_sha256):
        errors.append("RIGHTS_CLAIM_CORPUS_MISMATCH")
    if (type(rights.declared_disposition) is not str
            or rights.declared_disposition not in {
                "TRAINING_PERMITTED_CLAIM", "HOLD_CLAIM",
            }):
        errors.append("INVALID_RIGHTS_DISPOSITION_CLAIM")

    plan = request.plan
    if type(plan.run_id) is not str or _RUN_ID.fullmatch(plan.run_id) is None:
        errors.append("INVALID_RUN_ID")
    if type(plan.protocol) is not str or plan.protocol not in {
        "R15_FRESH", "R16_CHECKPOINTABLE",
    }:
        errors.append("INVALID_PROTOCOL")
    if type(plan.mode) is not str or plan.mode not in {
        "zero", "correct", "corrupt",
    }:
        errors.append("INVALID_MODE")
    if not _counter(plan.optimizer_steps):
        errors.append("INVALID_OPTIMIZER_STEPS")
    if not (type(plan.selected_cases) is int and 1 <= plan.selected_cases <= 1024):
        errors.append("INVALID_SELECTED_CASES")
    if not (type(plan.seed) is int and 0 <= plan.seed <= 2**32 - 1):
        errors.append("INVALID_SEED")
    if not (type(plan.learning_rate_ppb) is int
            and 1 <= plan.learning_rate_ppb <= 200_000):
        errors.append("INVALID_LEARNING_RATE")
    if (type(plan.mode) is str and plan.mode == "zero"
            and (type(plan.optimizer_steps) is not int
                 or plan.optimizer_steps != 0
                 or plan.expected_zero_digest_sha256 is not None)):
        errors.append("INVALID_ZERO_ARM")
    if type(plan.mode) is str and plan.mode in {"correct", "corrupt"}:
        if not _sha(plan.expected_zero_digest_sha256, _HEX64):
            errors.append("INVALID_CONTROL_INITIAL_DIGEST")
        if type(plan.protocol) is str and type(plan.optimizer_steps) is int:
            if plan.protocol == "R15_FRESH" and not 1 <= plan.optimizer_steps <= 8:
                errors.append("R15_STEPS_OUT_OF_BOUNDS")
            if plan.protocol == "R16_CHECKPOINTABLE" and not (
                    8 <= plan.optimizer_steps <= 32 and plan.optimizer_steps % 8 == 0):
                errors.append("R16_CHECKPOINT_BOUNDARY_REQUIRED")
        if (type(plan.selected_cases) is int and type(plan.optimizer_steps) is int
                and plan.selected_cases < plan.optimizer_steps * 8):
            errors.append("INSUFFICIENT_CASES_FOR_STEPS")

    lease = request.lease
    for name in ("host_instance", "caller_principal", "gpu_lease_id"):
        if not _id(getattr(lease, name)):
            errors.append("INVALID_" + name.upper())
    if not _counter(lease.gpu_lease_epoch, 1):
        errors.append("INVALID_GPU_LEASE_EPOCH")
    if not _sha(lease.request_nonce, _HEX64):
        errors.append("INVALID_NONCE_DECLARATION")
    if not (_counter(lease.issued_at_unix, 1)
            and _counter(lease.expires_at_unix, 1)
            and 1 <= lease.expires_at_unix - lease.issued_at_unix <= 3600):
        errors.append("INVALID_CLAIMED_LEASE_WINDOW")
    if not _counter(lease.expected_witness_epoch):
        errors.append("INVALID_REPLAY_EPOCH_DECLARATION")
    if not _sha(lease.expected_witness_head_sha256, _HEX64):
        errors.append("INVALID_REPLAY_HEAD_DECLARATION")
    if (_counter(lease.expected_witness_epoch)
            and _sha(lease.expected_witness_head_sha256, _HEX64)
            and ((lease.expected_witness_epoch == 0)
                 != (lease.expected_witness_head_sha256 == GENESIS_WITNESS_HEAD))):
        errors.append("INCONSISTENT_REPLAY_GENESIS_CLAIM")
    return tuple(errors)


def canonical_request_bytes(request: LaunchRequest) -> bytes:
    """Deterministic bytes after strict validation; never a signature."""
    errors = _reasons(request)
    if errors:
        raise ValueError("Invalid launch declaration: " + ",".join(errors))
    return json.dumps(
        asdict(request), ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def assess_launch_request(request: object) -> LaunchRequestAssessment:
    """Validate a source proposal; always refuse physical effect authority."""
    errors = _reasons(request)
    if errors:
        return LaunchRequestAssessment(
            status="SOURCE_DECLARATION_REJECTED",
            reasons=errors,
            request_sha256=None,
        )
    digest = hashlib.sha256(canonical_request_bytes(request)).hexdigest()
    return LaunchRequestAssessment(
        status="SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED",
        reasons=(),
        request_sha256=digest,
    )
