"""R140 Windows checkpoint-custody admission *contract*, not an executor.

The current R18 publisher can observe source bytes, but a pathname/metadata
snapshot is not a host-held, immutable filesystem capability. This module is
intentionally unable to open paths, dispatch work, issue a lease, or authorize
resume/training. It describes observations that a future *native host-owned*
authority must bind to real Windows handles and an independently durable
replay witness. All supplied observations remain untrusted assertions.

In particular, a passing synthetic preflight is **never** a physical Windows
test or permission for a CUDA launch, checkpoint adoption, or later replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_HEX_128 = re.compile(r"[0-9a-f]{32}\Z")
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_VOLUME_GUID = re.compile(
    r"\{[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}\Z"
)
_BASENAME = re.compile(r"checkpoint-[1-9][0-9]{0,9}\Z")


@dataclass(frozen=True, slots=True)
class WindowsCustodyIntent:
    """Source-declared expectations; no raw Windows pathname is accepted.

    ``host_instance`` is an opaque per-boot identity, not a machine nickname.
    ``volume_guid`` names a volume, not a drive letter or UNC share. File IDs
    and the process job token must be attested by a native host service; merely
    constructing this object creates no trusted capability.
    """

    host_instance: str
    volume_guid: str
    parent_file_id: str
    stage_file_id: str
    job_token: str
    lease_epoch: int
    dispatch_nonce: str
    target_basename: str


@dataclass(frozen=True, slots=True)
class WindowsCustodyObservation:
    """Untrusted report of observations a *real* host service must establish.

    No `verified` or `authorized` bit is part of this type: such a bit supplied
    by the caller could only assert, never establish, trusted provider binding.
    True-valued checks here are useful in source adversarial tests only.
    """

    host_instance: str
    volume_guid: str
    parent_file_id: str
    stage_file_id: str
    job_token: str
    lease_epoch: int
    dispatch_nonce: str
    target_basename: str
    # All ancestor and member handles must have been held across admission,
    # publication, and source-byte readback. Re-open-by-path is insufficient.
    pinned_ancestor_handles: bool
    parent_and_stage_file_ids_held: bool
    reparse_points_excluded: bool
    same_volume_nonreplacing_publish: bool
    # A Job Object alone cannot prevent a previously accepted remote command
    # from executing after a tunnel reconnect. Both boundaries are necessary.
    job_kill_on_close_enforced: bool
    job_breakaway_prohibited: bool
    all_descendant_processes_enrolled: bool
    prior_job_termination_acknowledged: bool
    pending_writes_drained: bool
    delayed_dispatch_replay_excluded: bool
    lease_owner_acked: bool
    # This must eventually be witnessed by an *independently durable* source,
    # not another field inside the same rollback-able checkpoint directory.
    external_nonce_witness_acknowledged: bool


@dataclass(frozen=True, slots=True)
class WindowsCustodyPreflight:
    status: str
    reasons: tuple[str, ...]
    synthetic_conditions_met: bool
    authorizes_training: bool = False
    authorizes_resume: bool = False
    authorizes_filesystem_mutation: bool = False
    physical_windows_tested: bool = False
    authoritative_host_binding_verified: bool = False


_MATCH_FIELDS = (
    "host_instance", "volume_guid", "parent_file_id", "stage_file_id",
    "job_token", "lease_epoch", "dispatch_nonce", "target_basename",
)
_OBSERVATION_CHECKS = (
    "pinned_ancestor_handles", "parent_and_stage_file_ids_held",
    "reparse_points_excluded", "same_volume_nonreplacing_publish",
    "job_kill_on_close_enforced", "job_breakaway_prohibited",
    "all_descendant_processes_enrolled", "prior_job_termination_acknowledged",
    "pending_writes_drained", "delayed_dispatch_replay_excluded",
    "lease_owner_acked", "external_nonce_witness_acknowledged",
)


def _intent_syntax_reasons(intent: WindowsCustodyIntent) -> list[str]:
    errors: list[str] = []
    for field in ("host_instance", "parent_file_id", "stage_file_id"):
        value = getattr(intent, field)
        if type(value) is not str or _HEX_128.fullmatch(value) is None:
            errors.append(f"INVALID_{field.upper()}")
    for field in ("job_token", "dispatch_nonce"):
        value = getattr(intent, field)
        if type(value) is not str or _HEX_256.fullmatch(value) is None:
            errors.append(f"INVALID_{field.upper()}")
    if (type(intent.volume_guid) is not str
            or _VOLUME_GUID.fullmatch(intent.volume_guid) is None):
        errors.append("INVALID_VOLUME_GUID")
    if (type(intent.lease_epoch) is not int
            or not (1 <= intent.lease_epoch <= (2**63 - 1))):
        errors.append("INVALID_LEASE_EPOCH")
    if (type(intent.target_basename) is not str
            or _BASENAME.fullmatch(intent.target_basename) is None
            or int(intent.target_basename.removeprefix("checkpoint-")) > 2**31 - 1):
        errors.append("INVALID_TARGET_BASENAME")
    return errors


def assess_windows_custody_preflight(
    intent: WindowsCustodyIntent, observations: WindowsCustodyObservation,
) -> WindowsCustodyPreflight:
    """Compare *claims* without yielding a mutable host capability.

    A native authority must ultimately prove immutable handle custody, host
    ownership, antirollback state, Job Object containment, and actual Windows
    crash/reconnect behavior. No such authority is imported here. Even if all
    synthetic claims align, every permission remains false.
    """
    reasons: list[str] = []
    if type(intent) is not WindowsCustodyIntent:
        reasons.append("INVALID_INTENT_TYPE")
    if type(observations) is not WindowsCustodyObservation:
        reasons.append("INVALID_OBSERVATION_TYPE")
    if reasons:
        return WindowsCustodyPreflight("HOLD_INVALID_REQUEST", tuple(reasons), False)
    reasons.extend(_intent_syntax_reasons(intent))
    for field in _MATCH_FIELDS:
        expected = getattr(intent, field)
        observed = getattr(observations, field)
        if type(observed) is not type(expected) or observed != expected:
            reasons.append(f"BINDING_MISMATCH_{field.upper()}")
    # In particular bool is an int subclass; integers must not fake a native
    # observation with truthiness (1) in tests or in a later parser.
    for field in _OBSERVATION_CHECKS:
        if getattr(observations, field) is not True:
            reasons.append(f"UNPROVEN_{field.upper()}")
    if reasons:
        return WindowsCustodyPreflight("HOLD_SYNTHETIC_PREFLIGHT", tuple(reasons), False)
    return WindowsCustodyPreflight(
        "SYNTHETIC_CONDITIONS_MET_NOT_AUTHORIZED", (), True,
    )
