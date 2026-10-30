"""R25 deterministic *synthetic* workload budget and lease-shadow policy.

The caller provides declared observation records; this module never reads
hardware, authenticates a device, reserves a GPU, or starts an optimizer.
Even a clean internal result is not an authorized run or neural evidence.

Units are explicit: elapsed seconds, UTC Unix seconds, GPU bytes, Celsius,
watt-hours, and US dollars. A power-state interval is NEVER retroactively
subtracted from elapsed time to make an invalid comparison valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


SCHEMA = "VERA_TRAININGDAY_R25_BUDGET_POLICY_SHADOW_V1"
MAX_SAMPLES = 4096
MAX_HEARTBEAT_AGE_SECONDS = 10.0
MAX_SAMPLE_GAP_SECONDS = 30.0
MAX_CLOCK_DIVERGENCE_SECONDS = 5.0
MAX_LEASE_AGE_SECONDS = 30.0
MAX_LEASE_HORIZON_SECONDS = 120.0
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _id(value: object) -> bool:
    return type(value) is str and _OPAQUE_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class BudgetLimits:
    active_seconds: float
    wall_seconds: float
    energy_wh: float
    cost_usd: float
    thermal_c: float
    peak_gpu_bytes: int
    required_free_gpu_bytes: int
    gpu_reserve_bytes: int
    heartbeat_max_age_seconds: float
    sample_max_gap_seconds: float
    clock_divergence_seconds: float
    lease_max_age_seconds: float
    lease_max_remaining_seconds: float


@dataclass(frozen=True)
class LeaseObservation:
    device_id: str
    resource_id: str
    owner_id: str
    lease_id: str
    observed_unix: float
    expires_unix: float
    active_lease_ids: tuple[str, ...]


@dataclass(frozen=True)
class BudgetObservation:
    monotonic_seconds: float
    wall_unix: float
    supervisor_heartbeat_unix: float
    optimizer_active_seconds: float
    peak_gpu_bytes: int
    free_gpu_bytes: int
    temperature_c: float
    energy_wh: float
    cost_usd: float
    connected: bool
    in_standby: bool


def _number(value: object, *, strictly_positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if strictly_positive else value >= 0)
    except OverflowError:
        return False


def _bytes(value: object) -> bool:
    return type(value) is int and 0 <= value <= 2**63 - 1


def _policy_valid(policy: object) -> bool:
    if not isinstance(policy, BudgetLimits):
        return False
    values = (
        policy.active_seconds, policy.wall_seconds, policy.energy_wh,
        policy.cost_usd, policy.thermal_c, policy.heartbeat_max_age_seconds,
        policy.sample_max_gap_seconds, policy.clock_divergence_seconds,
        policy.lease_max_age_seconds, policy.lease_max_remaining_seconds,
    )
    return (all(_number(v, strictly_positive=True) for v in values)
            and all(_bytes(v) for v in (
                policy.peak_gpu_bytes, policy.required_free_gpu_bytes,
                policy.gpu_reserve_bytes))
            and policy.peak_gpu_bytes > 0 and policy.required_free_gpu_bytes > 0
            and policy.gpu_reserve_bytes > 0
            and policy.heartbeat_max_age_seconds <= MAX_HEARTBEAT_AGE_SECONDS
            and policy.sample_max_gap_seconds <= MAX_SAMPLE_GAP_SECONDS
            and policy.clock_divergence_seconds <= MAX_CLOCK_DIVERGENCE_SECONDS
            and policy.lease_max_age_seconds <= MAX_LEASE_AGE_SECONDS
            and policy.lease_max_remaining_seconds <= MAX_LEASE_HORIZON_SECONDS)


def _lease_valid(lease: object) -> bool:
    return (
        isinstance(lease, LeaseObservation)
        and all(_id(v) for v in (
            lease.device_id, lease.resource_id, lease.owner_id, lease.lease_id))
        and _number(lease.observed_unix)
        and _number(lease.expires_unix)
        and type(lease.active_lease_ids) is tuple
        and len(lease.active_lease_ids) <= 64
        and all(_id(v) for v in lease.active_lease_ids)
    )


def _sample_valid(sample: object) -> bool:
    return (
        isinstance(sample, BudgetObservation)
        and all(_number(v) for v in (
            sample.monotonic_seconds, sample.wall_unix,
            sample.supervisor_heartbeat_unix, sample.optimizer_active_seconds,
            sample.temperature_c, sample.energy_wh, sample.cost_usd))
        and _bytes(sample.peak_gpu_bytes) and _bytes(sample.free_gpu_bytes)
        and type(sample.connected) is bool and type(sample.in_standby) is bool
    )


def assess_budget_run(*, policy: BudgetLimits, lease: LeaseObservation,
                      lease_renewals: tuple[LeaseObservation, ...],
                      observations: tuple[BudgetObservation, ...],
                      expected_owner: str, completed: bool,
                      run_mode: str) -> dict:
    """Evaluate fake-clock receipts but never authorize physical execution.

    A clean declaration is only locally consistent. Neither asserted
    connection, lease IDs, heartbeat timestamps, nor cost readings are
    independently signed by a host authority.
    """
    blockers: list[str] = []

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    result = {
        "schema": SCHEMA,
        "status": "HOLD",
        "evidence_class": "SYNTHETIC_POLICY_DECLARATIONS_ONLY",
        "blockers": blockers,
        "synthetic_constraints_satisfied": False,
        "observed_samples": 0,
        "declared_run_mode": None,
        "authorizes_gpu_start": False,
        "host_lease_authenticated": False,
        "valid_equal_time_neural_evidence": False,
        "authorizes_model_promotion": False,
    }

    if not _policy_valid(policy):
        block("MALFORMED_POLICY")
        return result
    if not _lease_valid(lease) or not _id(expected_owner):
        block("MALFORMED_LEASE")
        return result
    if (type(lease_renewals) is not tuple
            or len(lease_renewals) > MAX_SAMPLES - 1
            or any(not _lease_valid(view) for view in lease_renewals)):
        block("MALFORMED_LEASE_RENEWAL_CHAIN")
        return result
    if type(completed) is not bool:
        block("MALFORMED_COMPLETION_CLAIM")
        return result
    if type(run_mode) is not str or run_mode not in ("TRAINING", "ZERO_UPDATE_CONTROL"):
        block("MALFORMED_RUN_MODE")
        return result
    result["declared_run_mode"] = run_mode
    if not isinstance(observations, tuple) or not 2 <= len(observations) <= MAX_SAMPLES:
        block("MISSING_WITNESS_SAMPLES")
        return result
    if any(not _sample_valid(sample) for sample in observations):
        block("MALFORMED_OBSERVATION")
        return result
    result["observed_samples"] = len(observations)

    first, final = observations[0], observations[-1]
    # The chain is a chronology of untrusted *declarations*, never a real
    # atomic owner authority. Each renewal must precede predecessor expiry,
    # preserve the exact owner/resource/lease identity, and cover the window.
    lease_chain = (lease,) + lease_renewals
    for index, current in enumerate(lease_chain):
        if current.device_id != "Lappy" or current.resource_id != "CUDA0":
            block("RESOURCE_IDENTITY_UNVERIFIED")
        if current.owner_id != expected_owner:
            block("LEASE_OWNER_MISMATCH" if index == 0
                  else "LEASE_OWNER_CHANGE_UNAUTHORIZED")
        if index and current.lease_id != lease.lease_id:
            block("LEASE_ID_CHANGE_UNAUTHORIZED")
        if current.active_lease_ids != (current.lease_id,):
            block("LEASE_COLLISION")
        if current.expires_unix <= current.observed_unix:
            block("LEASE_EXPIRED")
        if (current.expires_unix - current.observed_unix
                > policy.lease_max_remaining_seconds):
            block("LEASE_HORIZON_TOO_LONG")
        if index == 0:
            if (current.observed_unix > first.wall_unix
                    or first.wall_unix - current.observed_unix
                    > policy.lease_max_age_seconds):
                block("LEASE_STALE")
        else:
            previous = lease_chain[index - 1]
            if current.observed_unix <= previous.observed_unix:
                block("NONMONOTONIC_LEASE_OBSERVATION")
            if (current.observed_unix - previous.observed_unix
                    > policy.lease_max_age_seconds):
                block("LEASE_RENEWAL_GAP_UNWITNESSED")
            if previous.expires_unix <= current.observed_unix:
                block("LEASE_EXPIRED_DURING_RUN")
            if current.observed_unix > final.wall_unix:
                block("LEASE_OBSERVATION_AFTER_RUN")
    last_lease = lease_chain[-1]
    if final.wall_unix - last_lease.observed_unix > policy.lease_max_age_seconds:
        block("LEASE_STALE")
    if last_lease.expires_unix <= final.wall_unix:
        block("LEASE_EXPIRED")
    if not completed:
        block("INCOMPLETE_RUN")

    active_progress = final.optimizer_active_seconds - first.optimizer_active_seconds
    if completed and run_mode == "TRAINING" and active_progress <= 0:
        block("NO_TRAINING_PROGRESS")
    if run_mode == "ZERO_UPDATE_CONTROL" and active_progress != 0:
        block("ZERO_CONTROL_CONTAINS_OPTIMIZER_TIME")

    if final.wall_unix - first.wall_unix > policy.wall_seconds:
        block("WALL_BUDGET_EXCEEDED")
    if final.optimizer_active_seconds > policy.active_seconds:
        block("ACTIVE_BUDGET_EXCEEDED")
    if final.energy_wh > policy.energy_wh:
        block("ENERGY_BUDGET_EXCEEDED")
    if final.cost_usd > policy.cost_usd:
        block("COST_BUDGET_EXCEEDED")

    for index, sample in enumerate(observations):
        if sample.in_standby:
            block("POWER_STANDBY_OBSERVED")
        if not sample.connected:
            block("HOST_DISCONNECTED")
        if (sample.supervisor_heartbeat_unix > sample.wall_unix
                or sample.wall_unix - sample.supervisor_heartbeat_unix
                > policy.heartbeat_max_age_seconds):
            block("SUPERVISOR_HEARTBEAT_STALE")
        if sample.temperature_c > policy.thermal_c:
            block("THERMAL_LIMIT_EXCEEDED")
        if sample.peak_gpu_bytes > policy.peak_gpu_bytes:
            block("PEAK_MEMORY_LIMIT_EXCEEDED")
        if sample.free_gpu_bytes < (policy.required_free_gpu_bytes
                                    + policy.gpu_reserve_bytes):
            block("GPU_HEADROOM_UNVERIFIED")
        if index == 0:
            continue
        previous = observations[index - 1]
        mono_delta = sample.monotonic_seconds - previous.monotonic_seconds
        wall_delta = sample.wall_unix - previous.wall_unix
        if (mono_delta < 0 or wall_delta < 0
                or sample.optimizer_active_seconds < previous.optimizer_active_seconds
                or sample.energy_wh < previous.energy_wh
                or sample.cost_usd < previous.cost_usd
                or sample.peak_gpu_bytes < previous.peak_gpu_bytes):
            block("NONMONOTONIC_RECEIPT")
        if (mono_delta > policy.sample_max_gap_seconds
                or wall_delta > policy.sample_max_gap_seconds):
            block("OBSERVATION_GAP_UNWITNESSED")
        if abs(mono_delta - wall_delta) > policy.clock_divergence_seconds:
            block("CLOCK_DIVERGENCE_SUSPECTED")
        if (abs((sample.monotonic_seconds - first.monotonic_seconds)
                - (sample.wall_unix - first.wall_unix))
                > policy.clock_divergence_seconds):
            block("CUMULATIVE_CLOCK_DIVERGENCE_SUSPECTED")
        # An optimizer cannot accumulate more active seconds than the same
        # process's elapsed monotonic interval. This is logical consistency,
        # not a physically witnessed clock or standby guard.
        if (sample.optimizer_active_seconds - first.optimizer_active_seconds
                > sample.monotonic_seconds - first.monotonic_seconds + 1e-6):
            block("IMPOSSIBLE_ACTIVE_TIME_ACCOUNTING")

    result["synthetic_constraints_satisfied"] = not blockers
    return result
