"""R140 host checkpoint-custody *interface*, not an authority implementation.

The caller-facing request is deliberately untrusted. A privileged, separately
deployed host-owned authority MUST authenticate the actual caller, validate
the lease against that caller and host, pin/attest the checkpoint using kernel
object handles, and persist a monotonic non-rollback commitment. Merely
injecting a Python object implementing this Protocol is not enforcement.

No host authority is bundled with this source-only module. Successful synthetic
tests establish request/response correlation only: they do not prove a service
is deployed, files remain immutable, power-loss durability, physical reload,
or permission to train/resume. A failed or ambiguous commit is never retried.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}\Z")
_MAX_STEP = 2**31 - 1
# Host authority counters require a bounded interoperable integer domain.
# Leave the largest signed 64-bit value available as a future sentinel.
_MAX_COUNTER = 2**63 - 2


@dataclass(frozen=True)
class CustodyRequest:
    """Untrusted proposal; identity fields are claims until the host binds them."""

    request_id: str
    caller_principal: str
    host_id: str
    lease_id: str
    run_id: str
    run_contract_sha256: str
    source_commit_sha256: str
    checkpoint_weights_sha256: str
    checkpoint_state_sha256: str
    step: int
    checkpoint_object_id: str
    authority_epoch: int

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            vars(self), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class HostReservation:
    """Backend response, valid only when minted/held by privileged authority."""

    request_sha256: str
    reservation_id: str
    caller_principal: str
    host_id: str
    lease_id: str
    authority_epoch: int
    expires_at: int


@dataclass(frozen=True)
class HostSnapshot:
    """Host assertion: snapshot identity must be pinned in the host process."""

    request_sha256: str
    reservation_id: str
    source_commit_sha256: str
    checkpoint_weights_sha256: str
    checkpoint_state_sha256: str
    object_id: str
    step: int
    descriptor_bound: bool

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            vars(self), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class HostCommit:
    """Backend claim of a durable commit; no source-level durability proof."""

    request_sha256: str
    reservation_id: str
    snapshot_sha256: str
    receipt_id: str
    authority_epoch: int
    monotonic_sequence: int
    state: str


@dataclass(frozen=True)
class CustodyResult:
    """A checked contract response, never a train/resume capability."""

    request_sha256: str
    receipt_id: str
    monotonic_sequence: int
    status: str = "SOURCE_ONLY_HOST_AUTHORITY_UNVERIFIED"
    authorizes_training: bool = False
    authorizes_resume: bool = False
    physical_durability_verified: bool = False
    physical_reload_verified: bool = False


class CustodyServiceError(RuntimeError):
    def __init__(self, reason: str, *, ambiguous_commit: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.ambiguous_commit = ambiguous_commit


class HostOwnedBackend(Protocol):
    """Host-protected implementation boundary; not satisfied by a client mock.

    ``reserve`` MUST independently attest OS caller identity, lease, host,
    operation, and request ID; reject replay and perform an atomic reservation.
    ``host_time`` MUST read trusted host time not accepted from the caller.
    ``snapshot`` MUST use stable kernel handles/descriptors and forbid hostile
    replacement, not repeated pathname checks. ``commit`` MUST persist a
    monotonic, externally anchored record immune to snapshot rollback.
    The backend MUST recheck lease expiry and host ownership atomically at
    ``commit``; the service's second host-time sample cannot close that race.
    ``abandon`` MUST preserve forensic UNKNOWN, never release ambiguity into
    an auto-replay. Host-specific implementations need independent validation.
    """

    def host_time(self) -> int: ...

    def reserve(self, request: CustodyRequest, request_sha256: str) -> HostReservation: ...

    def snapshot(self, request: CustodyRequest, reservation: HostReservation) -> HostSnapshot: ...

    def commit(self, request: CustodyRequest, reservation: HostReservation,
               snapshot: HostSnapshot) -> HostCommit: ...

    def abandon(self, reservation: HostReservation, reason: str) -> None: ...


def _valid_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _valid_hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _valid_counter(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_COUNTER


def _validate_request(request: CustodyRequest) -> None:
    if type(request) is not CustodyRequest:
        raise CustodyServiceError("MALFORMED_REQUEST")
    if not all(_valid_id(x) for x in (
        request.request_id, request.caller_principal, request.host_id,
        request.lease_id, request.run_id, request.checkpoint_object_id,
    )):
        raise CustodyServiceError("MALFORMED_REQUEST_IDENTIFIER")
    if not all(_valid_hash(x) for x in (
        request.run_contract_sha256, request.source_commit_sha256,
        request.checkpoint_weights_sha256, request.checkpoint_state_sha256,
    )):
        raise CustodyServiceError("MALFORMED_REQUEST_DIGEST")
    if not (type(request.step) is int and 1 <= request.step <= _MAX_STEP
            and _valid_counter(request.authority_epoch)):
        raise CustodyServiceError("MALFORMED_REQUEST_COUNTER")


def _assert_reservation(
    request: CustodyRequest, reservation: HostReservation, sha: str, now: int,
) -> None:
    if (type(reservation) is not HostReservation
            or not _valid_id(reservation.reservation_id)
            or reservation.request_sha256 != sha
            or reservation.caller_principal != request.caller_principal
            or reservation.host_id != request.host_id
            or reservation.lease_id != request.lease_id
            or reservation.authority_epoch != request.authority_epoch
            or type(reservation.expires_at) is not int
            or reservation.expires_at <= now):
        raise CustodyServiceError("RESERVATION_BINDING_MISMATCH")


def _assert_snapshot(
    request: CustodyRequest, reservation: HostReservation,
    snapshot: HostSnapshot, sha: str,
) -> None:
    if (type(snapshot) is not HostSnapshot
            or snapshot.request_sha256 != sha
            or snapshot.reservation_id != reservation.reservation_id
            or snapshot.source_commit_sha256 != request.source_commit_sha256
            or snapshot.checkpoint_weights_sha256 != request.checkpoint_weights_sha256
            or snapshot.checkpoint_state_sha256 != request.checkpoint_state_sha256
            or snapshot.object_id != request.checkpoint_object_id
            or type(snapshot.step) is not int or snapshot.step != request.step
            or snapshot.descriptor_bound is not True):
        raise CustodyServiceError("SNAPSHOT_BINDING_MISMATCH")


def correlate_contract(
    request: CustodyRequest,
    reservation: HostReservation,
    snapshot: HostSnapshot,
    commit: HostCommit,
    *, now: int,
) -> CustodyResult:
    """Check responses from the purported authority, not their authenticity."""
    _validate_request(request)
    if type(now) is not int or now < 0:
        raise CustodyServiceError("MALFORMED_CLOCK")
    sha = request.digest()
    _assert_reservation(request, reservation, sha, now)
    _assert_snapshot(request, reservation, snapshot, sha)
    if (type(commit) is not HostCommit
            or commit.request_sha256 != sha
            or commit.reservation_id != reservation.reservation_id
            or commit.snapshot_sha256 != snapshot.digest()
            or not _valid_id(commit.receipt_id)
            or commit.authority_epoch != request.authority_epoch
            or not _valid_counter(commit.monotonic_sequence)
            or commit.monotonic_sequence < 1
            or commit.state != "HOST_BACKEND_COMMITTED_UNVERIFIED"):
        raise CustodyServiceError("COMMIT_BINDING_MISMATCH")
    return CustodyResult(
        request_sha256=sha, receipt_id=commit.receipt_id,
        monotonic_sequence=commit.monotonic_sequence,
    )


class HostCustodyService:
    """Fail-closed facade for a *separately deployed*, privileged authority.

    This Python class cannot establish that its backend is host-owned. Deploy
    only inside a host-administered boundary whose provider identity cannot
    be swapped by a caller. It does not verify an external monotonic witness.
    All host trust proofs remain absent here.
    """

    def __init__(self, backend: HostOwnedBackend) -> None:
        self._backend = backend

    def _host_now(self) -> int:
        try:
            now = self._backend.host_time()
        except Exception as exc:
            raise CustodyServiceError("AUTHORITY_CLOCK_UNAVAILABLE") from exc
        if type(now) is not int or now < 0:
            raise CustodyServiceError("AUTHORITY_CLOCK_UNAVAILABLE")
        return now

    def execute(self, request: CustodyRequest) -> CustodyResult:
        _validate_request(request)
        now = self._host_now()
        sha = request.digest()
        reservation: HostReservation | None = None
        try:
            reservation = self._backend.reserve(request, sha)
        except Exception as exc:
            raise CustodyServiceError("RESERVATION_REJECTED") from exc
        # Never submit the rest of an operation to an authority that returned
        # an inconsistent reservation. Any failure quarantines this attempt.
        try:
            _assert_reservation(request, reservation, sha, now)
        except CustodyServiceError:
            # Do NOT pass an unverified/misbinding reservation to abandon:
            # it could designate another tenant's live reservation. The host
            # backend is responsible for reconciling its malformed response.
            raise CustodyServiceError("RESERVATION_BINDING_MISMATCH")
        try:
            snapshot = self._backend.snapshot(request, reservation)
            # Preflight consistency before invoking irreversible commit.
            _assert_snapshot(request, reservation, snapshot, sha)
            # A slow snapshot must not inherit a stale earlier lease window.
            _assert_reservation(request, reservation, sha, self._host_now())
        except Exception as exc:
            self._best_effort_abandon(reservation, "SNAPSHOT_REJECTED")
            raise CustodyServiceError("SNAPSHOT_REJECTED") from exc
        try:
            committed = self._backend.commit(request, reservation, snapshot)
        except Exception as exc:
            # An exception here may have happened AFTER irreversible commit.
            # Never issue a second call with the same request automatically.
            self._best_effort_abandon(reservation, "COMMIT_OUTCOME_UNKNOWN")
            raise CustodyServiceError(
                "COMMIT_OUTCOME_UNKNOWN", ambiguous_commit=True,
            ) from exc
        try:
            return correlate_contract(request, reservation, snapshot, committed, now=now)
        except CustodyServiceError as exc:
            self._best_effort_abandon(reservation, "COMMIT_RECEIPT_MISMATCH")
            raise CustodyServiceError(
                "COMMIT_RECEIPT_MISMATCH", ambiguous_commit=True,
            ) from exc

    def _best_effort_abandon(self, reservation: object, reason: str) -> None:
        if type(reservation) is not HostReservation:
            return
        try:
            self._backend.abandon(reservation, reason)
        except Exception:
            # Error/ambiguity remains a HOLD, never an implicit retry grant.
            pass
