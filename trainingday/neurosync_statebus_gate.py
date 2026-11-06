"""Read-only StateBus recovery seam, not an authenticated security service.

Neuro-Sync StateLedger.commit returns existing receipts before checking
current grant callbacks, and can also create new events. This module never
calls commit: an unauthorized or mismatched commit cannot be undone merely
because a wrapper subsequently rejects its receipt.

Only a host broker can authenticate callers, independently verify attestation
and source/artifact pins, and keep raw ledger/SQLite inaccessible to models.
No Python dataclass constructor or callback self-certifies any of that.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,95}$")
_OID = re.compile(r"^(?:sha1:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PAGE_SIZE = 500
_MAX_PAGES = 4


class GovernanceHold(PermissionError):
    """Current host policy/evidence does not support this read."""


class PhysicalTrainingHold(GovernanceHold):
    """No qualified host-owned physical TRAININGDAY launcher."""


def _valid(value: str, pattern: re.Pattern[str], label: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


@dataclass(frozen=True)
class HostPrincipal:
    principal_id: str
    target_id: str
    session_id: str

    def __post_init__(self) -> None:
        for name in ("principal_id", "target_id", "session_id"):
            _valid(getattr(self, name), _ID, name)


@dataclass(frozen=True)
class RepoPins:
    neurosync_git_oid: str
    training_git_oid: str
    neurosync_artifact_sha256: str
    training_artifact_sha256: str

    def __post_init__(self) -> None:
        for name in ("neurosync_git_oid", "training_git_oid"):
            _valid(getattr(self, name), _OID, name)
        for name in ("neurosync_artifact_sha256", "training_artifact_sha256"):
            _valid(getattr(self, name), _SHA, name)


@dataclass(frozen=True)
class CrossRepoEvidence:
    """Declared evidence, not proof absent independent host verification."""
    event_id: str
    content_hash: str
    pins: RepoPins
    host_test_receipt_sha256: str
    evidence_class: str

    def __post_init__(self) -> None:
        _valid(self.event_id, _ID, "event_id")
        _valid(self.content_hash, _SHA, "content_hash")
        _valid(self.host_test_receipt_sha256, _SHA, "host_test_receipt_sha256")
        if not isinstance(self.pins, RepoPins):
            raise TypeError("RepoPins required")
        if self.evidence_class not in {"DOC", "SOURCE_INSPECTION", "TEST", "BUILD",
                                       "INSTALL", "RUNTIME_PROBE", "FORMAL_VERIFY"}:
            raise ValueError("unsupported evidence class")


@dataclass(frozen=True)
class PolicyCheck:
    principal: HostPrincipal
    operation: str
    event_id: str
    content_hash: str
    host_receipt_sha256: str


@dataclass(frozen=True)
class ReadOnlyReceiptView:
    event_id: str
    content_hash: str
    chain_hash: str
    event_offset: int
    key_revision: int
    already_committed: bool = True


class EventLedger(Protocol):
    def events(self, target_id: str, session_id: str, *,
               after_offset: int, limit: int) -> tuple[Any, ...]: ...


class TrainingDayNeuroSyncGate:
    """Only read previously committed metadata using pinned ledger.events.

    It requires host-owned callbacks and ledger. The host must verify the
    actual `.events()` implementation is read-only: Python cannot enforce that
    an arbitrary user-supplied object has no hidden side effects.
    """

    def __init__(self, *, pinned: RepoPins, ledger: EventLedger,
                 authorize: Callable[[PolicyCheck], bool],
                 verify_attestation: Callable[[CrossRepoEvidence], bool]):
        if not isinstance(pinned, RepoPins):
            raise TypeError("host-owned RepoPins required")
        if not callable(getattr(ledger, "events", None)):
            raise TypeError("host-owned read-only ledger required")
        if not callable(authorize) or not callable(verify_attestation):
            raise TypeError("host-owned verification functions required")
        self._pins = pinned
        self._ledger = ledger
        self._authorize = authorize
        self._verify_attestation = verify_attestation

    def _authorize_current(self, principal: HostPrincipal,
                           evidence: CrossRepoEvidence, stage: str) -> None:
        req = PolicyCheck(principal, f"statebus.receipt-read.{stage}",
                          evidence.event_id, evidence.content_hash,
                          evidence.host_test_receipt_sha256)
        try:
            decision = self._authorize(req)
        except Exception:
            raise GovernanceHold("host authorization unavailable") from None
        if decision is not True:
            raise GovernanceHold("host authorization denied")

    def _attest_current(self, evidence: CrossRepoEvidence) -> None:
        if evidence.pins != self._pins:
            raise GovernanceHold("cross-repo binding mismatch")
        try:
            decision = self._verify_attestation(evidence)
        except Exception:
            raise GovernanceHold("host attestation unavailable") from None
        if decision is not True:
            raise GovernanceHold("host attestation rejected")

    def _read_existing(self, principal: HostPrincipal,
                       evidence: CrossRepoEvidence) -> ReadOnlyReceiptView:
        after = 0
        for _ in range(_MAX_PAGES):
            try:
                batch = self._ledger.events(principal.target_id, principal.session_id,
                                            after_offset=after, limit=_PAGE_SIZE)
            except Exception:
                raise GovernanceHold("host ledger read unavailable") from None
            if type(batch) is not tuple or len(batch) > _PAGE_SIZE:
                raise GovernanceHold("invalid ledger page")
            for entry in batch:
                offset = getattr(entry, "event_offset", None)
                revision = getattr(entry, "key_revision", None)
                chain = getattr(entry, "chain_hash", None)
                delta = getattr(entry, "delta", None)
                if (type(offset) is not int or offset <= after or
                        type(revision) is not int or revision < 1 or
                        type(chain) is not str or _SHA.fullmatch(chain) is None or
                        getattr(delta, "target_id", None) != principal.target_id or
                        getattr(delta, "session_id", None) != principal.session_id):
                    raise GovernanceHold("malformed ledger event")
                after = offset
                if getattr(delta, "event_id", None) == evidence.event_id:
                    if getattr(delta, "content_hash", None) != evidence.content_hash:
                        raise GovernanceHold("ledger evidence mismatch")
                    return ReadOnlyReceiptView(evidence.event_id, evidence.content_hash,
                                               chain, offset, revision)
            if len(batch) < _PAGE_SIZE:
                break
        raise GovernanceHold("committed event unavailable in bounded scan")

    def recover_committed_receipt(self, *, principal: HostPrincipal,
                                  evidence: CrossRepoEvidence) -> ReadOnlyReceiptView:
        """Never writes: reads existing committed metadata only."""
        if not isinstance(principal, HostPrincipal) or not isinstance(evidence, CrossRepoEvidence):
            raise GovernanceHold("trusted host context and evidence required")
        self._authorize_current(principal, evidence, "pre")
        self._attest_current(evidence)
        view = self._read_existing(principal, evidence)
        self._attest_current(evidence)
        self._authorize_current(principal, evidence, "post")
        return view

    @staticmethod
    def require_physical_training_authority(operation: str = "launch") -> None:
        raise PhysicalTrainingHold(
            f"TRAININGDAY physical {operation} HARD HOLD: "
            "independent host-owned launcher not installed")
