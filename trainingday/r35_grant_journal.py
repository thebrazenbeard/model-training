"""In-memory, synthetic one-time grant journal for hostile protocol tests.

This is **not** a bank custodian, a durable transaction manager, an audit
service, or a source of authority.  In particular, a callback returning True
is only a test observation: it cannot establish that an external audit write
was durable.  Never wire this module to a real sealed evaluation bank.

The narrow proved property is at-most-once grant consumption *inside one
live Python object*.  Uncertain audit outcomes, suspected crashes, and any
untrusted snapshot import permanently hold affected entries closed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock
from typing import Callable


class GrantHold(RuntimeError):
    """No synthetic grant transition may be treated as authorized."""


class AuditIndeterminate(GrantHold):
    """The audit callback could have written before reporting failure."""


class RollbackSuspected(GrantHold):
    """A snapshot revision is below a caller's diagnostic minimum."""


class GrantStatus(str, Enum):
    PREPARED = "PREPARED"
    IN_FLIGHT = "IN_FLIGHT"
    CONSUMED = "CONSUMED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


@dataclass(frozen=True)
class GrantRecord:
    grant_id: str
    nonce: str
    status: GrantStatus
    revision: int


@dataclass(frozen=True)
class AuditEvent:
    kind: str
    grant_id: str
    nonce: str
    revision: int


@dataclass(frozen=True)
class JournalSnapshot:
    revision: int
    records: tuple[GrantRecord, ...]


class SyntheticGrantJournal:
    """Thread-safe volatile test state; no external trust or persistence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._revision = 0
        self._records: dict[str, GrantRecord] = {}
        self._nonces: set[str] = set()
        self._blocked = False
        self._in_audit = False
        self._events: list[AuditEvent] = []

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if type(value) is not str or not value or value.strip() != value:
            raise GrantHold(f"invalid synthetic {name}")

    def _require_open(self) -> None:
        if self._blocked:
            raise GrantHold("journal requires independent reconciliation")
        if self._in_audit:
            raise GrantHold("reentrant operation during synthetic audit")

    def prepare(self, grant_id: str, nonce: str) -> GrantRecord:
        self._validate_identifier(grant_id, "grant id")
        self._validate_identifier(nonce, "nonce")
        with self._lock:
            self._require_open()
            if grant_id in self._records or nonce in self._nonces:
                raise GrantHold("grant id or one-time nonce already reserved")
            self._revision += 1
            record = GrantRecord(grant_id, nonce, GrantStatus.PREPARED,
                                 self._revision)
            self._records[grant_id] = record
            self._nonces.add(nonce)
            self._events.append(AuditEvent("PREPARE_SYNTHETIC", grant_id,
                                           nonce, self._revision))
            # Frozen dataclasses are not a security boundary: callers can use
            # object.__setattr__. Never expose the internal record itself.
            return replace(record)

    def lookup(self, grant_id: str) -> GrantRecord:
        self._validate_identifier(grant_id, "grant id")
        with self._lock:
            try:
                return replace(self._records[grant_id])
            except (KeyError, TypeError) as exc:
                raise GrantHold("synthetic grant is absent") from exc

    def _quarantine(self, record: GrantRecord) -> None:
        self._revision += 1
        self._records[record.grant_id] = replace(
            record, status=GrantStatus.RECONCILIATION_REQUIRED,
            revision=self._revision)
        self._events.append(AuditEvent("AUDIT_OUTCOME_UNKNOWN", record.grant_id,
                                       record.nonce, self._revision))

    def consume(self, grant_id: str, nonce: str, expected_revision: int,
                audit_write: Callable[[AuditEvent], bool]) -> GrantRecord:
        """Simulate at-most-once consumption; never unseal a real bank.

        The audit callback is an injected test seam, not an authenticated or
        transactionally coupled ledger.  Failure or ambiguous return makes
        the grant unretryable until an independently implemented reconciler
        (intentionally absent) establishes what occurred.
        """
        # Validate before dict lookup and equality. A caller-provided object
        # can otherwise impersonate a string key via __hash__/__eq__, or a
        # nonce via __ne__ returning False.
        self._validate_identifier(grant_id, "grant id")
        self._validate_identifier(nonce, "nonce")
        with self._lock:
            self._require_open()
            record = self.lookup(grant_id)
            if (record.status is not GrantStatus.PREPARED
                    or nonce != record.nonce
                    or type(expected_revision) is not int
                    or expected_revision != record.revision):
                raise GrantHold("synthetic grant does not match prepared revision")
            if not callable(audit_write):
                raise GrantHold("synthetic audit callback missing")

            self._revision += 1
            in_flight = replace(record, status=GrantStatus.IN_FLIGHT,
                                revision=self._revision)
            self._records[grant_id] = in_flight
            event = AuditEvent("CONSUME_SYNTHETIC", grant_id, nonce,
                               self._revision + 1)
            self._in_audit = True
            try:
                acknowledged = audit_write(event)
            except BaseException as exc:
                self._quarantine(in_flight)
                raise AuditIndeterminate(
                    "synthetic audit callback outcome unknown") from exc
            finally:
                self._in_audit = False
            if self._blocked or self._records.get(grant_id) != in_flight:
                # An audit callback can observe/report a crash or otherwise
                # seal the journal. Its later True return cannot undo that
                # more restrictive state.
                raise AuditIndeterminate(
                    "synthetic audit completed after crash or state change")
            if acknowledged is not True:
                self._quarantine(in_flight)
                raise AuditIndeterminate("synthetic audit was not acknowledged")

            self._revision += 1
            consumed = replace(in_flight, status=GrantStatus.CONSUMED,
                               revision=self._revision)
            self._records[grant_id] = consumed
            # The callback may retain or mutate its AuditEvent reference.
            # Rebuild the journal event from validated local primitives.
            self._events.append(AuditEvent("CONSUME_SYNTHETIC", grant_id,
                                           nonce, self._revision))
            return replace(consumed)

    def mark_crash_suspected(self) -> None:
        """Seal everything after ambiguous process-state or transport loss."""
        with self._lock:
            self._blocked = True
            self._revision += 1
            for grant_id, record in self._records.items():
                if record.status in (GrantStatus.PREPARED, GrantStatus.IN_FLIGHT):
                    self._records[grant_id] = replace(
                        record, status=GrantStatus.RECONCILIATION_REQUIRED,
                        revision=self._revision)
            self._events.append(AuditEvent("CRASH_SUSPECTED_SYNTHETIC", "", "",
                                           self._revision))

    def snapshot(self) -> JournalSnapshot:
        """Untrusted diagnostic snapshot. It carries no authority to resume."""
        with self._lock:
            return JournalSnapshot(self._revision, tuple(
                replace(self._records[key]) for key in sorted(self._records)))

    @classmethod
    def import_untrusted_snapshot(
        cls, snapshot: JournalSnapshot, *, asserted_min_revision: int | None = None
    ) -> SyntheticGrantJournal:
        """Never reactivate grants from caller-supplied state.

        ``asserted_min_revision`` is a *diagnostic* comparison only. A real
        service would require an independent, anti-rollback witness; none is
        available here, so even a matching snapshot stays permanently held.
        """
        if (type(snapshot) is not JournalSnapshot
                or type(snapshot.revision) is not int
                or snapshot.revision < 0
                or type(snapshot.records) is not tuple):
            raise GrantHold("malformed untrusted snapshot")
        if asserted_min_revision is not None:
            if type(asserted_min_revision) is not int or asserted_min_revision < 0:
                raise GrantHold("invalid diagnostic revision floor")
            if snapshot.revision < asserted_min_revision:
                raise RollbackSuspected("untrusted snapshot revision is stale")
        journal = cls()
        with journal._lock:
            journal._revision = max(0, snapshot.revision)
            journal._blocked = True
            for record in snapshot.records:
                if type(record) is not GrantRecord:
                    raise GrantHold("malformed untrusted snapshot record")
                journal._validate_identifier(record.grant_id, "grant id")
                journal._validate_identifier(record.nonce, "nonce")
                if (type(record.status) is not GrantStatus
                        or type(record.revision) is not int
                        or not 0 <= record.revision <= snapshot.revision):
                    raise GrantHold("malformed untrusted snapshot record state")
                if (record.grant_id in journal._records
                        or record.nonce in journal._nonces):
                    raise GrantHold("duplicate grant id or nonce in snapshot")
                journal._records[record.grant_id] = replace(
                    record, status=GrantStatus.RECONCILIATION_REQUIRED)
                journal._nonces.add(record.nonce)
        return journal
