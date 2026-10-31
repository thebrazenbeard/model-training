"""Source-only model of an independently retained monotonic checkpoint witness.

This is deliberately **not** an authorization service.  In particular, the
in-memory witness below is not durable, is not process-isolated, has no
authenticated caller, and cannot protect against a privileged adversary.  A
real deployment needs a host-owned, atomic, antirollback authority outside the
checkpoint/SQLite restore boundary, with authenticated caller and receipt
validation.  No function here authorizes checkpoint adoption, resume, or GPU
training.

The exercise is useful because it makes the CAS transition explicit: restoring
old application state cannot advance a witness already at a later epoch.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Protocol


GENESIS_HEAD = "0" * 64
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}\Z")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed checkpoint transition; every field is *untrusted* input."""

    run_id: str
    host_principal: str
    step: int
    checkpoint_sha256: str
    nonce: str
    expected_epoch: int
    expected_head: str


@dataclass(frozen=True, slots=True)
class WitnessRecord:
    run_id: str
    host_principal: str
    step: int
    checkpoint_sha256: str
    nonce: str
    epoch: int
    previous_head: str
    head: str


@dataclass(frozen=True, slots=True)
class Decision:
    reason: str
    witness_epoch: int | None = None
    witness_head: str | None = None
    synthetic_transition_observed: bool = False
    authorizes_resume: bool = False
    authorizes_training: bool = False
    physical_custody_verified: bool = False
    durable_antirollback_verified: bool = False


class WitnessPort(Protocol):
    """Required *privileged* interface; implementations need separate review.

    Both methods must refer to the same independently retained, monotonic
    authority.  The compare-and-append operation MUST be atomic and bound to
    authenticated caller identity; Protocol typing does not enforce either.
    """

    def compare_and_append(self, candidate: Candidate) -> WitnessRecord: ...


class WitnessConflict(RuntimeError):
    """A rejected transition.  `reason` is a stable synthetic test code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _is_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _validate(candidate: object) -> Candidate:
    if type(candidate) is not Candidate:
        raise WitnessConflict("MALFORMED_CANDIDATE")
    if not all(_is_id(value) for value in (
        candidate.run_id, candidate.host_principal, candidate.nonce,
    )):
        raise WitnessConflict("MALFORMED_IDENTIFIER")
    if not _is_sha(candidate.checkpoint_sha256) or not _is_sha(candidate.expected_head):
        raise WitnessConflict("MALFORMED_HASH")
    if (type(candidate.step) is not int or not 1 <= candidate.step <= 2**31 - 1
            or type(candidate.expected_epoch) is not int
            or not 0 <= candidate.expected_epoch <= 2**63 - 2):
        raise WitnessConflict("MALFORMED_COUNTER")
    return candidate


def _head_from_record(record: WitnessRecord) -> str:
    data = {
        "domain": "TRAININGDAY_DEMONSTRATION_WITNESS_V1",
        "run_id": record.run_id,
        "host_principal": record.host_principal,
        "step": record.step,
        "checkpoint_sha256": record.checkpoint_sha256,
        "nonce": record.nonce,
        "epoch": record.epoch,
        "previous_head": record.previous_head,
    }
    return hashlib.sha256(json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


class DemoVolatileWitness:
    """Single-process synthetic stand-in; NEVER production trust or durability."""

    def __init__(self) -> None:
        self._records: dict[str, WitnessRecord] = {}
        self._nonces: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def compare_and_append(self, candidate: Candidate) -> WitnessRecord:
        candidate = _validate(candidate)
        with self._lock:
            old = self._records.get(candidate.run_id)
            epoch = old.epoch if old else 0
            head = old.head if old else GENESIS_HEAD
            if (candidate.run_id, candidate.nonce) in self._nonces:
                raise WitnessConflict("NONCE_REPLAY")
            if old is not None and old.host_principal != candidate.host_principal:
                raise WitnessConflict("HOST_PRINCIPAL_CHANGED")
            if candidate.expected_epoch != epoch or candidate.expected_head != head:
                raise WitnessConflict("STALE_OR_DIVERGENT_SNAPSHOT")
            if old is not None and candidate.step <= old.step:
                raise WitnessConflict("STEP_NOT_MONOTONIC")
            record = WitnessRecord(
                run_id=candidate.run_id,
                host_principal=candidate.host_principal,
                step=candidate.step,
                checkpoint_sha256=candidate.checkpoint_sha256,
                nonce=candidate.nonce,
                epoch=epoch + 1,
                previous_head=head,
                head="",
            )
            record = WitnessRecord(
                run_id=record.run_id, host_principal=record.host_principal,
                step=record.step, checkpoint_sha256=record.checkpoint_sha256,
                nonce=record.nonce, epoch=record.epoch,
                previous_head=record.previous_head, head=_head_from_record(record),
            )
            self._records[candidate.run_id] = record
            self._nonces.add((candidate.run_id, candidate.nonce))
            return record

    def demo_read(self, run_id: str) -> WitnessRecord | None:
        """Convenience for synthetic tests; cannot attest external custody."""
        with self._lock:
            return self._records.get(run_id)


def assess_transition(candidate: object, witness: WitnessPort | None) -> Decision:
    """Attempt one bounded source-only CAS; any uncertainty is a HOLD.

    Caller-provided witness objects are not authenticated.  The returned
    `synthetic_transition_observed` is *not* a production trust assertion.
    A timeout after commit remains UNKNOWN; do not blindly retry it.
    """
    try:
        request = _validate(candidate)
    except WitnessConflict as exc:
        return Decision(reason=exc.reason)
    if witness is None or not callable(getattr(witness, "compare_and_append", None)):
        return Decision(reason="INDEPENDENT_WITNESS_UNAVAILABLE")
    try:
        reply = witness.compare_and_append(request)
    except WitnessConflict as exc:
        return Decision(reason=exc.reason)
    except Exception:
        return Decision(reason="WITNESS_OUTCOME_UNKNOWN_HOLD")
    if (type(reply) is not WitnessRecord
            or reply.run_id != request.run_id
            or reply.host_principal != request.host_principal
            or reply.step != request.step
            or reply.checkpoint_sha256 != request.checkpoint_sha256
            or reply.nonce != request.nonce
            or reply.epoch != request.expected_epoch + 1
            or reply.previous_head != request.expected_head
            or not _is_sha(reply.head)
            or reply.head != _head_from_record(reply)):
        return Decision(reason="WITNESS_REPLY_UNBOUND")
    return Decision(
        reason="SYNTHETIC_CAS_MATCH_NO_HOST_AUTHORITY",
        witness_epoch=reply.epoch,
        witness_head=reply.head,
        synthetic_transition_observed=True,
    )
