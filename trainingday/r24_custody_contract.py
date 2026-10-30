"""Fail-closed *synthetic* grant-contract assessor for private evaluation custody.

This module deliberately has no private-bank path, I/O, unseal, or launch API.
It can check that fabricated claims are mutually consistent; it cannot verify
OS process identity, control another principal's bank, or authorize bank access.

The supplied verifier and audit acknowledgement are untrusted caller inputs
unless an independent privileged broker implements and enforces them. Even a
successful contract check therefore returns ``bank_access_authorized=False``.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Callable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_MAX_GRANT_LIFETIME_SECONDS = 300


@dataclass(frozen=True)
class GrantRequest:
    run_id: str
    model_sha256: str
    adapter_sha256: str
    corpus_sha256: str
    bank_object_id: str
    evaluator_principal: str


@dataclass(frozen=True)
class GrantEnvelope:
    issuer: str
    subject: str
    nonce: str
    issued_at: int
    expires_at: int
    run_id: str
    model_sha256: str
    adapter_sha256: str
    corpus_sha256: str
    bank_object_id: str
    signature: str

    def canonical_payload(self) -> bytes:
        """Deterministic synthetic bytes, excluding the signature itself."""
        return json.dumps(
            {key: value for key, value in vars(self).items() if key != "signature"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")


@dataclass(frozen=True)
class CustodyContext:
    trusted_issuer: str
    bank_owner_principal: str
    trainer_principal: str
    audit_principal: str
    audit_acked: bool
    raw_bank_path_exposed: bool
    bank_object_identity_stable: bool


@dataclass(frozen=True)
class GrantDecision:
    reason: str
    contract_match: bool = False
    bank_access_authorized: bool = False
    requires_external_broker: bool = True


class SyntheticNonceJournal:
    """Atomic only within this one process; *not* durable cross-host replay custody."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def consume(self, issuer: str, nonce: str) -> bool:
        with self._lock:
            key = (issuer, nonce)
            if key in self._seen:
                return False
            self._seen.add(key)
            return True


def _good_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _good_id(value: object) -> bool:
    return isinstance(value, str) and _OPAQUE_ID.fullmatch(value) is not None


def _good_principal(value: object) -> bool:
    return _good_id(value)


def _reject(reason: str) -> GrantDecision:
    return GrantDecision(reason=reason)


def assess_grant(
    grant: GrantEnvelope,
    request: GrantRequest,
    context: CustodyContext,
    *,
    now: int,
    verify_signature: Callable[[bytes, str, str], bool] | None,
    nonce_journal: SyntheticNonceJournal,
) -> GrantDecision:
    """Check only synthetic grant consistency and force physical-access HOLD.

    A true result means *only* contract fields matched. The callback is not a
    production trust anchor and the journal is not a cross-process audit log.
    """
    if (not isinstance(grant, GrantEnvelope)
            or not isinstance(request, GrantRequest)
            or not isinstance(context, CustodyContext)):
        return _reject("MALFORMED_ENVELOPE")
    if not isinstance(now, int) or isinstance(now, bool) or now < 0:
        return _reject("INVALID_TIME")
    if not isinstance(nonce_journal, SyntheticNonceJournal):
        return _reject("INVALID_NONCE_JOURNAL")
    if not all(_good_hash(v) for v in (
        grant.model_sha256, grant.adapter_sha256, grant.corpus_sha256,
        request.model_sha256, request.adapter_sha256, request.corpus_sha256,
    )):
        return _reject("MALFORMED_HASH")
    if not all(_good_id(v) for v in (
        grant.nonce, grant.run_id, grant.bank_object_id,
        request.run_id, request.bank_object_id,
    )):
        return _reject("MALFORMED_IDENTIFIER")
    if not all(_good_principal(v) for v in (
        grant.issuer, grant.subject, request.evaluator_principal,
        context.trusted_issuer, context.bank_owner_principal,
        context.trainer_principal, context.audit_principal,
    )):
        return _reject("MALFORMED_PRINCIPAL")
    if not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in (
        grant.issued_at, grant.expires_at,
    )):
        return _reject("MALFORMED_TIME")
    if not isinstance(grant.signature, str) or not grant.signature:
        return _reject("SIGNATURE_MISSING")
    if not all(type(v) is bool for v in (
        context.audit_acked, context.raw_bank_path_exposed,
        context.bank_object_identity_stable,
    )):
        return _reject("MALFORMED_CONTEXT")

    principals = (
        context.trusted_issuer, context.bank_owner_principal,
        context.trainer_principal, request.evaluator_principal,
        context.audit_principal,
    )
    if len(principals) != len(set(principals)):
        return _reject("PRINCIPAL_ISOLATION_FAILED")
    if not context.audit_acked:
        return _reject("INDEPENDENT_AUDIT_MISSING")
    if context.raw_bank_path_exposed or not context.bank_object_identity_stable:
        return _reject("BANK_CUSTODY_ISOLATION_FAILED")
    if grant.issuer != context.trusted_issuer:
        return _reject("ISSUER_MISMATCH")
    if grant.subject != request.evaluator_principal:
        return _reject("SUBJECT_MISMATCH")
    if grant.issued_at > now:
        return _reject("NOT_YET_VALID")
    if grant.expires_at <= now:
        return _reject("EXPIRED")
    if grant.expires_at <= grant.issued_at or (
        grant.expires_at - grant.issued_at > _MAX_GRANT_LIFETIME_SECONDS
    ):
        return _reject("GRANT_LIFETIME_EXCEEDED")
    if verify_signature is None:
        return _reject("NO_TRUSTED_VERIFIER")
    try:
        verified = verify_signature(grant.canonical_payload(), grant.signature, grant.issuer)
    except Exception:
        return _reject("SIGNATURE_VERIFICATION_FAILED")
    if verified is not True:
        return _reject("SIGNATURE_INVALID")
    if any((getattr(grant, key) != getattr(request, key)) for key in (
        "run_id", "model_sha256", "adapter_sha256", "corpus_sha256", "bank_object_id",
    )):
        return _reject("REQUEST_BINDING_MISMATCH")
    if not nonce_journal.consume(grant.issuer, grant.nonce):
        return _reject("NONCE_REPLAY")
    return GrantDecision(reason="SYNTHETIC_MATCH_NO_HOST_AUTHORITY", contract_match=True)
