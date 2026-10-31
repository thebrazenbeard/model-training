"""R35 synthetic bank-grant *shape* contract; deliberately grants no access.

This module is not a security principal, signature verifier, replay store, or
private evaluation bank broker. The caller controls both the proposed claim
and the expected values; matching them proves internal consistency only.
An independently privileged bank custodian must perform actual authorization,
single-use nonce consumption, identity verification and durable access audit.
Nothing here reads an evaluation bank or can authorize a real bank read.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


SCHEMA = "VERA_R35_SYNTHETIC_BANK_GRANT_V1"
SCOPE = "R18_PRIVATE_SCORE_READ_ONLY"
MAX_TTL_SECONDS = 60.0

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_NONCE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_HEAD40 = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class ExpectedBankBinding:
    """Caller-declared expectation, explicitly NOT independent attestation."""

    bank_id: str
    principal_id: str
    plan_sha256: str
    evaluator_head: str
    attestation_sha256: str
    scope: str


@dataclass(frozen=True, slots=True)
class BankGrantRequest:
    """A synthetic grant request; no credentials, keys or bank contents."""

    schema: str
    bank_id: str
    principal_id: str
    plan_sha256: str
    evaluator_head: str
    attestation_sha256: str
    scope: str
    nonce: str
    issued_at_unix: float
    expires_at_unix: float


@dataclass(frozen=True, slots=True)
class BankGrantAssessment:
    status: str
    blockers: tuple[str, ...]
    authorizes_bank_read: bool = False
    consumes_nonce: bool = False
    attestation_independently_verified: bool = False
    host_custodian_installed: bool = False
    authorization_source: str = "NONE_SYNTHETIC_ONLY"


def _sha(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _identifier(value: object) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _finite_number(value: object) -> bool:
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        # math.isfinite converts integers to C double; huge Python ints can
        # overflow that conversion. An invalid clock is a HOLD, not a crash.
        return False


def _binding_valid(binding: ExpectedBankBinding | BankGrantRequest) -> bool:
    return (
        _identifier(binding.bank_id)
        and _identifier(binding.principal_id)
        and _sha(binding.plan_sha256, _SHA64)
        and _sha(binding.evaluator_head, _HEAD40)
        and _sha(binding.attestation_sha256, _SHA64)
        and type(binding.scope) is str
        and binding.scope == SCOPE
    )


def assess_synthetic_grant(
    request: object,
    expected: object,
    *,
    spent_nonces: object,
    now_unix: object,
) -> BankGrantAssessment:
    """Assess caller-declared fields; ALWAYS return non-authorizing status.

    A nonce in ``spent_nonces`` is a synthetic replay indicator, not an atomic
    claim or a reliable ledger. The caller also supplies ``now_unix``. Passing
    every check must never be interpreted as an authenticated bank grant.
    """
    if (
        type(request) is not BankGrantRequest
        or type(expected) is not ExpectedBankBinding
        or type(spent_nonces) is not frozenset
        or not _finite_number(now_unix)
        or now_unix <= 0
    ):
        return BankGrantAssessment("REJECTED_INVALID_SYNTHETIC_REQUEST", ("MALFORMED_CONTRACT",))

    blockers: list[str] = []
    spent_values_valid = all(type(nonce) is str for nonce in spent_nonces)
    if (
        type(request.schema) is not str
        or request.schema != SCHEMA
        or not _binding_valid(request)
        or not _binding_valid(expected)
        or type(request.nonce) is not str
        or _NONCE.fullmatch(request.nonce) is None
        or not spent_values_valid
    ):
        blockers.append("MALFORMED_CONTRACT")

    binding_fields = (
        "bank_id", "principal_id", "plan_sha256", "evaluator_head",
        "attestation_sha256", "scope",
    )
    if all(
        type(getattr(request, field)) is str
        and type(getattr(expected, field)) is str
        for field in binding_fields
    ):
        if any(
            getattr(request, field) != getattr(expected, field)
            for field in binding_fields
        ):
            blockers.append("BINDING_MISMATCH")

    if (
        not _finite_number(request.issued_at_unix)
        or not _finite_number(request.expires_at_unix)
        or request.issued_at_unix <= 0
        or request.issued_at_unix > now_unix
        or not now_unix < request.expires_at_unix
        or request.expires_at_unix <= request.issued_at_unix
        or request.expires_at_unix - request.issued_at_unix > MAX_TTL_SECONDS
    ):
        blockers.append("INVALID_TIME_WINDOW")

    if spent_values_valid and type(request.nonce) is str and request.nonce in spent_nonces:
        blockers.append("NONCE_ALREADY_SPENT_IN_DECLARED_SET")

    return BankGrantAssessment(
        status=("REJECTED_INVALID_SYNTHETIC_REQUEST" if blockers else
                "INTERNALLY_CONSISTENT_ONLY_CUSTODIAN_REQUIRED"),
        blockers=tuple(blockers),
    )
