"""Inspect a registrar-signed owner-key pin candidate; NEVER enroll or ADMIT.

All trust-root provenance, clock, anti-rollback epoch and corpus commitments
are *inputs*, not independently authenticated here. In particular, a caller
can generate both keys and a matching registry. Even valid signatures give
only portable candidate evidence, never host owner enrollment or training.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from trainingday.admission import AdmissionError, _strict_json
from trainingday.owner_controlled_rights_v1 import (
    PINNED_BASE, PENDING, _instant, inspect_owner_assertion,
)

SCHEMA = "VERA_TRAININGDAY_HOST_OWNER_PIN_CANDIDATE_V1"
DOMAIN = b"VERA_TRAININGDAY_HOST_OWNER_PIN_CANDIDATE_V1\x00"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
FIELDS = frozenset({
    "schema", "registry_epoch", "owner_identity", "owner_key_sha256",
    "registrar_key_sha256", "purpose", "base_revision", "key_state",
    "not_before_utc", "expires_utc",
})


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise AdmissionError(reason)


def inspect_owner_key_enrollment_candidate(*, enrollment_bytes: bytes,
        registrar_signature_hex: str, registrar_public_key: bytes,
        expected_registrar_key_sha256: str, owner_public_key: bytes,
        owner_receipt_bytes: bytes, owner_signature_hex: str,
        expected_corpus_id: str, expected_hashes: dict,
        expected_owner_identity: str, minimum_registry_epoch: int,
        now_utc: datetime) -> dict:
    """Verify two detached signatures under a *supplied* registrar pin.

    ``expected_registrar_key_sha256`` must eventually come from an ACL-held,
    independently enrolled host anchor; this function does not establish that.
    Nor can ``minimum_registry_epoch`` prove anti-rollback without an external
    protected high-water mark. The result always explicitly retains HOLD.
    """
    _require(type(enrollment_bytes) is bytes and
             0 < len(enrollment_bytes) <= 8192, "registry size invalid")
    registry = _strict_json(enrollment_bytes)
    _require(type(registry) is dict and set(registry) == FIELDS,
             "registry field set invalid")
    canonical = json.dumps(registry, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    _require(enrollment_bytes == canonical, "registry not canonical")
    _require(registry["schema"] == SCHEMA, "registry schema invalid")
    _require(registry["purpose"] == "QWEN35_VERA_TRAININGDAY" and
             registry["base_revision"] == PINNED_BASE,
             "registry purpose/model scope mismatch")
    _require(type(expected_owner_identity) is str and
             1 <= len(expected_owner_identity) <= 120 and
             registry["owner_identity"] == expected_owner_identity,
             "owner identity mismatch")
    _require(type(minimum_registry_epoch) is int and
             1 <= minimum_registry_epoch <= (2**63 - 1) and
             type(registry["registry_epoch"]) is int and
             minimum_registry_epoch <= registry["registry_epoch"] <= (2**63 - 1),
             "stale registry epoch")
    _require(registry["key_state"] == "ACTIVE", "owner key revoked or inactive")
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None and
             now_utc.utcoffset() is not None, "registry clock not authenticated")
    start = _instant(registry["not_before_utc"])
    end = _instant(registry["expires_utc"])
    _require(start < end and start <= now_utc.astimezone(timezone.utc) < end,
             "registry expired or not started")

    _require(type(registrar_public_key) is bytes and len(registrar_public_key) == 32
             and type(expected_registrar_key_sha256) is str and
             HEX64.fullmatch(expected_registrar_key_sha256) is not None,
             "registrar pin input invalid")
    registrar_fingerprint = hashlib.sha256(registrar_public_key).hexdigest()
    _require(registrar_fingerprint == expected_registrar_key_sha256,
             "registrar key not pinned by supplied fingerprint")
    _require(type(registry["registrar_key_sha256"]) is str and
             registry["registrar_key_sha256"] == registrar_fingerprint,
             "registry registrar fingerprint mismatch")
    _require(type(registrar_signature_hex) is str and
             HEX128.fullmatch(registrar_signature_hex) is not None,
             "registrar signature invalid")
    _require(type(owner_public_key) is bytes and len(owner_public_key) == 32,
             "owner public key invalid")
    owner_fingerprint = hashlib.sha256(owner_public_key).hexdigest()
    _require(owner_fingerprint != registrar_fingerprint,
             "registrar and owner keys must differ")
    _require(type(registry["owner_key_sha256"]) is str and
             registry["owner_key_sha256"] == owner_fingerprint,
             "owner key pin mismatch")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise AdmissionError("registrar signature dependency missing") from exc
    try:
        Ed25519PublicKey.from_public_bytes(registrar_public_key).verify(
            bytes.fromhex(registrar_signature_hex), DOMAIN + enrollment_bytes)
    except (ValueError, InvalidSignature) as exc:
        raise AdmissionError("registrar signature invalid") from exc

    inspected = inspect_owner_assertion(
        receipt_bytes=owner_receipt_bytes, signature_hex=owner_signature_hex,
        owner_public_key=owner_public_key,
        expected_corpus_id=expected_corpus_id, expected_hashes=expected_hashes,
        now_utc=now_utc)
    claim = _strict_json(owner_receipt_bytes)
    _require(claim["owner_identity"] == expected_owner_identity,
             "owner receipt identity mismatch")
    _require(inspected["status"] == PENDING,
             "owner withheld or nonasserting decision")
    return {
        "schema": SCHEMA,
        "status": "REGISTRAR_SIGNATURE_VALID_SUPPLIED_PIN_HOST_CUSTODY_PENDING",
        "registrar_signature_valid_for_supplied_key": True,
        "owner_signature_valid_for_supplied_key": True,
        "registrar_host_identity_authenticated": False,
        "owner_key_enrollment_authenticated": False,
        "owner_rights_independently_reviewed": False,
        "protected_evaluator_custody_authenticated": False,
        "replay_anchor_authenticated": False,
        "corpus_rights_admitted": False,
        "gpu_lease_granted": False,
        "physical_training_authorized": False,
    }
