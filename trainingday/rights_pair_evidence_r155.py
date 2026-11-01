"""Detached, dual-signature rights evidence; never an admission authority.

The public-key registry, revocation high-water mark, clock, and custody of
the underlying corpus are external trust boundaries.  A caller can supply a
matching pair of self-generated keys, so even two valid signatures MUST NOT
authorize physical training through this source-only verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from trainingday.admission import AdmissionError, _strict_json


SCHEMA = "VERA_TRAININGDAY_RIGHTS_PAIR_V1"
DOMAIN = SCHEMA.encode("ascii") + b"\x00"
PINNED_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
UNTRUSTED_RESULT = "SIGNATURES_VALID_EXTERNAL_CUSTODY_UNVERIFIED"
REJECT_RESULT = "SIGNED_REJECT_NO_TRAINING"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
UTC_TIME = re.compile(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
MAX_RECEIPT_BYTES = 16 * 1024
REQUIRED = frozenset({
    "schema", "corpus_family", "corpus_id", "manifest_sha256", "train_sha256",
    "dev_sha256", "protected_exclusion_sha256", "license_expression",
    "license_evidence_sha256", "base_revision", "purpose", "decision",
    "rights_holder_id", "rights_holder_key_sha256", "steward_id",
    "steward_key_sha256", "revocation_snapshot_sha256", "not_before_utc",
    "expires_utc",
})
LICENSE_FOR_FAMILY = {
    "DOLLY_V4": "CC-BY-SA-3.0",
    "REZON_NEGATIVE_PEER": "REZON-SECTION-5-PRIOR-WRITTEN",
    "LABYRINTH_SYNTHETIC": "PROVENANCE-REVIEW-REQUIRED",
}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AdmissionError(message)


def _hex64(value: object) -> bool:
    return type(value) is str and HEX64.fullmatch(value) is not None


def _time(value: object) -> datetime:
    _require(type(value) is str and UTC_TIME.fullmatch(value) is not None,
             "rights receipt UTC time malformed")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise AdmissionError("rights receipt UTC time invalid") from exc


def _verify_ed25519(public_key: bytes, signature_hex: str, payload: bytes) -> None:
    _require(type(public_key) is bytes and len(public_key) == 32,
             "rights key must be raw 32-byte Ed25519 public key")
    _require(type(signature_hex) is str and HEX128.fullmatch(signature_hex) is not None,
             "rights signature must be canonical lowercase Ed25519 hex")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise AdmissionError("Ed25519 verifier dependency unavailable; fail closed") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(signature_hex), payload)
    except (ValueError, InvalidSignature) as exc:
        raise AdmissionError("invalid detached Ed25519 rights signature") from exc


def inspect_rights_pair(
    *, receipt_bytes: bytes, holder_signature_hex: str,
    steward_signature_hex: str, holder_public_key: bytes,
    steward_public_key: bytes, expected_corpus_id: str,
    expected_manifest_sha256: str, expected_train_sha256: str,
    expected_dev_sha256: str, expected_protected_exclusion_sha256: str,
    revoked_key_sha256: frozenset[str], now_utc: datetime,
) -> dict[str, object]:
    """Inspect a detached evidence pair without conferring ADMIT or host rights.

    Even a valid pair from distinct supplied keys cannot authenticate those
    keys as independent rights-holder/steward identities.  Likewise the
    revocation snapshot and clock here are caller-provided and cannot prove
    freshness. An independent host-owned verifier must bind these externally.
    """
    _require(type(receipt_bytes) is bytes and 0 < len(receipt_bytes) <= MAX_RECEIPT_BYTES,
             "missing or oversized rights receipt")
    receipt = _strict_json(receipt_bytes)
    _require(type(receipt) is dict and set(receipt) == REQUIRED,
             "rights receipt schema/field set invalid")
    canonical = json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    _require(canonical == receipt_bytes, "rights receipt must use exact canonical JSON")
    _require(receipt["schema"] == SCHEMA, "rights receipt wrong schema")
    family = receipt["corpus_family"]
    _require(type(family) is str and family in LICENSE_FOR_FAMILY,
             "unrecognized rights corpus family")
    _require(receipt["license_expression"] == LICENSE_FOR_FAMILY[family],
             "rights license does not match corpus family")
    _require(receipt["purpose"] == "QWEN35_VERA_TRAININGDAY" and
             receipt["base_revision"] == PINNED_BASE_REVISION,
             "rights purpose/base revision mismatch")
    _require(receipt["decision"] in ("ADMIT", "REJECT") and
             type(receipt["decision"]) is str, "rights decision malformed")
    for key in ("corpus_id", "rights_holder_id", "steward_id"):
        _require(type(receipt[key]) is str and 1 <= len(receipt[key]) <= 120
                 and receipt[key].strip() == receipt[key], "invalid rights identity")
    _require(receipt["rights_holder_id"].casefold() != receipt["steward_id"].casefold(),
             "rights holder and steward identities must differ")
    for key in ("manifest_sha256", "train_sha256", "dev_sha256",
                "protected_exclusion_sha256", "license_evidence_sha256",
                "rights_holder_key_sha256", "steward_key_sha256",
                "revocation_snapshot_sha256"):
        _require(_hex64(receipt[key]), "missing canonical rights commitment")
    expected = {
        "corpus_id": expected_corpus_id,
        "manifest_sha256": expected_manifest_sha256,
        "train_sha256": expected_train_sha256,
        "dev_sha256": expected_dev_sha256,
        "protected_exclusion_sha256": expected_protected_exclusion_sha256,
    }
    _require(all(receipt[key] == value for key, value in expected.items()),
             "rights grant not bound to requested exact corpus bytes/splits")
    _require(type(revoked_key_sha256) is frozenset and
             all(_hex64(key) for key in revoked_key_sha256),
             "revocation input not a bounded fingerprint set")
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None
             and now_utc.utcoffset() is not None,
             "rights receipt requires explicit timezone-aware clock")
    start, end = _time(receipt["not_before_utc"]), _time(receipt["expires_utc"])
    instant = now_utc.astimezone(timezone.utc)
    _require(start < end and start <= instant < end,
             "rights receipt not current at supplied clock")
    _require(type(holder_public_key) is bytes and type(steward_public_key) is bytes
             and holder_public_key != steward_public_key,
             "independent roles cannot reuse a public key")
    holder_fp = hashlib.sha256(holder_public_key).hexdigest()
    steward_fp = hashlib.sha256(steward_public_key).hexdigest()
    _require(holder_fp == receipt["rights_holder_key_sha256"] and
             steward_fp == receipt["steward_key_sha256"],
             "rights signature keys differ from receipt binding")
    _require(not ({holder_fp, steward_fp} & revoked_key_sha256),
             "rights signature key is revoked")
    message = DOMAIN + receipt_bytes
    _verify_ed25519(holder_public_key, holder_signature_hex, message)
    _verify_ed25519(steward_public_key, steward_signature_hex, message)
    return {
        "schema": SCHEMA,
        "status": REJECT_RESULT if receipt["decision"] == "REJECT" else UNTRUSTED_RESULT,
        "signed_decision": receipt["decision"],
        "corpus_id": receipt["corpus_id"],
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "detached_signatures_verify_for_supplied_keys": True,
        "external_key_registry_authenticated": False,
        "revocation_freshness_authenticated": False,
        "rights_holder_independence_authenticated": False,
        "corpus_rights_admitted": False,
        "physical_training_authorized": False,
        "protected_evaluation_authorized": False,
    }
