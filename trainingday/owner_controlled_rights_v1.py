"""Owner-controlled R18 rights assertion: verify signature, NEVER issue ADMIT.
Unlike the legacy distinct-holder/steward profile, this is one owner-controlled
claim whose external key custody, source use rights, and host approval are UNKNOWN.
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from trainingday.admission import AdmissionError, _strict_json

SCHEMA = "OWNER_CONTROLLED_SELF_ATTEST_V1"
DOMAIN = b"VERA_OWNER_CONTROLLED_SELF_ATTEST_V1\x00"
PINNED_BASE = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PENDING = "OWNER_SIGNATURE_VALID_HOST_RATIFICATION_PENDING"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
FIELDS = frozenset({
    "schema", "corpus_family", "corpus_id", "manifest_sha256",
    "train_sha256", "dev_sha256", "protected_exclusion_sha256",
    "generator_source_sha256", "purpose", "base_revision",
    "owner_identity", "owner_key_sha256", "rights_basis",
    "decision", "not_before_utc", "expires_utc",
})
def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise AdmissionError(reason)

def _instant(raw: object) -> datetime:
    _require(type(raw) is str and re.fullmatch(
        r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", raw) is not None,
        "invalid owner rights timestamp")
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise AdmissionError("invalid owner rights UTC time") from exc

def inspect_owner_assertion(*, receipt_bytes: bytes, signature_hex: str,
    owner_public_key: bytes, expected_corpus_id: str, expected_hashes: dict,
    now_utc: datetime) -> dict:
    _require(type(receipt_bytes) is bytes and 0 < len(receipt_bytes) <= 16384,
             "owner assertion size invalid")
    claim = _strict_json(receipt_bytes)
    _require(type(claim) is dict and set(claim) == FIELDS, "wrong owner claim fields")
    canon = json.dumps(claim, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf8")
    _require(receipt_bytes == canon, "owner claim not canonical")
    _require(claim["schema"] == SCHEMA and claim["corpus_family"] == "R18_ARITHMETIC",
             "owner claim schema/corpus family mismatch")
    _require(claim["purpose"] == "QWEN35_VERA_TRAININGDAY" and
             claim["base_revision"] == PINNED_BASE, "owner scope/base mismatch")
    _require(claim["rights_basis"] == "OWNER_DECLARED_ORIGINAL_GENERATOR_PENDING_REVIEW",
             "owner provenance unresolved")
    _require(type(claim["decision"]) is str and claim["decision"] in
             ("ASSERT_RIGHTS_TO_TRAIN", "WITHHOLD"), "owner decision invalid")
    _require(type(claim["owner_identity"]) is str and
             1 <= len(claim["owner_identity"]) <= 120 and
             claim["owner_identity"].strip() == claim["owner_identity"],
             "owner identity invalid")
    for key in ("manifest_sha256", "train_sha256", "dev_sha256",
                "protected_exclusion_sha256", "generator_source_sha256",
                "owner_key_sha256"):
        _require(type(claim[key]) is str and HEX64.fullmatch(claim[key]) is not None,
                 "owner digest invalid")
    _require(type(expected_corpus_id) is str and claim["corpus_id"] == expected_corpus_id,
             "owner corpus mismatch")
    _require(type(expected_hashes) is dict and set(expected_hashes) == {
       "manifest_sha256", "train_sha256", "dev_sha256",
       "protected_exclusion_sha256", "generator_source_sha256"},
       "owner expected hash set invalid")
    _require(all(type(v) is str and claim[k] == v for k,v in expected_hashes.items()),
             "owner corpus/split hash mismatch")
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None and
             now_utc.utcoffset() is not None, "owner clock not authenticated")
    start, end = _instant(claim["not_before_utc"]), _instant(claim["expires_utc"])
    _require(start < end and start <= now_utc.astimezone(timezone.utc) < end,
             "owner attestation expired or not started")
    _require(type(owner_public_key) is bytes and len(owner_public_key) == 32
             and type(signature_hex) is str and HEX128.fullmatch(signature_hex) is not None,
             "invalid detached owner signature input")
    _require(hashlib.sha256(owner_public_key).hexdigest() == claim["owner_key_sha256"],
             "owner key fingerprint mismatch")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        raise AdmissionError("owner signature verification dependency missing") from exc
    try:
        Ed25519PublicKey.from_public_bytes(owner_public_key).verify(
            bytes.fromhex(signature_hex), DOMAIN + receipt_bytes)
    except (ValueError, InvalidSignature) as exc:
        raise AdmissionError("owner signature invalid") from exc
    return {
        "schema": SCHEMA,
        "status": "OWNER_WITHHELD" if claim["decision"] == "WITHHOLD" else PENDING,
        "signature_valid_for_supplied_key": True,
        "owner_key_enrollment_authenticated": False,
        "generator_rights_independently_reviewed": False,
        "protected_exclusion_custody_authenticated": False,
        "host_revocation_and_replay_authenticated": False,
        "corpus_rights_admitted": False,
        "physical_training_authorized": False,
        "protected_evaluation_authorized": False,
    }
