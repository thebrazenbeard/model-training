"""Validate an exact R18 owner-signed subject V2; NEVER admit training.

This is portable signature/subject-consistency evidence only. The signer key,
expected frozen hashes, time and minimum epoch are caller-provided, not host-
authenticated. A caller can forge those together and pass this validator;
all rights, evaluator, GPU and physical-training grants MUST remain false.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from trainingday.admission import AdmissionError, _strict_json
from trainingday.owner_controlled_rights_v1 import PINNED_BASE, _instant


SCHEMA = "OWNER_CONTROLLED_R18_EXACT_SUBJECT_V2"
DOMAIN = b"VERA_OWNER_CONTROLLED_R18_EXACT_SUBJECT_V2\x00"
BASE_ID = "Qwen/Qwen3.5-4B"
PURPOSE = "QWEN35_VERA_TRAININGDAY"
STATUS = "OWNER_EXACT_SUBJECT_SIGNATURE_VALID_SUPPLIED_INPUTS_HOLD"
HASH_FIELDS = frozenset({
    "frozen_training_packet_sha256", "manifest_sha256", "train_sha256",
    "train_content_sha256", "selected_id_v2_sha256",
    "generator_source_sha256", "dev_sha256",
    "protected_exclusion_sha256",
})
FIELDS = HASH_FIELDS | frozenset({
    "schema", "corpus_id", "purpose", "base_id", "base_revision",
    "owner_identity", "owner_key_sha256", "rights_basis", "decision",
    "owner_key_epoch", "owner_key_state", "not_before_utc", "expires_utc",
})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
MAX_EPOCH = (2 ** 63) - 1


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise AdmissionError(reason)


def inspect_owner_exact_subject_v2(*, receipt_bytes: bytes,
        signature_hex: str, owner_public_key: bytes,
        expected_hashes: dict, expected_owner_identity: str,
        minimum_owner_key_epoch: int, now_utc: datetime) -> dict:
    """Verify an exact signed subject against *caller-supplied* expectations.

    Neither the caller-provided owner key nor the expected R185/evaluator
    digests are authenticated here. The result never authorizes anything.
    """
    _require(type(receipt_bytes) is bytes and
             0 < len(receipt_bytes) <= 16384, "V2 owner assertion size invalid")
    claim = _strict_json(receipt_bytes)
    _require(type(claim) is dict and set(claim) == FIELDS,
             "V2 owner claim field set invalid")
    canon = json.dumps(claim, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
    _require(receipt_bytes == canon, "V2 owner claim not canonical")
    _require(claim["schema"] == SCHEMA and claim["corpus_id"] == "R18_ARITHMETIC",
             "V2 owner claim schema/corpus mismatch")
    _require(claim["purpose"] == PURPOSE and claim["base_id"] == BASE_ID
             and claim["base_revision"] == PINNED_BASE,
             "V2 owner purpose/base mismatch")
    _require(claim["rights_basis"] ==
             "OWNER_DECLARED_ORIGINAL_GENERATOR_PENDING_REVIEW",
             "V2 owner provenance still pending")
    _require(type(claim["decision"]) is str and claim["decision"] in
             ("ASSERT_RIGHTS_TO_TRAIN", "WITHHOLD"), "V2 owner decision invalid")
    _require(claim["owner_key_state"] == "ACTIVE",
             "V2 owner key revoked or inactive")

    _require(type(expected_owner_identity) is str and
             1 <= len(expected_owner_identity) <= 120 and
             expected_owner_identity.strip() == expected_owner_identity and
             claim["owner_identity"] == expected_owner_identity,
             "V2 owner identity mismatch")
    _require(type(minimum_owner_key_epoch) is int and
             1 <= minimum_owner_key_epoch <= MAX_EPOCH and
             type(claim["owner_key_epoch"]) is int and
             minimum_owner_key_epoch <= claim["owner_key_epoch"] <= MAX_EPOCH,
             "V2 owner key epoch stale")

    _require(type(expected_hashes) is dict and set(expected_hashes) == HASH_FIELDS,
             "V2 expected exact-subject field set invalid")
    _require(all(type(claim[k]) is str and HEX64.fullmatch(claim[k]) is not None
                 and type(expected_hashes[k]) is str and
                 claim[k] == expected_hashes[k]
                 for k in HASH_FIELDS), "V2 exact R185/evaluator subject mismatch")
    _require(type(claim["owner_key_sha256"]) is str and
             HEX64.fullmatch(claim["owner_key_sha256"]) is not None,
             "V2 owner key fingerprint invalid")
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None and
             now_utc.utcoffset() is not None, "V2 owner clock not authenticated")
    start, end = _instant(claim["not_before_utc"]), _instant(claim["expires_utc"])
    _require(start < end and start <= now_utc.astimezone(timezone.utc) < end,
             "V2 owner assertion expired or not started")

    _require(type(owner_public_key) is bytes and len(owner_public_key) == 32 and
             type(signature_hex) is str and HEX128.fullmatch(signature_hex) is not None,
             "V2 detached signature input invalid")
    _require(hashlib.sha256(owner_public_key).hexdigest() == claim["owner_key_sha256"],
             "V2 owner key fingerprint mismatch")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise AdmissionError("V2 owner signature dependency missing") from exc
    try:
        Ed25519PublicKey.from_public_bytes(owner_public_key).verify(
            bytes.fromhex(signature_hex), DOMAIN + receipt_bytes)
    except (ValueError, InvalidSignature) as exc:
        raise AdmissionError("V2 owner signature invalid") from exc

    # Even a valid assertion cannot prove where the key, frozen R185 subject,
    # protected evaluator hashes, clock or rollback floor came from.
    return {
        "schema": SCHEMA,
        "status": "OWNER_WITHHELD_HOLD" if claim["decision"] == "WITHHOLD" else STATUS,
        "signature_valid_for_supplied_key": True,
        "exact_subject_consistent_with_supplied_hashes": True,
        "owner_key_enrollment_authenticated": False,
        "frozen_training_packet_authenticated": False,
        "protected_evaluator_custody_authenticated": False,
        "generator_rights_independently_reviewed": False,
        "revocation_and_replay_authenticated": False,
        "corpus_rights_admitted": False,
        "gpu_lease_granted": False,
        "physical_training_authorized": False,
        "protected_evaluation_authorized": False,
    }
