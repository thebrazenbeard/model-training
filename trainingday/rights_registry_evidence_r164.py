"""Source-only registry integrity check for TrainingDay rights evidence.

The registry issuer public key, generation floor, previous digest, and clock
are inputs, NOT trusted roots.  A real host-owned issuer registry and durable
anti-rollback state must authenticate them before any corpus can be admitted.
Even an entirely valid self-signed registry plus rights pair stays DENIED.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from trainingday.admission import AdmissionError, _strict_json
from trainingday.rights_pair_evidence_r155 import (
    HEX64, MAX_RECEIPT_BYTES, _time, _verify_ed25519, inspect_rights_pair,
)


SCHEMA = "VERA_TRAININGDAY_RIGHTS_KEY_REGISTRY_V1"
DOMAIN = SCHEMA.encode("ascii") + b"\x00"
MAX_REGISTRY_BYTES = 64 * 1024
MAX_SUBJECTS = 128
SUBJECT_FIELDS = frozenset({"role", "id", "key_sha256"})
REGISTRY_FIELDS = frozenset({
    "schema", "issuer_id", "issuer_key_sha256", "generation",
    "previous_snapshot_sha256", "not_before_utc", "expires_utc",
    "subjects", "revoked_key_sha256",
})
UNTRUSTED_RESULT = "REGISTRY_INTEGRITY_ONLY_HOST_ROOT_UNVERIFIED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def inspect_registry_bound_rights_pair(
    *, registry_bytes: bytes, registry_signature_hex: str,
    issuer_public_key: bytes, rights_receipt_bytes: bytes,
    holder_signature_hex: str, steward_signature_hex: str,
    holder_public_key: bytes, steward_public_key: bytes,
    expected_corpus_id: str, expected_manifest_sha256: str,
    expected_train_sha256: str, expected_dev_sha256: str,
    expected_protected_exclusion_sha256: str,
    required_generation_floor: int, required_previous_snapshot_sha256: str | None,
    now_utc: datetime,
) -> dict[str, object]:
    """Check signed registry and rights-pair consistency, NEVER authorize TRAIN.

    All provided keys, clock and anti-rollback anchors are *caller-supplied*.
    Passing every check proves only consistency/integrity relative to those
    inputs, not an independent rights-holder, steward or host issuer.
    """
    _require(type(registry_bytes) is bytes
             and 0 < len(registry_bytes) <= MAX_REGISTRY_BYTES,
             "registry missing or oversized")
    registry = _strict_json(registry_bytes)
    _require(type(registry) is dict and set(registry) == REGISTRY_FIELDS,
             "registry field set invalid")
    canonical = json.dumps(registry, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    _require(canonical == registry_bytes, "registry must use canonical JSON")
    _require(registry["schema"] == SCHEMA, "registry schema mismatch")
    _require(type(registry["issuer_id"]) is str
             and 1 <= len(registry["issuer_id"]) <= 120
             and registry["issuer_id"].strip() == registry["issuer_id"],
             "registry issuer identity malformed")
    _require(type(issuer_public_key) is bytes and len(issuer_public_key) == 32,
             "registry issuer key malformed")
    issuer_fp = hashlib.sha256(issuer_public_key).hexdigest()
    _require(registry["issuer_key_sha256"] == issuer_fp,
             "registry signer key binding mismatch")
    _require(type(registry["generation"]) is int
             and 1 <= registry["generation"] <= 2**63 - 1,
             "registry generation invalid")
    _require(type(required_generation_floor) is int
             and required_generation_floor >= 1
             and registry["generation"] >= required_generation_floor,
             "registry generation rollback")
    previous = registry["previous_snapshot_sha256"]
    _require(previous is None or
             (type(previous) is str and HEX64.fullmatch(previous) is not None),
             "registry previous digest malformed")
    _require((registry["generation"] == 1 and previous is None)
             or (registry["generation"] > 1 and previous is not None),
             "registry chain shape invalid")
    _require(required_previous_snapshot_sha256 is None or
             (type(required_previous_snapshot_sha256) is str
              and HEX64.fullmatch(required_previous_snapshot_sha256) is not None),
             "required previous registry digest malformed")
    _require(previous == required_previous_snapshot_sha256,
             "registry previous digest mismatch")
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None
             and now_utc.utcoffset() is not None,
             "registry requires timezone-aware clock")
    start, end = _time(registry["not_before_utc"]), _time(registry["expires_utc"])
    instant = now_utc.astimezone(timezone.utc)
    _require(start < end and start <= instant < end,
             "registry expired or not yet valid")

    revoked = registry["revoked_key_sha256"]
    _require(type(revoked) is list and len(revoked) <= MAX_SUBJECTS
             and all(type(item) is str and HEX64.fullmatch(item) is not None
                     for item in revoked)
             and revoked == sorted(set(revoked)),
             "registry revocation list invalid")
    _require(issuer_fp not in revoked, "registry issuer is revoked")
    subjects = registry["subjects"]
    _require(type(subjects) is list and 2 <= len(subjects) <= MAX_SUBJECTS,
             "registry subject list invalid")
    seen_ids: set[tuple[str, str]] = set()
    seen_keys: set[str] = set()
    roster: dict[tuple[str, str], str] = {}
    for item in subjects:
        _require(type(item) is dict and set(item) == SUBJECT_FIELDS,
                 "registry subject malformed")
        role, subject_id, key_fp = item["role"], item["id"], item["key_sha256"]
        _require(type(role) is str and role in {"RIGHTS_HOLDER", "DATA_STEWARD"},
                 "registry subject role invalid")
        _require(type(subject_id) is str and 1 <= len(subject_id) <= 120
                 and subject_id.strip() == subject_id,
                 "registry subject identity invalid")
        _require(type(key_fp) is str and HEX64.fullmatch(key_fp) is not None,
                 "registry subject key invalid")
        record_id = (role, subject_id.casefold())
        _require(record_id not in seen_ids and key_fp not in seen_keys,
                 "registry duplicate identity/key")
        _require(key_fp != issuer_fp and key_fp not in revoked,
                 "registry issuer/subject conflict or revoked subject")
        seen_ids.add(record_id)
        seen_keys.add(key_fp)
        roster[record_id] = key_fp

    _verify_ed25519(issuer_public_key, registry_signature_hex,
                    DOMAIN + registry_bytes)
    _require(type(rights_receipt_bytes) is bytes
             and 0 < len(rights_receipt_bytes) <= MAX_RECEIPT_BYTES,
             "rights receipt missing or oversized")
    rights = _strict_json(rights_receipt_bytes)
    _require(type(rights) is dict, "rights receipt malformed")
    _require(rights.get("revocation_snapshot_sha256")
             == hashlib.sha256(registry_bytes).hexdigest(),
             "rights receipt not bound to exact registry snapshot")
    for role, identity_field, fp_field, public_key in (
        ("RIGHTS_HOLDER", "rights_holder_id", "rights_holder_key_sha256",
         holder_public_key),
        ("DATA_STEWARD", "steward_id", "steward_key_sha256",
         steward_public_key),
    ):
        subject_id = rights.get(identity_field)
        _require(type(subject_id) is str,
                 "rights identity missing in registry")
        key_fp = hashlib.sha256(public_key).hexdigest() if type(public_key) is bytes else None
        _require(roster.get((role, subject_id.casefold())) == key_fp
                 and rights.get(fp_field) == key_fp,
                 "rights signer absent or wrong role in registry")

    evidence = inspect_rights_pair(
        receipt_bytes=rights_receipt_bytes,
        holder_signature_hex=holder_signature_hex,
        steward_signature_hex=steward_signature_hex,
        holder_public_key=holder_public_key,
        steward_public_key=steward_public_key,
        expected_corpus_id=expected_corpus_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_train_sha256=expected_train_sha256,
        expected_dev_sha256=expected_dev_sha256,
        expected_protected_exclusion_sha256=expected_protected_exclusion_sha256,
        revoked_key_sha256=frozenset(revoked),
        now_utc=now_utc,
    )
    return {
        **evidence,
        "status": (evidence["status"] if evidence["signed_decision"] == "REJECT"
                   else UNTRUSTED_RESULT),
        "registry_snapshot_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "registry_generation_checked_against_supplied_floor": True,
        "registry_signature_valid_for_supplied_issuer_key": True,
        "registry_role_bindings_checked": True,
        "host_registry_issuer_authenticated": False,
        "host_registry_rollback_fence_authenticated": False,
        "corpus_rights_admitted": False,
        "physical_training_authorized": False,
        "protected_evaluation_authorized": False,
    }
