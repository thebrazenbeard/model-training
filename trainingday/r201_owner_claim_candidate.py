"""R201 owner intention evidence: check exact signed scope, NEVER grant TRAIN.

An owner may also be the data steward. This verifier only checks a detached
signature against a *caller supplied* public key and exact expected digests.
Neither owner identity nor Calc source rights, DEV custody, time, revocation,
or the externally required GPU/host/NAS authorities are authenticated here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone

from trainingday.calc_curriculum import (
    CALC_REPOSITORY, CALC_REVISION, CALC_SYMBOLIC_SHA256,
    SCHEMA as TRAIN_SCHEMA, TRAIN_COUNT,
)


SCHEMA = "VERA_TRAININGDAY_R201_OWNER_CLAIM_CANDIDATE_V1"
DOMAIN = (SCHEMA + "\x00").encode("ascii")
CORPUS_ID = "R201_NEW_CALC_POLYNOMIAL_TRAIN"
TRAIN_SHA256 = "fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PURPOSE = "QWEN35_VERA_TRAININGDAY"
RIGHTS_BASIS = "OWNER_ASSERTS_ORIGINAL_GENERATOR_UNREVIEWED"
DECISIONS = frozenset({"ASSERT_RIGHTS_TO_TRAIN", "WITHHOLD"})
HASH_FIELDS = frozenset({
    "train_sha256", "calc_symbolic_sha256", "private_dev_commitment_sha256",
    "protected_exclusion_receipt_sha256", "owner_key_sha256",
})
FIELDS = HASH_FIELDS | frozenset({
    "schema", "corpus_id", "train_schema", "train_count",
    "calc_repository", "calc_revision", "model_id", "model_revision",
    "purpose", "owner_identity", "rights_basis", "decision",
    "owner_key_epoch", "not_before_utc", "expires_utc",
})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
UTC = re.compile(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


class ClaimEvidenceError(ValueError):
    """R201 evidence rejected without changing any authority state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimEvidenceError(message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    seen: set[str] = set()
    for key, value in pairs:
        # This protocol defines ASCII field names; reject NFKC/case and mixed-
        # script aliases rather than interpreting first-/last-key-wins JSON.
        _require(type(key) is str and key.isascii() and key.isprintable(),
                 "R201 owner claim JSON key must be printable ASCII")
        alias = key.casefold()
        _require(alias not in seen, "R201 owner claim duplicate key")
        seen.add(alias)
        result[key] = value
    return result


def _strict_json(raw: bytes) -> object:
    try:
        return json.loads(raw.decode("utf-8", errors="strict"),
                          object_pairs_hook=_strict_object,
                          parse_constant=lambda _: _require(False, "nonfinite JSON value"))
    except ClaimEvidenceError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ClaimEvidenceError("R201 owner claim invalid UTF-8 JSON") from exc


def _canonical_claim_bytes(claim: object) -> bytes:
    """Encode only canonical, representable UTF-8; reject escaped surrogates.

    Python's JSON decoder accepts escaped lone UTF-16 surrogates, but
    ``ensure_ascii=False`` cannot encode them as UTF-8. Normalize that
    malformed input to our evidence-rejection contract before any authority
    or signature result can be returned.
    """
    try:
        return json.dumps(claim, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")).encode("utf-8")
    except UnicodeError as exc:
        raise ClaimEvidenceError("R201 owner claim contains invalid Unicode") from exc
    except ValueError as exc:
        raise ClaimEvidenceError("R201 owner claim contains nonfinite numeric value") from exc


def _time(value: object) -> datetime:
    _require(type(value) is str and UTC.fullmatch(value) is not None,
             "R201 owner UTC timestamp malformed")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ClaimEvidenceError("R201 owner UTC timestamp invalid") from exc


def inspect_r201_owner_claim_candidate(
    *, claim_bytes: bytes, detached_signature_hex: str,
    supplied_owner_public_key: bytes, expected_owner_identity: str,
    expected_private_dev_commitment_sha256: str,
    expected_protected_exclusion_receipt_sha256: str,
    minimum_owner_key_epoch: int, now_utc: datetime,
) -> dict[str, object]:
    """Check synthetic/portable evidence, never return an ADMIT/launch grant.

    The two protected evaluator hashes are input commitments only: passing
    them does not establish independent custody. No DEV content is read here.
    """
    _require(type(claim_bytes) is bytes and 0 < len(claim_bytes) <= 16384,
             "R201 owner claim missing or oversized")
    claim = _strict_json(claim_bytes)
    _require(type(claim) is dict and set(claim) == FIELDS,
             "R201 owner claim field set invalid")
    _require(claim_bytes == _canonical_claim_bytes(claim),
             "R201 owner claim must be canonical JSON")
    expected = {
        "schema": SCHEMA, "corpus_id": CORPUS_ID,
        "train_schema": TRAIN_SCHEMA, "train_count": TRAIN_COUNT,
        "calc_repository": CALC_REPOSITORY, "calc_revision": CALC_REVISION,
        "calc_symbolic_sha256": CALC_SYMBOLIC_SHA256,
        "train_sha256": TRAIN_SHA256, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "purpose": PURPOSE,
        "rights_basis": RIGHTS_BASIS,
    }
    _require(all(type(claim[key]) is type(value) and claim[key] == value
                 for key, value in expected.items()),
             "R201 owner claim exact training subject mismatch")
    _require(type(expected_owner_identity) is str
             and 1 <= len(expected_owner_identity) <= 120
             and claim["owner_identity"] == expected_owner_identity
             and claim["owner_identity"].strip() == claim["owner_identity"]
             and not any(unicodedata.category(char) in {"Cc", "Cf"}
                         for char in claim["owner_identity"]),
             "R201 owner identity mismatch")
    _require(type(claim["decision"]) is str and claim["decision"] in DECISIONS,
             "R201 owner decision invalid")
    _require(all(type(claim[field]) is str
                 and HEX64.fullmatch(claim[field]) is not None
                 and claim[field] != "0" * 64 for field in HASH_FIELDS),
             "R201 owner digest missing or invalid")
    _require(type(expected_private_dev_commitment_sha256) is str
             and claim["private_dev_commitment_sha256"]
                 == expected_private_dev_commitment_sha256
             and type(expected_protected_exclusion_receipt_sha256) is str
             and claim["protected_exclusion_receipt_sha256"]
                 == expected_protected_exclusion_receipt_sha256,
             "R201 owner claim protected evaluator binding mismatch")
    _require(type(minimum_owner_key_epoch) is int
             and 1 <= minimum_owner_key_epoch <= (2**63 - 1)
             and type(claim["owner_key_epoch"]) is int
             and minimum_owner_key_epoch <= claim["owner_key_epoch"] <= (2**63 - 1),
             "R201 owner epoch stale or invalid")
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None
             and now_utc.utcoffset() is not None,
             "R201 owner clock must be timezone aware")
    begin, end = _time(claim["not_before_utc"]), _time(claim["expires_utc"])
    _require(begin < end and begin <= now_utc.astimezone(timezone.utc) < end,
             "R201 owner claim outside signed validity interval")
    _require(type(supplied_owner_public_key) is bytes
             and len(supplied_owner_public_key) == 32,
             "R201 owner public key input invalid")
    _require(hashlib.sha256(supplied_owner_public_key).hexdigest()
             == claim["owner_key_sha256"], "R201 owner key fingerprint mismatch")
    _require(type(detached_signature_hex) is str
             and HEX128.fullmatch(detached_signature_hex) is not None,
             "R201 owner detached signature format invalid")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise ClaimEvidenceError("Ed25519 verification dependency absent; hold") from exc
    try:
        Ed25519PublicKey.from_public_bytes(supplied_owner_public_key).verify(
            bytes.fromhex(detached_signature_hex), DOMAIN + claim_bytes)
    except (InvalidSignature, ValueError) as exc:
        raise ClaimEvidenceError("R201 owner detached signature invalid") from exc
    return {
        "schema": SCHEMA,
        "status": ("OWNER_WITHHELD_SOURCE_ONLY_HOLD" if claim["decision"] == "WITHHOLD"
                   else "OWNER_CLAIM_SIGNATURE_VALID_SUPPLIED_KEY_SOURCE_ONLY_HOLD"),
        "signed_decision": claim["decision"],
        "claim_sha256": hashlib.sha256(claim_bytes).hexdigest(),
        "signature_valid_for_supplied_key": True,
        "r201_exact_subject_checked": True,
        "owner_key_enrollment_authenticated": False,
        "owner_source_rights_independently_reviewed": False,
        "protected_evaluator_custody_authenticated": False,
        "host_epoch_and_clock_authenticated": False,
        "corpus_rights_admitted": False,
        "gpu_lease_granted": False,
        "physical_training_authorized": False,
        "protected_evaluation_authorized": False,
    }
