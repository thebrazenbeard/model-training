"""Unsigned, exact-subject R202 admission *request* validator; never grant TRAIN.

This checks a human-reviewable request against frozen R202 public-source
commitments and separately SUPPLIED evaluator hashes. It does not authenticate
any person, rights, private evaluator, clock, GPU, NAS, or host. In particular,
passing this check is intentionally never an authorization to train.
"""

from __future__ import annotations

import hashlib
import json
import re

from trainingday.cognitive_curriculum import (
    CORPUS_ID, SCHEMA as TRAIN_SCHEMA, TRAIN_COUNT, train_bytes,
)


SCHEMA = "VERA_TRAININGDAY_R202_ADMISSION_REQUEST_CANDIDATE_V1"
GENERATOR_GIT_BLOB_SHA1 = "0c0298561cff2568ef55cb2fcc90d7d245f268b0"
TRAIN_SHA256 = "3e9c67be9e690588c47e979af0f03981175fb4fc6c22cefcc5df8782c805726a"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PURPOSE = "NONCOMMERCIAL_VERA_TRAINING_ONLY"
DECISIONS = frozenset({"ASSERT_SCOPE_AND_REQUEST_REVIEW", "WITHHOLD"})
RIGHTS_BASES = frozenset({"ORIGINAL_CREATED_OR_COMMISSIONED", "DOCUMENTED_LICENSE"})
REQUIRED_EXCLUSIONS = frozenset({
    "private_chat_and_trauma_exports",
    "unlicensed_third_party_text_or_code",
    "protected_dev_and_test_material",
    "personal_or_confidential_information",
    "credentials_keys_and_secrets",
})
FIELDS = frozenset({
    "schema", "corpus_id", "train_schema", "train_count", "train_sha256",
    "generator_git_blob_sha1", "model_id", "model_revision", "purpose",
    "rights_holder_identity", "data_steward_identity", "rights_basis",
    "decision", "exclusions", "private_dev_commitment_sha256",
    "protected_exclusion_receipt_sha256",
})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class R202AdmissionEvidenceError(ValueError):
    """Reject a malformed or wrong-subject unsigned proposal; no authority effect."""


def _require(valid: bool, reason: str) -> None:
    if not valid:
        raise R202AdmissionEvidenceError(reason)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    seen = set()
    for key, value in pairs:
        _require(key.isascii() and key.isprintable(), "non-ASCII JSON key")
        _require(key.casefold() not in seen, "duplicate or aliased JSON key")
        seen.add(key.casefold())
        result[key] = value
    return result


def inspect_r202_admission_request_candidate(
    *, request_bytes: bytes, expected_private_dev_commitment_sha256: str,
    expected_protected_exclusion_receipt_sha256: str,
) -> dict[str, object]:
    """Validate public request *syntax and scope*, returning permanent HOLD.

    Expected evaluator hashes are caller-provided and themselves unauthenticated.
    Their format and match do not prove private DEV exists or remains secret.
    """
    _require(type(request_bytes) is bytes and 0 < len(request_bytes) <= 16384,
             "missing or oversized request")
    try:
        request = json.loads(request_bytes.decode("utf-8", errors="strict"),
                             object_pairs_hook=_unique_keys,
                             parse_constant=lambda _: _require(False, "nonfinite JSON"))
        canonical = json.dumps(request, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, R202AdmissionEvidenceError):
            raise
        raise R202AdmissionEvidenceError("malformed request JSON") from exc
    _require(type(request) is dict and set(request) == FIELDS,
             "wrong request fields")
    _require(canonical == request_bytes, "request JSON is not canonical")

    expected_subject = {
        "schema": SCHEMA, "corpus_id": CORPUS_ID,
        "train_schema": TRAIN_SCHEMA, "train_count": TRAIN_COUNT,
        "train_sha256": TRAIN_SHA256,
        "generator_git_blob_sha1": GENERATOR_GIT_BLOB_SHA1,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "purpose": PURPOSE,
    }
    _require(all(type(request[key]) is type(value) and request[key] == value
                 for key, value in expected_subject.items()),
             "R202 exact TRAIN/model/purpose/source subject mismatch")
    _require(hashlib.sha256(train_bytes()).hexdigest() == TRAIN_SHA256,
             "installed public generator no longer matches pinned TRAIN rows")
    for name in ("rights_holder_identity", "data_steward_identity"):
        value = request[name]
        _require(type(value) is str and value.strip() == value
                 and 1 <= len(value) <= 120
                 and all(char.isprintable() and char not in "\u202a\u202b\u202d\u202e\u2066\u2067\u2068\u2069"
                         for char in value), "invalid declared identity")
    _require(type(request["rights_basis"]) is str
             and request["rights_basis"] in RIGHTS_BASES,
             "unsupported or unspecified rights basis")
    _require(type(request["decision"]) is str
             and request["decision"] in DECISIONS, "invalid decision")
    exclusions = request["exclusions"]
    _require(type(exclusions) is list and all(type(item) is str for item in exclusions)
             and exclusions == sorted(set(exclusions))
             and REQUIRED_EXCLUSIONS.issubset(exclusions),
             "required exclusions absent, duplicated or unsorted")
    for field, supplied in (
        ("private_dev_commitment_sha256", expected_private_dev_commitment_sha256),
        ("protected_exclusion_receipt_sha256", expected_protected_exclusion_receipt_sha256),
    ):
        actual = request[field]
        _require(type(actual) is str and HEX64.fullmatch(actual) is not None
                 and actual != "0" * 64 and type(supplied) is str
                 and actual == supplied, "missing/mismatched protected commitment")

    return {
        "schema": SCHEMA,
        "status": "UNSIGNED_OWNER_WITHHOLD_HOLD" if request["decision"] == "WITHHOLD"
                  else "UNSIGNED_EXACT_SCOPE_REQUEST_HOLD",
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "exact_public_train_scope_checked": True,
        "supplied_evaluator_digests_match": True,
        "same_rights_holder_and_steward_permitted": True,
        "claimant_identity_authenticated": False,
        "rights_or_license_independently_authenticated": False,
        "request_signed_by_authenticated_owner": False,
        "protected_evaluation_custody_verified": False,
        "exclusions_verified_against_corpus": False,
        "model_tokenizer_and_optimizer_fit_qualified": False,
        "gpu_host_lease_granted": False,
        "nas_independent_monotonic_witness_qualified": False,
        "data_rights_admitted": False,
        "authorizes_training": False,
    }
