"""Source-only trust-boundary regression. Test keys never confer authority."""

import hashlib
import json
import unittest
from datetime import datetime, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trainingday.admission import AdmissionError
from trainingday.owner_controlled_rights_v1 import DOMAIN as OWNER_DOMAIN
from trainingday.owner_controlled_rights_v1 import PINNED_BASE
from trainingday.owner_controlled_rights_v1 import SCHEMA as OWNER_SCHEMA
from trainingday.owner_key_enrollment_candidate_v1 import (
    DOMAIN, SCHEMA, inspect_owner_key_enrollment_candidate,
)


NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
HASH_KEYS = ("manifest_sha256", "train_sha256", "dev_sha256",
             "protected_exclusion_sha256", "generator_source_sha256")


def public(private):
    return private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def fixture():
    admin = Ed25519PrivateKey.generate()
    owner = Ed25519PrivateKey.generate()
    admin_pub, owner_pub = public(admin), public(owner)
    hashes = {key: digest(key.encode("ascii")) for key in HASH_KEYS}
    enrollment = {
        "schema": SCHEMA,
        "registry_epoch": 7,
        "owner_identity": "Patrick-owner-synthetic",
        "owner_key_sha256": digest(owner_pub),
        "registrar_key_sha256": digest(admin_pub),
        "purpose": "QWEN35_VERA_TRAININGDAY",
        "base_revision": PINNED_BASE,
        "key_state": "ACTIVE",
        "not_before_utc": "2026-11-02T11:00:00Z",
        "expires_utc": "2026-11-03T00:00:00Z",
    }
    claim = {
        "schema": OWNER_SCHEMA,
        "corpus_family": "R18_ARITHMETIC",
        "corpus_id": "r18-synthetic-test-only",
        **hashes,
        "purpose": "QWEN35_VERA_TRAININGDAY",
        "base_revision": PINNED_BASE,
        "owner_identity": enrollment["owner_identity"],
        "owner_key_sha256": digest(owner_pub),
        "rights_basis": "OWNER_DECLARED_ORIGINAL_GENERATOR_PENDING_REVIEW",
        "decision": "ASSERT_RIGHTS_TO_TRAIN",
        "not_before_utc": "2026-11-02T11:00:00Z",
        "expires_utc": "2026-11-03T00:00:00Z",
    }
    return admin, owner, hashes, enrollment, claim


def inspect(admin, owner, hashes, enrollment, claim, *,
            external_registrar_pin=None, minimum_registry_epoch=7, now=NOW):
    admin_pub, owner_pub = public(admin), public(owner)
    registry_bytes, receipt_bytes = canonical(enrollment), canonical(claim)
    return inspect_owner_key_enrollment_candidate(
        enrollment_bytes=registry_bytes,
        registrar_signature_hex=admin.sign(DOMAIN + registry_bytes).hex(),
        registrar_public_key=admin_pub,
        expected_registrar_key_sha256=(external_registrar_pin or digest(admin_pub)),
        owner_public_key=owner_pub,
        owner_receipt_bytes=receipt_bytes,
        owner_signature_hex=owner.sign(OWNER_DOMAIN + receipt_bytes).hex(),
        expected_corpus_id="r18-synthetic-test-only",
        expected_hashes=hashes,
        expected_owner_identity="Patrick-owner-synthetic",
        minimum_registry_epoch=minimum_registry_epoch,
        now_utc=now,
    )


class OwnerKeyEnrollmentCandidateTests(unittest.TestCase):
    def test_self_generated_registrar_and_owner_keys_still_cannot_admit(self):
        result = inspect(*fixture())
        self.assertEqual(result["status"],
                         "REGISTRAR_SIGNATURE_VALID_SUPPLIED_PIN_HOST_CUSTODY_PENDING")
        self.assertIs(result["registrar_signature_valid_for_supplied_key"], True)
        self.assertIs(result["owner_signature_valid_for_supplied_key"], True)
        for field in ("registrar_host_identity_authenticated",
                      "owner_key_enrollment_authenticated",
                      "owner_rights_independently_reviewed",
                      "protected_evaluator_custody_authenticated",
                      "replay_anchor_authenticated",
                      "corpus_rights_admitted",
                      "gpu_lease_granted",
                      "physical_training_authorized"):
            self.assertIs(result[field], False, field)

    def test_fresh_self_signed_registrar_is_not_host_pin(self):
        admin, owner, hashes, enrollment, claim = fixture()
        host_root_pin = digest(public(Ed25519PrivateKey.generate()))
        with self.assertRaisesRegex(AdmissionError, "registrar key not pinned"):
            inspect(admin, owner, hashes, enrollment, claim,
                    external_registrar_pin=host_root_pin)

    def test_changed_owner_key_or_identity_is_not_enrolled(self):
        admin, owner, hashes, enrollment, claim = fixture()
        enrollment["owner_key_sha256"] = digest(public(Ed25519PrivateKey.generate()))
        with self.assertRaisesRegex(AdmissionError, "owner key pin mismatch"):
            inspect(admin, owner, hashes, enrollment, claim)
        admin, owner, hashes, enrollment, claim = fixture()
        enrollment["owner_identity"] = "other-owner"
        with self.assertRaisesRegex(AdmissionError, "owner identity mismatch"):
            inspect(admin, owner, hashes, enrollment, claim)

    def test_registrar_and_owner_must_use_distinct_signing_keys(self):
        admin, owner, hashes, enrollment, claim = fixture()
        same_pub = public(owner)
        enrollment["registrar_key_sha256"] = digest(same_pub)
        with self.assertRaisesRegex(AdmissionError, "keys must differ"):
            inspect(owner, owner, hashes, enrollment, claim)

    def test_revoked_expired_and_stale_epoch_rejected(self):
        admin, owner, hashes, enrollment, claim = fixture()
        enrollment["key_state"] = "REVOKED"
        with self.assertRaisesRegex(AdmissionError, "revoked"):
            inspect(admin, owner, hashes, enrollment, claim)
        admin, owner, hashes, enrollment, claim = fixture()
        with self.assertRaisesRegex(AdmissionError, "expired"):
            inspect(admin, owner, hashes, enrollment, claim,
                    now=datetime(2026, 11, 3, 12, tzinfo=timezone.utc))
        admin, owner, hashes, enrollment, claim = fixture()
        with self.assertRaisesRegex(AdmissionError, "stale registry epoch"):
            inspect(admin, owner, hashes, enrollment, claim,
                    minimum_registry_epoch=8)

    def test_receipt_hash_or_model_scope_rejected(self):
        admin, owner, hashes, enrollment, claim = fixture()
        claim["train_sha256"] = "0" * 64
        with self.assertRaisesRegex(AdmissionError, "hash mismatch"):
            inspect(admin, owner, hashes, enrollment, claim)
        admin, owner, hashes, enrollment, claim = fixture()
        enrollment["purpose"] = "UNRELATED_PURPOSE"
        with self.assertRaisesRegex(AdmissionError, "scope"):
            inspect(admin, owner, hashes, enrollment, claim)

    def test_withheld_owner_consent_never_counts_as_candidate(self):
        admin, owner, hashes, enrollment, claim = fixture()
        claim["decision"] = "WITHHOLD"
        with self.assertRaisesRegex(AdmissionError, "withheld"):
            inspect(admin, owner, hashes, enrollment, claim)

    def test_registry_replay_and_fields_are_bounded(self):
        admin, owner, hashes, enrollment, claim = fixture()
        enrollment["registry_epoch"] = True
        with self.assertRaisesRegex(AdmissionError, "stale registry epoch"):
            inspect(admin, owner, hashes, enrollment, claim)
        admin, owner, hashes, enrollment, claim = fixture()
        enrollment["owner_key_sha256"] = "0" * 64
        enrollment["owner_key_sha256_extra"] = "0" * 64
        with self.assertRaisesRegex(AdmissionError, "field set"):
            inspect(admin, owner, hashes, enrollment, claim)

    def test_canonical_and_registrar_signature_bounded(self):
        admin, owner, hashes, enrollment, claim = fixture()
        bad_registry = canonical(enrollment) + b" "
        receipt_bytes = canonical(claim)
        with self.assertRaisesRegex(AdmissionError, "canonical"):
            inspect_owner_key_enrollment_candidate(
                enrollment_bytes=bad_registry,
                registrar_signature_hex=admin.sign(DOMAIN + bad_registry).hex(),
                registrar_public_key=public(admin),
                expected_registrar_key_sha256=digest(public(admin)),
                owner_public_key=public(owner),
                owner_receipt_bytes=receipt_bytes,
                owner_signature_hex=owner.sign(OWNER_DOMAIN + receipt_bytes).hex(),
                expected_corpus_id="r18-synthetic-test-only",
                expected_hashes=hashes,
                expected_owner_identity="Patrick-owner-synthetic",
                minimum_registry_epoch=7, now_utc=NOW)
        admin, owner, hashes, enrollment, claim = fixture()
        with self.assertRaisesRegex(AdmissionError, "registrar signature invalid"):
            inspect_owner_key_enrollment_candidate(
                enrollment_bytes=canonical(enrollment),
                registrar_signature_hex="0" * 128,
                registrar_public_key=public(admin),
                expected_registrar_key_sha256=digest(public(admin)),
                owner_public_key=public(owner),
                owner_receipt_bytes=canonical(claim),
                owner_signature_hex=owner.sign(OWNER_DOMAIN + canonical(claim)).hex(),
                expected_corpus_id="r18-synthetic-test-only",
                expected_hashes=hashes,
                expected_owner_identity="Patrick-owner-synthetic",
                minimum_registry_epoch=7, now_utc=NOW)


if __name__ == "__main__":
    unittest.main()
