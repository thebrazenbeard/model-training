"""Synthetic owner/R185 exact-subject signatures; no live consent or ADMIT."""

import hashlib
import json
import unittest
from datetime import datetime, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trainingday.admission import AdmissionError
from trainingday.owner_controlled_rights_v1 import SCHEMA as V1_SCHEMA
from trainingday.owner_exact_subject_v2 import (
    BASE_ID, DOMAIN, HASH_FIELDS, PINNED_BASE, SCHEMA, STATUS,
    inspect_owner_exact_subject_v2,
)


NOW = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def fixture():
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    subject = {k: hashlib.sha256(k.encode("ascii")).hexdigest()
               for k in HASH_FIELDS}
    claim = {
        "schema": SCHEMA, "corpus_id": "R18_ARITHMETIC",
        "purpose": "QWEN35_VERA_TRAININGDAY", "base_id": BASE_ID,
        "base_revision": PINNED_BASE, **subject,
        "owner_identity": "synthetic-owner", "owner_key_sha256": hashlib.sha256(public).hexdigest(),
        "rights_basis": "OWNER_DECLARED_ORIGINAL_GENERATOR_PENDING_REVIEW",
        "decision": "ASSERT_RIGHTS_TO_TRAIN", "owner_key_epoch": 7,
        "owner_key_state": "ACTIVE", "not_before_utc": "2026-11-02T17:00:00Z",
        "expires_utc": "2026-11-03T00:00:00Z",
    }
    return key, public, subject, claim


def inspect(key, public, subject, claim, *, now=NOW, epoch=7):
    raw = canonical(claim)
    return inspect_owner_exact_subject_v2(
        receipt_bytes=raw, signature_hex=key.sign(DOMAIN + raw).hex(),
        owner_public_key=public, expected_hashes=subject,
        expected_owner_identity="synthetic-owner", minimum_owner_key_epoch=epoch,
        now_utc=now)


class OwnerExactSubjectV2Tests(unittest.TestCase):
    def test_valid_owner_signature_and_exact_hashes_still_hard_hold(self):
        result = inspect(*fixture())
        self.assertEqual(result["status"], STATUS)
        self.assertIs(result["signature_valid_for_supplied_key"], True)
        self.assertIs(result["exact_subject_consistent_with_supplied_hashes"], True)
        for name in ("owner_key_enrollment_authenticated",
                     "frozen_training_packet_authenticated",
                     "protected_evaluator_custody_authenticated",
                     "generator_rights_independently_reviewed",
                     "revocation_and_replay_authenticated", "corpus_rights_admitted",
                     "gpu_lease_granted", "physical_training_authorized",
                     "protected_evaluation_authorized"):
            self.assertIs(result[name], False, name)
        self.assertFalse(any("gold" in field or "prompt" in field
                             or "answer" in field for field in result))

    def test_result_contains_only_structural_boolean_evidence(self):
        result = inspect(*fixture())
        self.assertEqual(set(result), {
            "schema", "status", "signature_valid_for_supplied_key",
            "exact_subject_consistent_with_supplied_hashes",
            "owner_key_enrollment_authenticated",
            "frozen_training_packet_authenticated",
            "protected_evaluator_custody_authenticated",
            "generator_rights_independently_reviewed",
            "revocation_and_replay_authenticated", "corpus_rights_admitted",
            "gpu_lease_granted", "physical_training_authorized",
            "protected_evaluation_authorized",
        })
        for field in HASH_FIELDS:
            self.assertNotIn(field, result)

    def test_each_exact_subject_hash_is_independently_bound(self):
        for field in HASH_FIELDS:
            with self.subTest(field=field):
                key, pub, subject, claim = fixture()
                claim[field] = "a" * 64
                with self.assertRaisesRegex(AdmissionError, "exact R185/evaluator subject"):
                    inspect(key, pub, subject, claim)

    def test_packet_train_content_and_id_v2_are_not_interchangeable(self):
        for left, right in (("frozen_training_packet_sha256", "train_content_sha256"),
                            ("train_content_sha256", "selected_id_v2_sha256"),
                            ("train_sha256", "selected_id_v2_sha256")):
            with self.subTest(left=left, right=right):
                key, pub, subject, claim = fixture()
                claim[left], claim[right] = claim[right], claim[left]
                with self.assertRaises(AdmissionError):
                    inspect(key, pub, subject, claim)

    def test_caller_can_coforge_subject_but_gets_no_authority(self):
        key, pub, subject, claim = fixture()
        subject["frozen_training_packet_sha256"] = "a" * 64
        claim["frozen_training_packet_sha256"] = "a" * 64
        result = inspect(key, pub, subject, claim)
        self.assertEqual(result["status"], STATUS)
        self.assertIs(result["frozen_training_packet_authenticated"], False)
        self.assertIs(result["physical_training_authorized"], False)

    def test_withheld_claim_is_valid_but_never_admits(self):
        key, pub, subject, claim = fixture()
        claim["decision"] = "WITHHOLD"
        result = inspect(key, pub, subject, claim)
        self.assertEqual(result["status"], "OWNER_WITHHELD_HOLD")
        self.assertIs(result["corpus_rights_admitted"], False)

    def test_wrong_base_purpose_corpus_or_rights_basis_rejected(self):
        bad = (("base_id", "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"),
               ("base_revision", "0" * 40), ("purpose", "OTHER"),
               ("corpus_id", "LABYRINTH_COGNITIVE"),
               ("rights_basis", "RIGHTS_APPROVED"))
        for field, val in bad:
            with self.subTest(field=field):
                key, pub, subject, claim = fixture()
                claim[field] = val
                with self.assertRaises(AdmissionError):
                    inspect(key, pub, subject, claim)

    def test_inactive_key_or_stale_epoch_rejected(self):
        key, pub, subject, claim = fixture()
        claim["owner_key_state"] = "REVOKED"
        with self.assertRaisesRegex(AdmissionError, "revoked"):
            inspect(key, pub, subject, claim)
        for bad_epoch in (True, 0, -1, 2 ** 64):
            with self.subTest(epoch=bad_epoch):
                key, pub, subject, claim = fixture()
                claim["owner_key_epoch"] = bad_epoch
                with self.assertRaisesRegex(AdmissionError, "epoch"):
                    inspect(key, pub, subject, claim)
        key, pub, subject, claim = fixture()
        with self.assertRaisesRegex(AdmissionError, "epoch"):
            inspect(key, pub, subject, claim, epoch=8)

    def test_clock_expiry_and_unsigned_replay_rejected(self):
        key, pub, subject, claim = fixture()
        with self.assertRaisesRegex(AdmissionError, "expired"):
            inspect(key, pub, subject, claim,
                    now=datetime(2026, 11, 4, tzinfo=timezone.utc))
        with self.assertRaisesRegex(AdmissionError, "clock"):
            inspect(key, pub, subject, claim, now=datetime(2026, 11, 2))

    def test_wrong_owner_public_key_or_signature_rejected(self):
        key, pub, subject, claim = fixture()
        other_pub = fixture()[1]
        with self.assertRaisesRegex(AdmissionError, "fingerprint"):
            inspect(key, other_pub, subject, claim)
        raw = canonical(claim)
        with self.assertRaisesRegex(AdmissionError, "signature invalid"):
            inspect_owner_exact_subject_v2(receipt_bytes=raw,
                signature_hex="0" * 128, owner_public_key=pub,
                expected_hashes=subject, expected_owner_identity="synthetic-owner",
                minimum_owner_key_epoch=7, now_utc=NOW)

    def test_canonical_exact_fields_and_v1_schema_are_distinct(self):
        key, pub, subject, claim = fixture()
        for change in ("extra", "v1", "dev_missing"):
            with self.subTest(change=change):
                key, pub, subject, claim = fixture()
                if change == "extra":
                    claim["authz"] = True
                elif change == "v1":
                    claim["schema"] = V1_SCHEMA
                else:
                    del claim["dev_sha256"]
                with self.assertRaises(AdmissionError):
                    inspect(key, pub, subject, claim)
        key, pub, subject, claim = fixture()
        raw = json.dumps(claim, sort_keys=True, indent=2).encode()
        with self.assertRaisesRegex(AdmissionError, "canonical"):
            inspect_owner_exact_subject_v2(receipt_bytes=raw,
                signature_hex=key.sign(DOMAIN + raw).hex(), owner_public_key=pub,
                expected_hashes=subject, expected_owner_identity="synthetic-owner",
                minimum_owner_key_epoch=7, now_utc=NOW)

    def test_duplicate_json_key_cannot_hide_private_fields(self):
        key, pub, subject, claim = fixture()
        raw = canonical(claim)
        duplicate = raw.replace(b'"dev_sha256":',
                                b'"dev_sha256":"' + b'0' * 64 + b'","dev_sha256":')
        with self.assertRaisesRegex(AdmissionError, "duplicate"):
            inspect_owner_exact_subject_v2(receipt_bytes=duplicate,
                signature_hex=key.sign(DOMAIN + duplicate).hex(),
                owner_public_key=pub, expected_hashes=subject,
                expected_owner_identity="synthetic-owner",
                minimum_owner_key_epoch=7, now_utc=NOW)


if __name__ == "__main__":
    unittest.main()
