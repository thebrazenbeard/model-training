"""Synthetic signer tests only; never enroll/issue a live rights authority."""
import hashlib
import json
import unittest
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from trainingday.admission import AdmissionError
from trainingday.owner_controlled_rights_v1 import (
    DOMAIN, PINNED_BASE, SCHEMA, PENDING, inspect_owner_assertion,
)

HASH_KEYS = ("manifest_sha256", "train_sha256", "dev_sha256",
             "protected_exclusion_sha256", "generator_source_sha256")
NOW = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)

def fixture():
    secret = Ed25519PrivateKey.generate()
    pub = secret.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    hashes = {key: hashlib.sha256(key.encode()).hexdigest() for key in HASH_KEYS}
    receipt = dict(schema=SCHEMA, corpus_family="R18_ARITHMETIC",
        corpus_id="r18-test-only", **hashes, purpose="QWEN35_VERA_TRAININGDAY",
        base_revision=PINNED_BASE, owner_identity="owner-test",
        owner_key_sha256=hashlib.sha256(pub).hexdigest(),
        rights_basis="OWNER_DECLARED_ORIGINAL_GENERATOR_PENDING_REVIEW",
        decision="ASSERT_RIGHTS_TO_TRAIN", not_before_utc="2026-11-01T17:00:00Z",
        expires_utc="2026-11-02T00:00:00Z")
    return secret, pub, hashes, receipt
def inspect(secret, pub, hashes, receipt, now=NOW):
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()
    signature = secret.sign(DOMAIN + body).hex()
    return inspect_owner_assertion(receipt_bytes=body, signature_hex=signature,
        owner_public_key=pub, expected_corpus_id="r18-test-only",
        expected_hashes=hashes, now_utc=now)

class OwnerControlledRightsTests(unittest.TestCase):
    def test_one_owner_signature_is_valid_but_never_admits(self):
        private, public, hashes, receipt = fixture()
        result = inspect(private, public, hashes, receipt)
        self.assertEqual(result["status"], PENDING)
        self.assertTrue(result["signature_valid_for_supplied_key"])
        for key in ("owner_key_enrollment_authenticated",
                    "generator_rights_independently_reviewed",
                    "protected_exclusion_custody_authenticated",
                    "host_revocation_and_replay_authenticated",
                    "corpus_rights_admitted", "physical_training_authorized",
                    "protected_evaluation_authorized"):
            self.assertIs(result[key], False)

    def test_owner_withhold_cannot_grant(self):
        p, pub, hashes, c = fixture()
        c["decision"] = "WITHHOLD"
        self.assertEqual(inspect(p, pub, hashes, c)["status"], "OWNER_WITHHELD")

    def test_scope_uses_exact_corpus_family_and_revision(self):
        for field, value in (("corpus_family", "LABYRINTH_SYNTHETIC"),
                             ("base_revision", "0" * 40),
                             ("rights_basis", "LICENSED"),
                             ("purpose", "OTHER_PURPOSE")):
            with self.subTest(field=field):
                p, pub, hashes, c = fixture()
                c[field] = value
                with self.assertRaises(AdmissionError):
                    inspect(p, pub, hashes, c)
    def test_every_train_dev_protected_generator_commitment_must_match(self):
        for field in HASH_KEYS:
            with self.subTest(field=field):
                p, pub, hashes, c = fixture()
                c[field] = "a" * 64
                with self.assertRaisesRegex(AdmissionError, "hash mismatch"):
                    inspect(p, pub, hashes, c)

    def test_expired_claim_fails(self):
        p, pub, hashes, c = fixture()
        with self.assertRaisesRegex(AdmissionError, "expired"):
            inspect(p, pub, hashes, c, datetime(2026, 11, 3, tzinfo=timezone.utc))

    def test_key_and_signature_cannot_be_substituted(self):
        p, pub, hashes, c = fixture()
        forged_public = fixture()[1]
        with self.assertRaises(AdmissionError):
            inspect(p, forged_public, hashes, c)
        body = json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaises(AdmissionError):
            inspect_owner_assertion(receipt_bytes=body, signature_hex="0" * 128,
                owner_public_key=pub, expected_corpus_id="r18-test-only",
                expected_hashes=hashes, now_utc=NOW)

    def test_invalid_canonical_or_claim_shape_rejected(self):
        p, pub, hashes, c = fixture()
        c["unexpected"] = "grant"
        with self.assertRaises(AdmissionError):
            inspect(p, pub, hashes, c)
        p, pub, hashes, c = fixture()
        body = json.dumps(c, indent=2, sort_keys=True).encode()
        with self.assertRaisesRegex(AdmissionError, "canonical"):
            inspect_owner_assertion(receipt_bytes=body,
                signature_hex=p.sign(DOMAIN + body).hex(), owner_public_key=pub,
                expected_corpus_id="r18-test-only",
                expected_hashes=hashes, now_utc=NOW)

if __name__ == "__main__":
    unittest.main()
