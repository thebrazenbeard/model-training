"""Synthetic signatures prove integrity only, never real rights/host authority."""

from __future__ import annotations

import hashlib
import json
import unittest
from unittest import mock
from datetime import datetime, timezone

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from trainingday.admission import AdmissionError
from trainingday.rights_pair_evidence_r155 import (
    DOMAIN, PINNED_BASE_REVISION, REJECT_RESULT, SCHEMA, UNTRUSTED_RESULT,
    _verify_ed25519, inspect_rights_pair,
)


class MissingCryptoFailClosedTests(unittest.TestCase):
    def test_dependency_absence_is_not_a_signature_bypass(self):
        with mock.patch.dict("sys.modules", {"cryptography.exceptions": None}):
            with self.assertRaisesRegex(AdmissionError, "dependency unavailable"):
                _verify_ed25519(b"x" * 32, "0" * 128, b"test")


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "optional Ed25519 verifier unavailable")
class RightsPairEvidenceR155Tests(unittest.TestCase):
    def setUp(self):
        self.holder = Ed25519PrivateKey.generate()
        self.steward = Ed25519PrivateKey.generate()
        self.holder_pub = self.holder.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.steward_pub = self.steward.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.claim = {
            "schema": SCHEMA,
            "corpus_family": "DOLLY_V4",
            "corpus_id": "dolly-v4-exact",
            "manifest_sha256": "a" * 64,
            "train_sha256": "b" * 64,
            "dev_sha256": "c" * 64,
            "protected_exclusion_sha256": "d" * 64,
            "license_expression": "CC-BY-SA-3.0",
            "license_evidence_sha256": "e" * 64,
            "base_revision": PINNED_BASE_REVISION,
            "purpose": "QWEN35_VERA_TRAININGDAY",
            "decision": "ADMIT",
            "rights_holder_id": "external-rights-holder",
            "rights_holder_key_sha256": hashlib.sha256(self.holder_pub).hexdigest(),
            "steward_id": "independent-data-steward",
            "steward_key_sha256": hashlib.sha256(self.steward_pub).hexdigest(),
            "revocation_snapshot_sha256": "f" * 64,
            "not_before_utc": "2026-11-01T00:00:00Z",
            "expires_utc": "2026-11-02T00:00:00Z",
        }
        self.now = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)

    def _call(self, claim=None, **changes):
        claim = dict(self.claim if claim is None else claim)
        raw = json.dumps(claim, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        kwargs = {
            "receipt_bytes": raw,
            "holder_signature_hex": self.holder.sign(DOMAIN + raw).hex(),
            "steward_signature_hex": self.steward.sign(DOMAIN + raw).hex(),
            "holder_public_key": self.holder_pub,
            "steward_public_key": self.steward_pub,
            "expected_corpus_id": self.claim["corpus_id"],
            "expected_manifest_sha256": self.claim["manifest_sha256"],
            "expected_train_sha256": self.claim["train_sha256"],
            "expected_dev_sha256": self.claim["dev_sha256"],
            "expected_protected_exclusion_sha256": self.claim["protected_exclusion_sha256"],
            "revoked_key_sha256": frozenset(),
            "now_utc": self.now,
        }
        kwargs.update(changes)
        return inspect_rights_pair(**kwargs)

    def test_valid_synthetic_pair_cannot_authorize_external_data(self):
        report = self._call()
        self.assertEqual(report["status"], UNTRUSTED_RESULT)
        self.assertEqual(report["signed_decision"], "ADMIT")
        self.assertTrue(report["detached_signatures_verify_for_supplied_keys"])
        self.assertFalse(report["external_key_registry_authenticated"])
        self.assertFalse(report["revocation_freshness_authenticated"])
        self.assertFalse(report["rights_holder_independence_authenticated"])
        self.assertFalse(report["corpus_rights_admitted"])
        self.assertFalse(report["physical_training_authorized"])
        self.assertFalse(report["protected_evaluation_authorized"])

    def test_signed_reject_remains_rejection(self):
        claim = dict(self.claim, decision="REJECT")
        report = self._call(claim)
        self.assertEqual(report["status"], REJECT_RESULT)
        self.assertFalse(report["corpus_rights_admitted"])

    def test_unsigned_byte_tamper_rejected(self):
        raw = json.dumps(self.claim, sort_keys=True, separators=(",", ":")).encode()
        mutated = raw.replace(b'"ADMIT"', b'"REJECT"')
        with self.assertRaisesRegex(AdmissionError, "signature"):
            self._call(receipt_bytes=mutated)

    def test_wrong_exact_shard_or_manifest_binding_rejected(self):
        for field in ("manifest", "train", "dev", "protected_exclusion"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(AdmissionError, "not bound"):
                    self._call(**{f"expected_{field}_sha256": "0" * 64})

    def test_revoked_holder_or_steward_cannot_pass(self):
        for key in (self.holder_pub, self.steward_pub):
            with self.assertRaisesRegex(AdmissionError, "revoked"):
                self._call(revoked_key_sha256=frozenset({hashlib.sha256(key).hexdigest()}))

    def test_key_swap_and_fake_identifier_rejected(self):
        with self.assertRaisesRegex(AdmissionError, "reuse"):
            self._call(holder_public_key=self.steward_pub)
        with self.assertRaisesRegex(AdmissionError, "reuse"):
            self._call(steward_public_key=self.holder_pub)
        with self.assertRaisesRegex(AdmissionError, "identities"):
            self._call(dict(self.claim, steward_id="external-rights-holder"))

    def test_expired_future_and_unaware_clock_rejected(self):
        for instant in (datetime(2026, 10, 31, tzinfo=timezone.utc),
                        datetime(2026, 11, 3, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(AdmissionError, "not current"):
                self._call(now_utc=instant)
        with self.assertRaisesRegex(AdmissionError, "timezone"):
            self._call(now_utc=datetime(2026, 11, 1))

    def test_wrong_family_license_base_or_purpose_rejected(self):
        for key, value in (("license_expression", "MIT"),
                           ("base_revision", "0" * 40),
                           ("purpose", "SHARE_EVAL_BANK"),
                           ("corpus_family", "UNKNOWN")):
            with self.subTest(key=key):
                with self.assertRaises(AdmissionError):
                    self._call(dict(self.claim, **{key: value}))

    def test_noncanonical_and_duplicate_receipt_keys_rejected(self):
        raw = json.dumps(self.claim, indent=2, sort_keys=True).encode()
        with self.assertRaisesRegex(AdmissionError, "canonical"):
            self._call(receipt_bytes=raw)
        duplicate = b'{"schema":"X","schema":"Y"}'
        with self.assertRaisesRegex(AdmissionError, "duplicate"):
            self._call(receipt_bytes=duplicate)


if __name__ == "__main__":
    unittest.main()
