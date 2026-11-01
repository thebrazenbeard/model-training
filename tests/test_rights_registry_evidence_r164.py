"""Adversarial synthetic registry tests; never real dataset-rights grants."""

from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timezone

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from trainingday.admission import AdmissionError
from trainingday.rights_pair_evidence_r155 import DOMAIN as RIGHTS_DOMAIN
from trainingday.rights_pair_evidence_r155 import PINNED_BASE_REVISION, SCHEMA as RIGHTS_SCHEMA
from trainingday.rights_pair_evidence_r155 import REJECT_RESULT
from trainingday.rights_registry_evidence_r164 import (
    DOMAIN, SCHEMA, UNTRUSTED_RESULT, inspect_registry_bound_rights_pair,
)


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "optional Ed25519 verifier unavailable")
class RegistryEvidenceR164Tests(unittest.TestCase):
    def setUp(self):
        self.issuer = Ed25519PrivateKey.generate()
        self.holder = Ed25519PrivateKey.generate()
        self.steward = Ed25519PrivateKey.generate()
        self.issuer_pub = self._pub(self.issuer)
        self.holder_pub = self._pub(self.holder)
        self.steward_pub = self._pub(self.steward)
        self.now = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
        self.registry = {
            "schema": SCHEMA,
            "issuer_id": "synthetic-registry-issuer",
            "issuer_key_sha256": self._hash(self.issuer_pub),
            "generation": 5,
            "previous_snapshot_sha256": "1" * 64,
            "not_before_utc": "2026-11-01T00:00:00Z",
            "expires_utc": "2026-11-02T00:00:00Z",
            "subjects": [
                {"role": "RIGHTS_HOLDER", "id": "synthetic-holder",
                 "key_sha256": self._hash(self.holder_pub)},
                {"role": "DATA_STEWARD", "id": "synthetic-steward",
                 "key_sha256": self._hash(self.steward_pub)},
            ],
            "revoked_key_sha256": [],
        }
        self.claim = {
            "schema": RIGHTS_SCHEMA,
            "corpus_family": "DOLLY_V4",
            "corpus_id": "synthetic-corpus",
            "manifest_sha256": "a" * 64,
            "train_sha256": "b" * 64,
            "dev_sha256": "c" * 64,
            "protected_exclusion_sha256": "d" * 64,
            "license_expression": "CC-BY-SA-3.0",
            "license_evidence_sha256": "e" * 64,
            "base_revision": PINNED_BASE_REVISION,
            "purpose": "QWEN35_VERA_TRAININGDAY",
            "decision": "ADMIT",
            "rights_holder_id": "synthetic-holder",
            "rights_holder_key_sha256": self._hash(self.holder_pub),
            "steward_id": "synthetic-steward",
            "steward_key_sha256": self._hash(self.steward_pub),
            "revocation_snapshot_sha256": "f" * 64,
            "not_before_utc": "2026-11-01T00:00:00Z",
            "expires_utc": "2026-11-02T00:00:00Z",
        }

    @staticmethod
    def _pub(key):
        return key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    @staticmethod
    def _hash(value: bytes):
        return hashlib.sha256(value).hexdigest()

    def _call(self, *, registry=None, claim=None, **changes):
        reg_bytes = canonical(self.registry if registry is None else registry)
        rights = dict(self.claim if claim is None else claim)
        rights.setdefault("revocation_snapshot_sha256", self._hash(reg_bytes))
        # The canonical pair must commit the *actual* registry bytes unless the
        # test intentionally supplies a conflicting digest.
        if claim is None:
            rights["revocation_snapshot_sha256"] = self._hash(reg_bytes)
        receipt = canonical(rights)
        kwargs = dict(
            registry_bytes=reg_bytes,
            registry_signature_hex=self.issuer.sign(DOMAIN + reg_bytes).hex(),
            issuer_public_key=self.issuer_pub,
            rights_receipt_bytes=receipt,
            holder_signature_hex=self.holder.sign(RIGHTS_DOMAIN + receipt).hex(),
            steward_signature_hex=self.steward.sign(RIGHTS_DOMAIN + receipt).hex(),
            holder_public_key=self.holder_pub,
            steward_public_key=self.steward_pub,
            expected_corpus_id=self.claim["corpus_id"],
            expected_manifest_sha256=self.claim["manifest_sha256"],
            expected_train_sha256=self.claim["train_sha256"],
            expected_dev_sha256=self.claim["dev_sha256"],
            expected_protected_exclusion_sha256=self.claim["protected_exclusion_sha256"],
            required_generation_floor=5,
            required_previous_snapshot_sha256="1" * 64,
            now_utc=self.now,
        )
        kwargs.update(changes)
        return inspect_registry_bound_rights_pair(**kwargs)

    def test_valid_self_issued_registry_and_rights_pair_stays_denied(self):
        result = self._call()
        self.assertEqual(result["status"], UNTRUSTED_RESULT)
        self.assertTrue(result["registry_signature_valid_for_supplied_issuer_key"])
        self.assertTrue(result["registry_role_bindings_checked"])
        for field in ("host_registry_issuer_authenticated",
                      "host_registry_rollback_fence_authenticated",
                      "corpus_rights_admitted", "physical_training_authorized",
                      "protected_evaluation_authorized"):
            self.assertFalse(result[field], field)

    def test_signed_reject_stays_reject(self):
        claim = dict(self.claim, decision="REJECT",
                     revocation_snapshot_sha256=self._hash(canonical(self.registry)))
        self.assertEqual(self._call(claim=claim)["status"], REJECT_RESULT)

    def test_unsigned_root_substitution_rejected(self):
        other = self._pub(Ed25519PrivateKey.generate())
        with self.assertRaisesRegex(AdmissionError, "signer key binding"):
            self._call(issuer_public_key=other)

    def test_role_swap_rejected(self):
        registry = dict(self.registry)
        registry["subjects"] = [
            {**r, "role": "DATA_STEWARD" if r["role"] == "RIGHTS_HOLDER"
             else "RIGHTS_HOLDER"} for r in self.registry["subjects"]
        ]
        with self.assertRaisesRegex(AdmissionError, "wrong role"):
            self._call(registry=registry)

    def test_revoked_signer_or_issuer_rejected(self):
        for fingerprint in (self._hash(self.holder_pub),
                            self._hash(self.steward_pub), self._hash(self.issuer_pub)):
            with self.subTest(fingerprint=fingerprint):
                registry = dict(self.registry, revoked_key_sha256=[fingerprint])
                with self.assertRaisesRegex(AdmissionError, "revoked"):
                    self._call(registry=registry)

    def test_stale_generation_previous_digest_and_clock_rejected(self):
        with self.assertRaisesRegex(AdmissionError, "rollback"):
            self._call(required_generation_floor=6)
        with self.assertRaisesRegex(AdmissionError, "previous digest mismatch"):
            self._call(required_previous_snapshot_sha256="2" * 64)
        with self.assertRaisesRegex(AdmissionError, "chain shape"):
            self._call(registry=dict(self.registry, previous_snapshot_sha256=None),
                       required_previous_snapshot_sha256=None)
        with self.assertRaisesRegex(AdmissionError, "expired"):
            self._call(now_utc=datetime(2026, 11, 3, tzinfo=timezone.utc))

    def test_duplicate_identity_or_reused_key_rejected(self):
        for extra in (
            dict(self.registry["subjects"][0]),
            dict(self.registry["subjects"][0], id="another-holder"),
        ):
            with self.subTest(extra=extra):
                registry = dict(self.registry, subjects=self.registry["subjects"] + [extra])
                with self.assertRaisesRegex(AdmissionError, "duplicate"):
                    self._call(registry=registry)

    def test_rights_snapshot_digest_not_interchangeable(self):
        claim = dict(self.claim, revocation_snapshot_sha256="0" * 64)
        with self.assertRaisesRegex(AdmissionError, "not bound"):
            self._call(claim=claim)

    def test_boolean_generation_and_noncanonical_registry_rejected(self):
        registry = dict(self.registry, generation=True)
        with self.assertRaisesRegex(AdmissionError, "generation invalid"):
            self._call(registry=registry)
        raw = json.dumps(self.registry, indent=2).encode("utf-8")
        with self.assertRaisesRegex(AdmissionError, "canonical"):
            self._call(registry_bytes=raw)


if __name__ == "__main__":
    unittest.main()
