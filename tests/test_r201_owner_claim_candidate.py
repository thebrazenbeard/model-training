"""Synthetic R201 owner-claim integrity tests: never host/data authority."""

from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timezone

from trainingday.r201_owner_claim_candidate import (
    CORPUS_ID, ClaimEvidenceError as AdmissionError, DOMAIN, MODEL_ID, MODEL_REVISION, PURPOSE, RIGHTS_BASIS,
    SCHEMA, TRAIN_SHA256, TRAIN_SCHEMA, inspect_r201_owner_claim_candidate,
)
from trainingday.calc_curriculum import (
    CALC_REPOSITORY, CALC_REVISION, CALC_SYMBOLIC_SHA256, TRAIN_COUNT,
)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    Ed25519PrivateKey = None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography unavailable")
class R201OwnerClaimCandidateTest(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.dev = hashlib.sha256(b"synthetic-secret-evaluator-commitment").hexdigest()
        self.excl = hashlib.sha256(b"synthetic-separate-exclusion-receipt").hexdigest()
        self.claim = {
            "schema": SCHEMA, "corpus_id": CORPUS_ID,
            "train_schema": TRAIN_SCHEMA, "train_count": TRAIN_COUNT,
            "train_sha256": TRAIN_SHA256,
            "calc_repository": CALC_REPOSITORY,
            "calc_revision": CALC_REVISION,
            "calc_symbolic_sha256": CALC_SYMBOLIC_SHA256,
            "private_dev_commitment_sha256": self.dev,
            "protected_exclusion_receipt_sha256": self.excl,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "purpose": PURPOSE, "owner_identity": "synthetic-test-owner",
            "owner_key_sha256": hashlib.sha256(self.public).hexdigest(),
            "owner_key_epoch": 5, "rights_basis": RIGHTS_BASIS,
            "decision": "ASSERT_RIGHTS_TO_TRAIN",
            "not_before_utc": "2026-11-01T00:00:00Z",
            "expires_utc": "2026-11-10T00:00:00Z",
        }

    def inspect(self, claim=None, *, signature=None, raw=None, **overrides):
        value = self.claim if claim is None else claim
        wire = canonical(value) if raw is None else raw
        signature_hex = (self.private.sign(DOMAIN + wire).hex()
                         if signature is None else signature)
        params = {
            "claim_bytes": wire, "detached_signature_hex": signature_hex,
            "supplied_owner_public_key": self.public,
            "expected_owner_identity": "synthetic-test-owner",
            "expected_private_dev_commitment_sha256": self.dev,
            "expected_protected_exclusion_receipt_sha256": self.excl,
            "minimum_owner_key_epoch": 5,
            "now_utc": datetime(2026, 11, 3, tzinfo=timezone.utc),
        }
        params.update(overrides)
        return inspect_r201_owner_claim_candidate(**params)

    def test_valid_self_issued_signature_never_admits(self):
        result = self.inspect()
        self.assertTrue(result["signature_valid_for_supplied_key"])
        self.assertTrue(result["r201_exact_subject_checked"])
        self.assertFalse(any(result[field] for field in (
            "owner_key_enrollment_authenticated",
            "owner_source_rights_independently_reviewed",
            "protected_evaluator_custody_authenticated",
            "host_epoch_and_clock_authenticated", "corpus_rights_admitted",
            "gpu_lease_granted", "physical_training_authorized",
            "protected_evaluation_authorized")))

    def test_owner_withhold_never_admits(self):
        self.claim["decision"] = "WITHHOLD"
        self.assertEqual(self.inspect()["status"], "OWNER_WITHHELD_SOURCE_ONLY_HOLD")

    def test_r18_wrong_corpus_and_purpose_refuse_even_if_resigned(self):
        for key, bad in (
            ("schema", "OWNER_CONTROLLED_R18_EXACT_SUBJECT_V2"),
            ("corpus_id", "R18_ARITHMETIC"),
            ("train_count", 1000), ("train_sha256", "a" * 64),
            ("calc_revision", "a" * 40), ("calc_symbolic_sha256", "f" * 64),
            ("model_id", "some-derivative/Qwen3.5-4B"),
            ("model_revision", "a" * 40),
            ("purpose", "UNRELATED_TRAINING"),
        ):
            with self.subTest(field=key):
                evil = dict(self.claim, **{key: bad})
                with self.assertRaises(AdmissionError):
                    self.inspect(evil)

    def test_protected_eval_commitments_required_but_untrusted(self):
        with self.assertRaises(AdmissionError):
            self.inspect(expected_private_dev_commitment_sha256="b" * 64)
        with self.assertRaises(AdmissionError):
            self.inspect(expected_protected_exclusion_receipt_sha256="b" * 64)
        with self.assertRaises(AdmissionError):
            self.inspect(dict(self.claim, private_dev_commitment_sha256="0" * 64))
        # A forged coherent new digest *can* pass source integrity and remains HOLD.
        invented = "c" * 64
        self.assertFalse(self.inspect(dict(self.claim, private_dev_commitment_sha256=invented),
                                      expected_private_dev_commitment_sha256=invented)
                         ["protected_evaluator_custody_authenticated"])

    def test_modified_wire_wrong_key_and_wrong_domain_refuse(self):
        wire = canonical(self.claim)
        with self.assertRaises(AdmissionError):
            self.inspect(raw=wire.replace(b'"schema":', b'"schema": '))
        with self.assertRaises(AdmissionError):
            self.inspect(signature=self.private.sign(b"R18\x00" + wire).hex())
        with self.assertRaises(AdmissionError):
            self.inspect(supplied_owner_public_key=b"a" * 32)

    def test_duplicate_and_ambiguous_keys_refuse(self):
        wire = canonical(self.claim)
        duplicate = wire.replace(b'"schema":', b'"schema":"wrong","schema":')
        with self.assertRaises(AdmissionError):
            self.inspect(raw=duplicate)
        mixed = dict(self.claim)
        mixed["SCHEMA"] = SCHEMA
        with self.assertRaises(AdmissionError):
            self.inspect(mixed)

    def test_signed_invisible_owner_identity_refuses(self):
        for bad in ("synthetic\u200dtest-owner", "synthetic\ntest-owner"):
            with self.subTest(identity=repr(bad)):
                forged = dict(self.claim, owner_identity=bad)
                with self.assertRaises(AdmissionError):
                    self.inspect(forged, expected_owner_identity=bad)

    def test_escaped_lone_surrogates_reject_with_declared_error(self):
        original = canonical(self.claim)
        for key, original_value in (
            ("owner_identity", "synthetic-test-owner"),
            ("rights_basis", RIGHTS_BASIS),
        ):
            for malformed in (b"\\ud800", b"\\udfff"):
                with self.subTest(key=key, escaped=malformed):
                    needle = (f'"{key}":"{original_value}"').encode("ascii")
                    replacement = b'"' + key.encode("ascii") + b'":"' + malformed + b'"'
                    raw = original.replace(needle, replacement)
                    self.assertNotEqual(raw, original)
                    with self.assertRaisesRegex(AdmissionError, "invalid Unicode"):
                        self.inspect(raw=raw)

    def test_valid_surrogate_pair_remains_canonicality_rejection(self):
        original = canonical(self.claim)
        raw = original.replace(b'"owner_identity":"synthetic-test-owner"',
                               b'"owner_identity":"\\ud83d\\ude00"')
        # Valid escape sequence decodes to a Unicode scalar value. Its
        # canonical wire spelling must still be unescaped UTF-8, so rejection
        # here is not the invalid-Unicode exception path.
        with self.assertRaisesRegex(AdmissionError, "canonical JSON"):
            self.inspect(raw=raw)

    def test_overflowed_json_exponent_uses_declared_evidence_error(self):
        original = canonical(self.claim)
        needle = b'"owner_key_epoch":5'
        self.assertIn(needle, original)
        for exponent in (b"1e999", b"-1e999"):
            with self.subTest(exponent=exponent):
                raw = original.replace(needle, b'"owner_key_epoch":' + exponent)
                with self.assertRaisesRegex(AdmissionError, "nonfinite numeric value"):
                    self.inspect(raw=raw)

    def test_replay_epoch_expiry_and_boolean_epoch_refuse(self):
        with self.assertRaises(AdmissionError):
            self.inspect(minimum_owner_key_epoch=6)
        with self.assertRaises(AdmissionError):
            self.inspect(dict(self.claim, owner_key_epoch=True))
        with self.assertRaises(AdmissionError):
            self.inspect(now_utc=datetime(2026, 12, 1, tzinfo=timezone.utc))
        with self.assertRaises(AdmissionError):
            self.inspect(now_utc=datetime(2026, 11, 3))


if __name__ == "__main__":
    unittest.main()
