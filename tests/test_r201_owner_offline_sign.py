"""Owner-operated R201 offline signing tests; synthetic keys NEVER enroll."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday.r201_owner_claim_candidate import (
    DOMAIN, ClaimEvidenceError, inspect_r201_owner_claim_candidate,
)
from trainingday.r201_owner_offline_sign import main, sign_owner_claim_offline

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat,
    )
except ImportError:
    Ed25519PrivateKey = None


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography unavailable")
class OwnerOfflineSignTest(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()  # synthetic, in-memory only
        self.dev = hashlib.sha256(b"fake DEV protected value").hexdigest()
        self.exclusion = hashlib.sha256(b"fake exclusions").hexdigest()
        self.now = datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc)
        self.fields = dict(
            private_key=self.key, owner_identity="synthetic-owner",
            dev_sha256=self.dev, exclusion_sha256=self.exclusion,
            epoch=4, not_before_utc="2026-11-03T00:00:00Z",
            expires_utc="2026-11-04T23:00:00Z",
            decision="ASSERT_RIGHTS_TO_TRAIN", now_utc=self.now,
        )

    def test_real_sign_verify_still_false_all_grants(self):
        package = sign_owner_claim_offline(**self.fields)
        wire = package["claim_bytes"]
        signature = bytes.fromhex(package["detached_signature_hex"])
        public = bytes.fromhex(package["public_key_hex"])
        self.key.public_key().verify(signature, DOMAIN + wire)
        claim = json.loads(wire)
        self.assertEqual(claim["private_dev_commitment_sha256"], self.dev)
        self.assertEqual(claim["protected_exclusion_receipt_sha256"], self.exclusion)
        self.assertEqual(hashlib.sha256(wire).hexdigest(), package["claim_sha256"])
        evidence = inspect_r201_owner_claim_candidate(
            claim_bytes=wire, detached_signature_hex=signature.hex(),
            supplied_owner_public_key=public,
            expected_owner_identity="synthetic-owner",
            expected_private_dev_commitment_sha256=self.dev,
            expected_protected_exclusion_receipt_sha256=self.exclusion,
            minimum_owner_key_epoch=4, now_utc=self.now,
        )
        self.assertTrue(evidence["signature_valid_for_supplied_key"])
        for field in (
            "owner_key_enrollment_authenticated",
            "owner_source_rights_independently_reviewed",
            "protected_evaluator_custody_authenticated",
            "host_epoch_and_clock_authenticated", "corpus_rights_admitted",
            "gpu_lease_granted", "physical_training_authorized",
            "protected_evaluation_authorized",
        ):
            with self.subTest(field=field):
                self.assertFalse(evidence[field])

    def test_source_verifier_authority_drift_always_refuses_export(self):
        """One accidental true or missing flag means no signed receipt."""
        package = sign_owner_claim_offline(**self.fields)
        canonical = inspect_r201_owner_claim_candidate(
            claim_bytes=package["claim_bytes"],
            detached_signature_hex=package["detached_signature_hex"],
            supplied_owner_public_key=bytes.fromhex(package["public_key_hex"]),
            expected_owner_identity="synthetic-owner",
            expected_private_dev_commitment_sha256=self.dev,
            expected_protected_exclusion_receipt_sha256=self.exclusion,
            minimum_owner_key_epoch=4, now_utc=self.now,
        )
        for flag in (
            "owner_key_enrollment_authenticated",
            "owner_source_rights_independently_reviewed",
            "protected_evaluator_custody_authenticated",
            "host_epoch_and_clock_authenticated", "corpus_rights_admitted",
            "gpu_lease_granted", "physical_training_authorized",
            "protected_evaluation_authorized",
        ):
            with self.subTest(field=flag):
                escalated = dict(canonical, **{flag: True})
                with patch("trainingday.r201_owner_offline_sign.inspect_r201_owner_claim_candidate",
                           return_value=escalated):
                    with self.assertRaisesRegex(ClaimEvidenceError, "denial contract changed"):
                        sign_owner_claim_offline(**self.fields)
                omitted = dict(canonical)
                del omitted[flag]
                with patch("trainingday.r201_owner_offline_sign.inspect_r201_owner_claim_candidate",
                           return_value=omitted):
                    with self.assertRaisesRegex(ClaimEvidenceError, "denial contract changed"):
                        sign_owner_claim_offline(**self.fields)

    def test_withhold_stays_hold(self):
        package = sign_owner_claim_offline(**dict(self.fields, decision="WITHHOLD"))
        self.assertEqual(json.loads(package["claim_bytes"])["decision"], "WITHHOLD")

    def test_wrong_key_type_missing_and_reused_commitments_refuse(self):
        for change in (
            dict(private_key=b"fake private key"),
            dict(dev_sha256="0" * 64),
            dict(exclusion_sha256="not-a-hash"),
            dict(exclusion_sha256=self.dev),
            dict(epoch=True),
            dict(epoch=0),
            dict(owner_identity="synthetic\u200downer"),
        ):
            with self.subTest(change=repr(change)):
                with self.assertRaises(ClaimEvidenceError):
                    sign_owner_claim_offline(**dict(self.fields, **change))

    def test_bad_time_and_ambiguous_identity_refuse_before_sign(self):
        for change in (
            dict(not_before_utc="2026-12-01T00:00:00Z"),
            dict(expires_utc="2026-11-03T00:00:00Z"),
            dict(now_utc=datetime(2026, 11, 3, 23, 0)),
            dict(owner_identity="owner\ud800"),
            dict(owner_identity=" synthetic-owner"),
        ):
            with self.subTest(change=repr(change)):
                with self.assertRaises(ClaimEvidenceError):
                    sign_owner_claim_offline(**dict(self.fields, **change))

    def test_cli_requires_explicit_ack_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            place = Path(tmp)
            selected = place / "operator-chosen-existing.pem"
            selected.write_bytes(self.key.private_bytes(
                Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
            out = place / "signed-request"
            argv = [
                "--existing-key-pem", str(selected),
                "--owner-identity", "synthetic-owner",
                "--private-dev-commitment-sha256", self.dev,
                "--protected-exclusion-receipt-sha256", self.exclusion,
                "--owner-key-epoch", "4",
                "--not-before-utc", "2000-01-01T00:00:00Z",
                "--expires-utc", "2099-12-31T00:00:00Z",
                "--decision", "ASSERT_RIGHTS_TO_TRAIN",
                "--output-prefix", str(out),
            ]
            # Fixed valid interval includes the actual clock, never a test-only clock.
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertFalse(any(place.glob("signed-request.*")))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv + ["--ack-owner-rights"]), 0)
            for suffix in ("claim.json", "signature.hex", "public_key.hex", "receipt.json"):
                self.assertTrue((place / ("signed-request." + suffix)).is_file())
            receipt = json.loads((place / "signed-request.receipt.json").read_text())
            self.assertFalse(receipt["all_training_authority"])
            self.assertNotIn("PRIVATE KEY", "".join(
                p.read_text(errors="ignore") for p in place.glob("signed-request.*")))
            with self.assertRaises(SystemExit):
                main(argv + ["--ack-owner-rights"])


if __name__ == "__main__":
    unittest.main()
