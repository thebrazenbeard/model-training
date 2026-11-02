"""Synthetic owner/private bridge matching; no protected data or host grants."""

import copy
import unittest
from datetime import datetime, timezone

from trainingday.admission import AdmissionError
from trainingday.owner_exact_subject_v2 import DOMAIN
from trainingday.owner_private_consistency_v1 import (
    STATUS, inspect_owner_private_consistency_v1,
)
from tests.test_owner_exact_subject_v2 import canonical, fixture as owner_fixture
from tests import test_private_dev_commitment_v1 as private_fixture


NOW = datetime(2026, 11, 2, 18, tzinfo=timezone.utc)
MATCHED = ("manifest_sha256", "train_sha256", "train_content_sha256",
           "selected_id_v2_sha256", "generator_source_sha256", "dev_sha256",
           "protected_exclusion_sha256")


class OwnerPrivateConsistencyTests(unittest.TestCase):
    def setUp(self):
        source = private_fixture.PrivateDevCommitmentTests()
        source.setUp()
        self.report = source.make()
        self.key, self.public, self.subject, self.claim = owner_fixture()
        for field in MATCHED:
            self.claim[field] = self.report[field]

    def inspect(self, *, claim=None, report=None, packet_sha=None, key=None, public=None):
        claim = self.claim if claim is None else claim
        report = self.report if report is None else report
        key = self.key if key is None else key
        public = self.public if public is None else public
        raw = canonical(claim)
        return inspect_owner_private_consistency_v1(
            owner_receipt_bytes=raw,
            owner_signature_hex=key.sign(DOMAIN + raw).hex(),
            owner_public_key=public,
            private_bridge_report=report,
            expected_frozen_packet_sha256=(packet_sha if packet_sha is not None
                                           else self.claim["frozen_training_packet_sha256"]),
            expected_owner_identity="synthetic-owner",
            minimum_owner_key_epoch=7,
            now_utc=NOW,
        )

    def test_matching_two_source_evidence_records_still_hard_hold(self):
        result = self.inspect()
        self.assertEqual(result["status"], STATUS)
        self.assertTrue(result["caller_supplied_hashes_consistent"])
        for field in ("owner_identity_authenticated", "owner_rights_admitted",
                      "private_evaluator_custody_authenticated", "private_eval_unseal_authorized",
                      "gpu_lease_granted", "physical_training_authorized",
                      "independent_source_authenticated"):
            self.assertIs(result[field], False, field)
        self.assertNotIn("dev-1", str(result))
        self.assertNotIn("Start with", str(result))
        self.assertNotIn("gold", str(result))

    def test_each_shared_digest_mismatch_is_rejected_independently(self):
        for field in MATCHED:
            report = dict(self.report)
            report[field] = "f" * 64 if report[field] != "f" * 64 else "e" * 64
            with self.subTest(field=field), self.assertRaises(AdmissionError):
                self.inspect(report=report)

    def test_packet_digest_is_distinct_host_input_not_in_private_bridge(self):
        with self.assertRaises(AdmissionError):
            self.inspect(packet_sha="0" * 64)

    def test_forged_consistent_key_hashes_never_grant_authority(self):
        # The owner caller can sign a modified claim and supply matching
        # source-only report hashes; this remains structural, NOT admission.
        claim, report = copy.deepcopy(self.claim), copy.deepcopy(self.report)
        claim["train_content_sha256"] = report["train_content_sha256"] = "b" * 64
        result = self.inspect(claim=claim, report=report)
        self.assertEqual(result["status"], STATUS)
        self.assertIs(result["independent_source_authenticated"], False)
        self.assertIs(result["physical_training_authorized"], False)

    def test_withheld_or_malformed_owner_claim_rejected(self):
        claim = dict(self.claim)
        claim["decision"] = "WITHHOLD"
        with self.assertRaises(AdmissionError):
            self.inspect(claim=claim)
        claim = dict(self.claim)
        claim["corpus_id"] = "LABYRINTH_COGNITIVE"
        with self.assertRaises(AdmissionError):
            self.inspect(claim=claim)

    def test_broker_cannot_launder_positive_evaluator_flags(self):
        for field in ("independent_custody_authenticated", "train_content_commitment_authenticated",
                      "owner_identity_authenticated", "source_use_rights_verified",
                      "host_ratification_authenticated", "authorizes_training",
                      "authorizes_eval_unseal", "private_content_returned"):
            report = dict(self.report)
            report[field] = True
            with self.subTest(field=field), self.assertRaises(AdmissionError):
                self.inspect(report=report)

    def test_private_content_extra_keys_and_source_status_rejected(self):
        report = dict(self.report)
        report["dev_rows"] = ["secret synthetic prompt"]
        with self.assertRaises(AdmissionError):
            self.inspect(report=report)
        report = dict(self.report)
        report["status"] = "ADMITTED"
        with self.assertRaises(AdmissionError):
            self.inspect(report=report)

    def test_replayed_or_foreign_owner_key_and_epoch_rejected(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        other = Ed25519PrivateKey.generate()
        pub = other.public_key().public_bytes_raw()
        with self.assertRaises(AdmissionError):
            self.inspect(key=other, public=pub)
        claim = dict(self.claim)
        claim["owner_key_epoch"] = 6
        with self.assertRaises(AdmissionError):
            self.inspect(claim=claim)


if __name__ == "__main__":
    unittest.main()
