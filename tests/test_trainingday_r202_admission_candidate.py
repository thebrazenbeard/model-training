"""Source-only admission-request tests: no actual consent, keys, or DEV data."""

import json
import unittest

from trainingday.r202_admission_candidate import (
    CORPUS_ID, SCHEMA, TRAIN_SCHEMA, TRAIN_COUNT, TRAIN_SHA256,
    GENERATOR_GIT_BLOB_SHA1, MODEL_ID, MODEL_REVISION, PURPOSE,
    REQUIRED_EXCLUSIONS, R202AdmissionEvidenceError,
    inspect_r202_admission_request_candidate,
)


# Synthetic digests are NOT a custodian, sealed DEV set, or exclusion evidence.
FAKE_DEV = "1" * 64
FAKE_EXCLUSIONS = "2" * 64


def fixture():
    return {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "train_schema": TRAIN_SCHEMA,
        "train_count": TRAIN_COUNT,
        "train_sha256": TRAIN_SHA256,
        "generator_git_blob_sha1": GENERATOR_GIT_BLOB_SHA1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "purpose": PURPOSE,
        "rights_holder_identity": "SYNTHETIC_OWNER_NOT_PATRICK",
        "data_steward_identity": "SYNTHETIC_OWNER_NOT_PATRICK",
        "rights_basis": "ORIGINAL_CREATED_OR_COMMISSIONED",
        "decision": "ASSERT_SCOPE_AND_REQUEST_REVIEW",
        "exclusions": sorted(REQUIRED_EXCLUSIONS),
        "private_dev_commitment_sha256": FAKE_DEV,
        "protected_exclusion_receipt_sha256": FAKE_EXCLUSIONS,
    }


def json_bytes(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def inspect(record):
    return inspect_r202_admission_request_candidate(
        request_bytes=json_bytes(record),
        expected_private_dev_commitment_sha256=FAKE_DEV,
        expected_protected_exclusion_receipt_sha256=FAKE_EXCLUSIONS,
    )


class R202AdmissionRequestTest(unittest.TestCase):
    def test_synthetic_same_owner_and_steward_never_grants_training(self):
        result = inspect(fixture())
        self.assertEqual(result["status"], "UNSIGNED_EXACT_SCOPE_REQUEST_HOLD")
        self.assertTrue(result["exact_public_train_scope_checked"])
        for key, value in result.items():
            if key.endswith(("authenticated", "qualified", "verified", "granted", "admitted")) \
                    or key == "authorizes_training":
                self.assertFalse(value, key)

    def test_withhold_stays_hold(self):
        subject = fixture()
        subject["decision"] = "WITHHOLD"
        self.assertEqual(inspect(subject)["status"], "UNSIGNED_OWNER_WITHHOLD_HOLD")

    def test_wrong_public_dataset_digest_is_rejected(self):
        subject = fixture()
        subject["train_sha256"] = "3" * 64
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "subject mismatch"):
            inspect(subject)

    def test_wrong_source_blob_is_rejected(self):
        subject = fixture()
        subject["generator_git_blob_sha1"] = "4" * 40
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "subject mismatch"):
            inspect(subject)

    def test_commercial_scope_is_rejected(self):
        subject = fixture()
        subject["purpose"] = "ANY_COMMERCIAL_MODEL_TRAINING"
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "subject mismatch"):
            inspect(subject)

    def test_different_model_revision_rejected(self):
        subject = fixture()
        subject["model_revision"] = "5" * 40
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "subject mismatch"):
            inspect(subject)

    def test_missing_exclusion_rejected(self):
        subject = fixture()
        subject["exclusions"].remove("protected_dev_and_test_material")
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "required exclusions"):
            inspect(subject)

    def test_wrong_evaluator_commitment_rejected(self):
        subject = fixture()
        subject["private_dev_commitment_sha256"] = "6" * 64
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "protected commitment"):
            inspect(subject)

    def test_missing_exclusion_receipt_rejected(self):
        subject = fixture()
        subject["protected_exclusion_receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "protected commitment"):
            inspect(subject)

    def test_anonymous_rights_holder_rejected(self):
        subject = fixture()
        subject["rights_holder_identity"] = " "
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "identity"):
            inspect(subject)

    def test_duplicate_json_field_rejected(self):
        valid = json_bytes(fixture()).decode("utf-8")
        injected = valid.replace('"purpose":', '"purpose":"bad","purpose":', 1)
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "duplicate"):
            inspect_r202_admission_request_candidate(
                request_bytes=injected.encode("utf-8"),
                expected_private_dev_commitment_sha256=FAKE_DEV,
                expected_protected_exclusion_receipt_sha256=FAKE_EXCLUSIONS)

    def test_unsigned_request_not_authority_by_construction(self):
        subject = fixture()
        subject["authorizes_training"] = True
        with self.assertRaisesRegex(R202AdmissionEvidenceError, "wrong request fields"):
            inspect(subject)


if __name__ == "__main__":
    unittest.main()
