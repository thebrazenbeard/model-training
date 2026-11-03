"""No-Torch negative tests of short-retention synthetic checkpoint receipts."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday import qwen35_full32_trl_cross_job_artifact_cpu as custody


class CrossJobArtifactReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="trainingday-cross-job-negative-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / custody.witness.MANIFEST).write_bytes(b"source-manifest")
        digest = hashlib.sha256(b"source-manifest").hexdigest()
        self.receipt = {"schema": custody.SCHEMA, "writer_run_id": "123456",
                        "writer_job": "writer", "writer_pid": 333,
                        "checkpoint_manifest_sha256": digest,
                        "direct_adapter_sha256": "a" * 64,
                        "direct_optimizer_sha256": "b" * 64,
                        "direct_logits_sha256": "c" * 64,
                        "first_step_loss": 1.0, "second_step_loss": 0.5,
                        "training_authorized": False}
        self._save()

    def _save(self):
        (self.root / custody.RECEIPT).write_text(json.dumps(self.receipt), encoding="utf-8")

    def _read(self):
        with patch.object(custody.witness, "_validate_checkpoint", return_value={}):
            return custody._read_receipt(self.root)

    def test_valid_receipt_references_checked_manifest(self):
        self.assertEqual(self._read(), self.receipt)

    def test_corrupt_manifest_rejected_even_with_expected_receipt(self):
        (self.root / custody.witness.MANIFEST).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "manifest corrupted"):
            self._read()

    def test_positive_grant_rejected(self):
        self.receipt["training_authorized"] = True
        self._save()
        with self.assertRaisesRegex(ValueError, "cannot grant"):
            self._read()

    def test_wrong_writer_job_run_id_and_pid_rejected(self):
        original = copy.deepcopy(self.receipt)
        for field, bad in (("writer_job", "reader"), ("writer_run_id", "N/A"),
                           ("writer_pid", -1)):
            with self.subTest(field=field):
                self.receipt = copy.deepcopy(original)
                self.receipt[field] = bad
                self._save()
                with self.assertRaisesRegex(ValueError, "writer identity invalid"):
                    self._read()

    def test_wrong_digest_and_unexpected_field_rejected(self):
        self.receipt["direct_adapter_sha256"] = "NOT_SHA256"
        self._save()
        with self.assertRaisesRegex(ValueError, "digest malformed"):
            self._read()
        self.receipt["direct_adapter_sha256"] = "a" * 64
        self.receipt["untrusted_grant"] = True
        self._save()
        with self.assertRaisesRegex(ValueError, "schema changed"):
            self._read()

    def test_oversized_receipt_rejected_before_deserialization(self):
        (self.root / custody.RECEIPT).write_text(" " * 5000, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "oversized"):
            self._read()

    def test_reader_must_be_distinct_job_same_run(self):
        with patch.object(custody.witness, "_validate_checkpoint", return_value={}):
            for run_id, job in (("123456", "writer"), ("654321", "reader")):
                with self.subTest(run_id=run_id, job=job):
                    with patch.dict(os.environ, {"GITHUB_RUN_ID": run_id, "GITHUB_JOB": job}):
                        with self.assertRaisesRegex(ValueError, "distinct job in the SAME"):
                            custody.read(self.root)


if __name__ == "__main__":
    unittest.main()
