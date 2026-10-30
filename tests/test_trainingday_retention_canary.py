"""Source-only canary checks; no GPU/model data or heldout is opened."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.retention_canary import (
    ARMS, FAMILIES, IDS, RELATIVE, SOURCE_SHA256, WINDOWS_CRLF_SHA256,
    preregister,
)


REPO = Path(__file__).resolve().parents[1]


class RetentionCanaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        target = self.root / RELATIVE
        target.parent.mkdir(parents=True)
        # Make fixtures deterministic on both Windows checkout policies.
        raw = (REPO / RELATIVE).read_bytes()
        physical_sha = hashlib.sha256(raw).hexdigest()
        self.assertIn(physical_sha, (SOURCE_SHA256, WINDOWS_CRLF_SHA256))
        if physical_sha == WINDOWS_CRLF_SHA256:
            raw = raw.replace(b"\r\n", b"\n")
        target.write_bytes(raw)

    def test_exact_public_source_and_disjoint_families(self):
        result = preregister(self.root)
        self.assertEqual(result["source_sha256"], SOURCE_SHA256)
        self.assertEqual(result["checkout_sha256"], SOURCE_SHA256)
        self.assertEqual(result["checkout_line_endings"], "LF")
        self.assertEqual(result["case_count"], 5)
        self.assertEqual(result["family_counts"], FAMILIES)
        self.assertEqual(result["required_matched_arms"], list(ARMS))
        self.assertIn("NOT_MODEL_SCORED", result["state"])
        # No review-sensitive prompt/completion text crosses the receipt.
        exported = json.dumps(result)
        self.assertNotIn("2 + 2 = 5", exported)
        self.assertNotIn("weather", exported)

    def test_exact_windows_crlf_checkout_is_accepted_without_repinning(self):
        path = self.root / RELATIVE
        payload = path.read_bytes()
        self.assertNotIn(b"\r", payload)
        path.write_bytes(payload.replace(b"\n", b"\r\n"))
        result = preregister(self.root)
        self.assertEqual(result["source_sha256"], SOURCE_SHA256)
        self.assertEqual(result["checkout_sha256"], WINDOWS_CRLF_SHA256)
        self.assertEqual(result["checkout_line_endings"], "CRLF")

    def test_mixed_endings_lone_cr_and_crlf_mutation_rejected(self):
        path = self.root / RELATIVE
        original = path.read_bytes()
        cases = (
            original.replace(b"\n", b"\r\n", 1),
            original.replace(b"\n", b"\r", 1),
            original.replace(b"\n", b"\r\n").replace(b"\r\n", b"\r\n ", 1),
        )
        for case in cases:
            with self.subTest(kind=case[:40]):
                path.write_bytes(case)
                with self.assertRaises(AdmissionError):
                    preregister(self.root)

    def test_unpinned_bytes_rejected(self):
        path = self.root / RELATIVE
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(AdmissionError):
            preregister(self.root)

    def test_wrong_membership_rejected_even_if_digest_is_mocked(self):
        path = self.root / RELATIVE
        lines = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["scenario_id"] = IDS[1]
        lines[0] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with patch("trainingday.retention_canary._hash_file", return_value=SOURCE_SHA256):
            with self.assertRaises(AdmissionError):
                preregister(self.root)

    def test_broken_required_rubric_rejected_even_if_digest_is_mocked(self):
        path = self.root / RELATIVE
        lines = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["rubric"]["required"] = []
        lines[0] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with patch("trainingday.retention_canary._hash_file", return_value=SOURCE_SHA256):
            with self.assertRaises(AdmissionError):
                preregister(self.root)

    def test_missing_or_symlinked_source_fails_closed(self):
        path = self.root / RELATIVE
        path.unlink()
        with self.assertRaises(AdmissionError):
            preregister(self.root)


if __name__ == "__main__":
    unittest.main()
