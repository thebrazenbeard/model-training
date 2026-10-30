"""Source-only canary checks; no GPU/model data or heldout is opened."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.retention_canary import (
    ARMS, FAMILIES, IDS, RELATIVE, SOURCE_SHA256, preregister,
)


REPO = Path(__file__).resolve().parents[1]


class RetentionCanaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        target = self.root / RELATIVE
        target.parent.mkdir(parents=True)
        shutil.copyfile(REPO / RELATIVE, target)

    def test_exact_public_source_and_disjoint_families(self):
        result = preregister(self.root)
        self.assertEqual(result["source_sha256"], SOURCE_SHA256)
        self.assertEqual(result["case_count"], 5)
        self.assertEqual(result["family_counts"], FAMILIES)
        self.assertEqual(result["required_matched_arms"], list(ARMS))
        self.assertIn("NOT_MODEL_SCORED", result["state"])
        # No review-sensitive prompt/completion text crosses the receipt.
        exported = json.dumps(result)
        self.assertNotIn("2 + 2 = 5", exported)
        self.assertNotIn("weather", exported)

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
