"""Source-only new-lineage arithmetic tests, never protected DEV evidence."""

import hashlib
import json
import re
import subprocess
import sys
import unittest

from trainingday.synthetic_arithmetic import (
    DEV_COUNT, SCHEMA, TRAIN_COUNT, SyntheticArithmeticError,
    _private_dev_rows, commitment, train_bytes, train_rows,
)


# Public UNIT-TEST nonce, intentionally NOT an evaluator secret.
TEST_NONCE = bytes(range(32))
TRAIN_SHA256 = "1ba1b03adbfeccebaadd47674d8624ea80e6b4dabf8678d1d7084c88055b1f63"


def _arithmetically_correct(row):
    match = re.fullmatch(r"Compute ([0-9]+) ([+*/-]) ([0-9]+)\.", row["prompt"])
    if not match:
        return False
    left, op, right = int(match[1]), match[2], int(match[3])
    expected = {
        "+": lambda: left + right,
        "-": lambda: left - right,
        "*": lambda: left * right,
        "/": lambda: left // right if right and left % right == 0 else None,
    }[op]()
    return expected is not None and row["answer"] == str(expected)


class NewArithmeticSourceTests(unittest.TestCase):
    def test_public_train_is_exactly_1024_unique_correct_rows(self):
        rows = train_rows()
        self.assertEqual(len(rows), TRAIN_COUNT)
        self.assertEqual(TRAIN_COUNT, 1024)
        self.assertEqual(len({row["id"] for row in rows}), TRAIN_COUNT)
        self.assertEqual(len({row["prompt"] for row in rows}), TRAIN_COUNT)
        self.assertTrue(all(row["split"] == "train" and _arithmetically_correct(row)
                            for row in rows))
        raw = train_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), TRAIN_SHA256)
        self.assertEqual([json.loads(line) for line in raw.splitlines()], rows)
        self.assertEqual(raw, train_bytes())

    def test_private_dev_test_fixture_is_disjoint_but_not_authenticated(self):
        train = train_rows()
        dev = _private_dev_rows(TEST_NONCE)
        self.assertEqual(len(dev), DEV_COUNT)
        self.assertEqual(len({row["id"] for row in dev}), DEV_COUNT)
        self.assertEqual(len({row["prompt"] for row in dev}), DEV_COUNT)
        self.assertTrue(all(row["split"] == "dev" and _arithmetically_correct(row)
                            for row in dev))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in dev})
        self.assertFalse({row["prompt"] for row in train} & {row["prompt"] for row in dev})
        self.assertNotEqual(dev, _private_dev_rows(bytes(reversed(range(32)))))

    def test_commitment_exposes_no_dev_data_or_authority(self):
        result = commitment(TEST_NONCE)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["train_sha256"], TRAIN_SHA256)
        self.assertEqual(result["train_count"], 1024)
        self.assertEqual(result["dev_count"], 64)
        self.assertFalse(result["r185_corpus_replacement"])
        self.assertFalse(result["private_dev_rows_returned"])
        for denied in ("rights_holder_authenticated", "data_rights_granted",
                       "protected_evaluator_custody_verified", "gpu_lease_granted",
                       "authorizes_training", "nonce_entropy_independently_verified"):
            self.assertFalse(result[denied], denied)
        self.assertNotIn("nonce", result)
        self.assertNotIn("dev_rows", result)
        self.assertNotIn("dev_answers", result)
        self.assertEqual(len(result["private_dev_sha256"]), 64)
        self.assertNotEqual(result["private_dev_sha256"],
                            commitment(bytes(reversed(range(32))))["private_dev_sha256"])

    def test_nonce_shape_and_import_side_effect_guard(self):
        for bad in (b"", b"\x00" * 32, b"x" * 31, "text", bytearray(range(32))):
            with self.subTest(bad_type=type(bad).__name__), self.assertRaises(
                    SyntheticArithmeticError):
                commitment(bad)
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import trainingday.synthetic_arithmetic; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
