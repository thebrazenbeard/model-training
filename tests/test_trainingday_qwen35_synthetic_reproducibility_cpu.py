"""Offline corruption controls for the synthetic reproducibility receipt."""

import unittest

from trainingday.qwen35_synthetic_reproducibility_cpu import assert_reproduced


class ReproducibilityTest(unittest.TestCase):
    def setUp(self):
        self.first = {"file_sha256": "a" * 64,
                      "tensor_values_sha256": "b" * 64,
                      "tensor_count": 427, "bytes": 123}

    def test_identical_receipts(self):
        assert_reproduced(self.first, dict(self.first))

    def test_tensor_values_drift_rejected(self):
        with self.assertRaisesRegex(AssertionError, "tensor values drifted"):
            assert_reproduced(self.first, {**self.first,
                                          "tensor_values_sha256": "c" * 64})

    def test_serialization_drift_rejected(self):
        with self.assertRaisesRegex(AssertionError, "serialization bytes drifted"):
            assert_reproduced(self.first, {**self.first,
                                          "file_sha256": "d" * 64})

    def test_tensor_inventory_drift_rejected(self):
        with self.assertRaisesRegex(AssertionError, "inventory changed"):
            assert_reproduced(self.first, {**self.first, "tensor_count": 426})


if __name__ == "__main__":
    unittest.main()
