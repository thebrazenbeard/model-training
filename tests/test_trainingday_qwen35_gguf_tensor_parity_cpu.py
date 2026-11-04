"""Independent numeric negative controls for the post-cast GGUF tensor oracle."""

import unittest

import numpy as np

from trainingday.qwen35_gguf_tensor_parity_cpu import assert_exact_post_cast_tensor


class TensorParityTests(unittest.TestCase):
    def test_exact_float16_cast_matches(self):
        source = np.array([[0.123456, -0.789123]], dtype=np.float32)
        assert_exact_post_cast_tensor(source, source.astype(np.float16), "tensor", "float16")

    def test_one_changed_value_rejected(self):
        source = np.array([[0.125, 0.75]], dtype=np.float32)
        corrupt = source.astype(np.float16)
        corrupt[0, 1] = np.float16(0.5)
        with self.assertRaisesRegex(AssertionError, "1 mismatching"):
            assert_exact_post_cast_tensor(source, corrupt, "tensor", "float16")

    def test_equal_element_count_but_wrong_shape_rejected(self):
        source = np.zeros((2, 3), dtype=np.float32)
        wrong = np.zeros((3, 2), dtype=np.float16)
        with self.assertRaisesRegex(AssertionError, "HF shape"):
            assert_exact_post_cast_tensor(source, wrong, "tensor", "float16")

    def test_wrong_dtype_rejected(self):
        source = np.array([1.0], dtype=np.float32)
        with self.assertRaisesRegex(AssertionError, "GGUF dtype"):
            assert_exact_post_cast_tensor(source, source.copy(), "tensor", "float16")

    def test_nonfinite_source_rejected(self):
        source = np.array([float("nan")], dtype=np.float32)
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            assert_exact_post_cast_tensor(source, source.astype(np.float16), "tensor", "float16")


if __name__ == "__main__":
    unittest.main()
