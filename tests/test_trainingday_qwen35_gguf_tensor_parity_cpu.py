"""Independent numeric negative controls for the post-cast GGUF tensor oracle."""

import unittest

import numpy as np

from trainingday.qwen35_gguf_tensor_parity_cpu import (
    assert_exact_post_cast_tensor, selected_tensor_coverage_manifest,
    transform_hf_values,
)


class TensorParityTests(unittest.TestCase):
    def test_six_tensor_witness_explicitly_disclaims_full_coverage(self):
        manifest = selected_tensor_coverage_manifest(730, 722)
        self.assertEqual(manifest["scope"], "SIX_SELECTED_TENSOR_PAIRS_ONLY")
        self.assertEqual(manifest["checked_pair_count"], 6)
        self.assertEqual(manifest["unverified_hf_tensor_count"], 724)
        self.assertEqual(manifest["unverified_gguf_tensor_count"], 716)
        self.assertEqual(manifest["full_tensor_conversion_parity"], "NOT_TESTED")
        with self.assertRaisesRegex(AssertionError, "larger"):
            selected_tensor_coverage_manifest(6, 6)

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

    def test_zero_centered_rmsnorm_requires_plus_one(self):
        source = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        converted = np.array([1.0, 1.25, 0.5], dtype=np.float32)
        assert_exact_post_cast_tensor(source, converted, "norm", "float32", "rmsnorm_plus_one")
        with self.assertRaisesRegex(AssertionError, "mismatching"):
            assert_exact_post_cast_tensor(source, source, "norm", "float32", "rmsnorm_plus_one")

    def test_gdn_v_grouped_to_tiled_preserves_qk_and_swaps_v_groups(self):
        source = np.repeat(np.arange(128, dtype=np.float32)[:, None], 64, axis=1)
        converted = transform_hf_values(source, "gdn_v_tiled")
        np.testing.assert_array_equal(converted[:64], source[:64])
        # Original V groups: 64:80,80:96 / 96:112,112:128.
        # Tiled order: 64:80,96:112 / 80:96,112:128.
        self.assertEqual(converted[64, 0], 64)
        self.assertEqual(converted[80, 0], 96)
        self.assertEqual(converted[96, 0], 80)
        self.assertEqual(converted[112, 0], 112)
        assert_exact_post_cast_tensor(source, converted.astype(np.float16),
                                      "qkv", "float16", "gdn_v_tiled")
        with self.assertRaisesRegex(AssertionError, "mismatching"):
            assert_exact_post_cast_tensor(source, source.astype(np.float16),
                                          "qkv", "float16", "gdn_v_tiled")

    def test_unknown_representation_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unexpected GGUF"):
            transform_hf_values(np.zeros(2, dtype=np.float32), "skip-check")


if __name__ == "__main__":
    unittest.main()
