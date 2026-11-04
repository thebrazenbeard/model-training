"""Independent negatives for HF↔GGUF tiny synthetic logprob comparison."""

import unittest

import numpy as np

from trainingday.qwen35_gguf_logprobs_cpu import (
    SAMPLES, assert_sampled_logprobs, extract_top_logprobs,
)


def fixtures():
    hf = np.log(np.ones(256, dtype=np.float64) / 256)
    top = [{"id": k, "logprob": float(hf[k])} for k in range(SAMPLES)]
    return hf, top


class LogprobTests(unittest.TestCase):
    def test_exact_logprob_agrees(self):
        hf, top = fixtures()
        receipt = assert_sampled_logprobs(hf, top)
        self.assertEqual(receipt["compared_tokens"], SAMPLES)
        self.assertEqual(receipt["max_abs_logprob_delta"], 0)

    def test_reported_probability_corruption_rejected(self):
        hf, top = fixtures()
        top[4]["logprob"] += 1.0
        with self.assertRaisesRegex(AssertionError, "divergence"):
            assert_sampled_logprobs(hf, top)

    def test_wrong_token_id_rejected(self):
        hf, top = fixtures()
        top[4]["id"] = 9999
        with self.assertRaisesRegex(AssertionError, "out-of-range"):
            assert_sampled_logprobs(hf, top)

    def test_duplicate_token_id_rejected(self):
        hf, top = fixtures()
        top[4]["id"] = 3
        with self.assertRaisesRegex(AssertionError, "duplicated"):
            assert_sampled_logprobs(hf, top)

    def test_no_probability_witness_rejected(self):
        hf, _ = fixtures()
        with self.assertRaisesRegex(AssertionError, "missing"):
            assert_sampled_logprobs(hf, [])

    def test_nonfinite_probability_rejected(self):
        hf, top = fixtures()
        top[4]["logprob"] = float("nan")
        with self.assertRaisesRegex(AssertionError, "invalid"):
            assert_sampled_logprobs(hf, top)

    def test_pinned_completion_probabilities_field_accepted(self):
        _, top = fixtures()
        self.assertEqual(extract_top_logprobs(
            {"completion_probabilities": [{"top_logprobs": top}]}
        ), top)

    def test_outdated_docs_prose_probs_field_not_silently_accepted(self):
        _, top = fixtures()
        with self.assertRaisesRegex(AssertionError, "completion_probabilities"):
            extract_top_logprobs({"probs": [{"top_logprobs": top}]})


if __name__ == "__main__":
    unittest.main()
