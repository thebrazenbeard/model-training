"""Independent negatives for HF↔GGUF tiny synthetic logprob comparison."""

import unittest

import numpy as np

from trainingday.qwen35_gguf_logprobs_cpu import (
    SAMPLES, assert_sampled_logprobs, extract_top_logprobs,
)


def fixtures():
    hf = np.log(np.ones(256, dtype=np.float64) / 256)
    top = [{"id": k, "logprob": float(hf[k])} for k in range(SAMPLES)]
    return hf, top, [0]


class LogprobTests(unittest.TestCase):
    def test_exact_logprob_agrees(self):
        hf, top, generated = fixtures()
        receipt = assert_sampled_logprobs(hf, top, generated)
        self.assertEqual(receipt["compared_tokens"], SAMPLES)
        self.assertEqual(receipt["max_abs_logprob_delta"], 0)

    def test_reported_probability_corruption_rejected(self):
        hf, top, generated = fixtures()
        top[4]["logprob"] += 1.0
        with self.assertRaisesRegex(AssertionError, "divergence"):
            assert_sampled_logprobs(hf, top, generated)

    def test_wrong_token_id_rejected(self):
        hf, top, generated = fixtures()
        top[4]["id"] = 9999
        with self.assertRaisesRegex(AssertionError, "out-of-range"):
            assert_sampled_logprobs(hf, top, generated)

    def test_duplicate_token_id_rejected(self):
        hf, top, generated = fixtures()
        top[4]["id"] = 3
        with self.assertRaisesRegex(AssertionError, "duplicated"):
            assert_sampled_logprobs(hf, top, generated)

    def test_no_probability_witness_rejected(self):
        hf, _, generated = fixtures()
        with self.assertRaisesRegex(AssertionError, "missing"):
            assert_sampled_logprobs(hf, [], generated)

    def test_nonfinite_probability_rejected(self):
        hf, top, generated = fixtures()
        top[4]["logprob"] = float("nan")
        with self.assertRaisesRegex(AssertionError, "invalid"):
            assert_sampled_logprobs(hf, top, generated)

    def test_wrong_topk_membership_rejected_even_when_probabilities_match(self):
        # Deterministic falsifier: a server can return perfect numbers from
        # HF's non-top tokens while disagreeing on the actual argmax/top-k.
        logits = np.array([10.0] + [0.0] * 255)
        hf = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        reported = [{"id": i, "logprob": float(hf[i])}
                    for i in range(100, 100 + SAMPLES)]
        with self.assertRaisesRegex(AssertionError, "top-k token ID rank"):
            assert_sampled_logprobs(hf, reported, [0])

    def test_wrong_topk_rank_rejected(self):
        logits = np.array(list(range(256, 0, -1)), dtype=np.float64)
        hf = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        reported = [{"id": i, "logprob": float(hf[i])} for i in range(SAMPLES)]
        reported[0], reported[1] = reported[1], reported[0]
        with self.assertRaisesRegex(AssertionError, "top-k token ID rank"):
            assert_sampled_logprobs(hf, reported, [0])

    def test_wrong_generation_rejected_even_when_top12_agrees(self):
        hf, top, _ = fixtures()
        with self.assertRaisesRegex(AssertionError, "generated token"):
            assert_sampled_logprobs(hf, top, [42])

    def test_missing_generation_rejected(self):
        hf, top, _ = fixtures()
        with self.assertRaisesRegex(AssertionError, "missing a single"):
            assert_sampled_logprobs(hf, top, None)

    def test_pinned_completion_probabilities_field_accepted(self):
        _, top, _ = fixtures()
        self.assertEqual(extract_top_logprobs(
            {"completion_probabilities": [{"top_logprobs": top}]}
        ), top)

    def test_outdated_docs_prose_probs_field_not_silently_accepted(self):
        _, top, _ = fixtures()
        with self.assertRaisesRegex(AssertionError, "completion_probabilities"):
            extract_top_logprobs({"probs": [{"top_logprobs": top}]})


if __name__ == "__main__":
    unittest.main()
