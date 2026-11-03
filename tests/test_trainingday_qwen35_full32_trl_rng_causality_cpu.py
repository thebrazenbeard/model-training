"""Dependency-free negative tests for the stochastic CPU continuation contract."""

import unittest

from trainingday.qwen35_full32_trl_rng_causality_cpu import assert_causal_resume


def _fixture():
    a, b, c, d = (x * 64 for x in "abcd")
    direct = {"adapter_sha256": c, "optimizer_state_sha256": c,
              "logits_sha256": c}
    good = {**direct, "after_resume_optimizer_steps": [2]}
    bad = {"adapter_sha256": d, "optimizer_state_sha256": d,
           "logits_sha256": d, "after_resume_optimizer_steps": [2]}
    return {"initial": a, "first": b, "direct": direct,
            "restored": good, "rng_omitted": bad}


class CausalityAdmissionTest(unittest.TestCase):
    def test_positive_control(self):
        assert_causal_resume(**_fixture())

    def test_no_second_step_update_rejected(self):
        f = _fixture()
        f["direct"]["adapter_sha256"] = f["first"]
        f["restored"]["adapter_sha256"] = f["first"]
        with self.assertRaisesRegex(AssertionError, "second optimizer step"):
            assert_causal_resume(**f)

    def test_rng_omission_causal_null_rejected(self):
        f = _fixture()
        f["rng_omitted"]["adapter_sha256"] = f["direct"]["adapter_sha256"]
        with self.assertRaisesRegex(AssertionError, "omitting RNG"):
            assert_causal_resume(**f)

    def test_fresh_child_optimizer_mismatch_rejected(self):
        f = _fixture()
        f["restored"]["optimizer_state_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "fresh child"):
            assert_causal_resume(**f)

    def test_step_count_missing_rejected(self):
        f = _fixture()
        f["rng_omitted"]["after_resume_optimizer_steps"] = [1]
        with self.assertRaisesRegex(AssertionError, "real second AdamW step"):
            assert_causal_resume(**f)


if __name__ == "__main__":
    unittest.main()
