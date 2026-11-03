"""Source contracts for real-batch R16 TRL checkpoint sample continuity."""

import unittest

from trainingday import r16_trl_cpu_sft_lifecycle as lifecycle


class SampleWitnessContractTests(unittest.TestCase):
    def test_synthetic_examples_are_distinct_and_completion_masked(self):
        rows = [lifecycle._synthetic_example(i) for i in range(32)]
        self.assertEqual(len({tuple(row["input_ids"]) for row in rows}), 32)
        for index, row in enumerate(rows):
            with self.subTest(index=index):
                self.assertEqual(row["input_ids"][1], index + 4)
                self.assertEqual(row["attention_mask"], [1] * 8)
                self.assertEqual(row["completion_mask"], [0] * 4 + [1] * 4)
                self.assertEqual(row["labels"][:4], [-100] * 4)

    def test_bad_synthetic_row_identity_is_rejected(self):
        for invalid in (True, -1, 32, 3.5, "0", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    lifecycle._synthetic_example(invalid)

    def test_full_sequence_continuity_and_mismatch_sensitivity(self):
        control = list(range(16))
        self.assertEqual(
            lifecycle._verify_sample_continuity(control[:8], control[8:], control),
            control,
        )
        perturbed = control[8:].copy()
        perturbed[0], perturbed[1] = perturbed[1], perturbed[0]
        with self.assertRaisesRegex(AssertionError, "sample sequence"):
            lifecycle._verify_sample_continuity(control[:8], perturbed, control)

    def test_uniform_or_invalid_witnesses_cannot_pass(self):
        for invalid in ([0] * 16, [0] * 15, list(range(15)) + [True],
                        list(range(15)) + [32]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(AssertionError):
                    lifecycle._verify_sample_continuity(
                        invalid[:8], invalid[8:], invalid,
                    )


if __name__ == "__main__":
    unittest.main()
