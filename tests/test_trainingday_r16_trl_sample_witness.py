"""Source contracts for real-batch R16 TRL checkpoint sample continuity."""

import unittest

from trainingday import r16_trl_cpu_sft_lifecycle as lifecycle


class SampleWitnessContractTests(unittest.TestCase):
    def test_synthetic_examples_are_distinct_and_completion_masked(self):
        rows = [lifecycle._synthetic_example(i) for i in range(32)]
        self.assertEqual(len({tuple(row["input_ids"]) for row in rows}), 32)
        self.assertEqual(len({tuple(row["labels"]) for row in rows}), 32)
        for index, row in enumerate(rows):
            with self.subTest(index=index):
                self.assertEqual(row["input_ids"][1], index + 4)
                self.assertEqual(row["attention_mask"], [1] * 8)
                self.assertEqual(row["completion_mask"], [0] * 4 + [1] * 4)
                self.assertEqual(row["labels"][:4], [-100] * 4)
                self.assertEqual(row["labels"][4:], row["input_ids"][4:])

    def test_full_effective_batch_is_bound_to_independent_source_row(self):
        row = lifecycle._synthetic_example(7)
        digest = lifecycle._effective_batch_sha256(
            7, row["input_ids"], row["attention_mask"], row["labels"])
        self.assertEqual(len(digest), 64)
        for field in ("input_ids", "attention_mask", "labels"):
            changed = {k: v.copy() for k, v in row.items()}
            changed[field][5] += 1
            with self.subTest(field=field):
                with self.assertRaisesRegex(AssertionError, "EFFECTIVE_BATCH_MISMATCH"):
                    lifecycle._effective_batch_sha256(
                        7, changed["input_ids"], changed["attention_mask"],
                        changed["labels"])

    def test_full_batch_sequence_rejects_resume_only_label_corruption(self):
        original = [lifecycle._synthetic_example(i) for i in range(16)]
        hashes = [lifecycle._effective_batch_sha256(
            i, row["input_ids"], row["attention_mask"], row["labels"])
            for i, row in enumerate(original)]
        self.assertEqual(lifecycle._verify_effective_batch_continuity(
            hashes[:8], hashes[8:], hashes), hashes)
        altered = hashes[8:].copy()
        altered[0] = "0" * 64 if altered[0] != "0" * 64 else "1" * 64
        with self.assertRaisesRegex(AssertionError, "effective batch sequence"):
            lifecycle._verify_effective_batch_continuity(
                hashes[:8], altered, hashes)

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
