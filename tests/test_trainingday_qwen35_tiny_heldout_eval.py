"""Dependency-free split and completion-mask falsifiers for a CPU experiment."""

from __future__ import annotations

import unittest

from trainingday import qwen35_tiny_heldout_eval as trial


class HeldoutControlSourceTests(unittest.TestCase):
    def test_exactly_disjoint_canonical_pair_generators(self):
        for seed in trial.SEEDS:
            train, withheld = trial._pairs(seed)
            self.assertEqual(len(train) + len(withheld), 55)
            self.assertFalse(set(train) & set(withheld))
            self.assertTrue(all(a <= b for a, b in train + withheld))
            self.assertEqual(len(set(train + withheld)), 55)
            self.assertEqual((train, withheld), trial._pairs(seed))

    def test_heldout_generator_differs_between_seeds(self):
        self.assertNotEqual(trial._pairs(trial.SEEDS[0])[1],
                            trial._pairs(trial.SEEDS[1])[1])

    def test_only_answer_token_supervised(self):
        for seed in trial.SEEDS:
            for pair in sum((list(x) for x in trial._pairs(seed)), []):
                ids, labels = trial._row(pair)
                self.assertEqual(len(ids), 7)
                self.assertEqual(labels, [-100] * 5 + [ids[5], -100])
                self.assertEqual(ids[5], trial.NUMBER_OFFSET + sum(pair))
                self.assertEqual(ids[6], trial.EOS)

    def test_shuffle_changes_only_feedback_not_prompt(self):
        pair = (3, 7)
        normal, labels = trial._row(pair)
        shuffled, altered = trial._row(pair, answer_override=2)
        self.assertEqual(normal[:5], shuffled[:5])
        self.assertNotEqual(labels[5], altered[5])
        self.assertEqual(labels[:5] + labels[-1:], altered[:5] + altered[-1:])

    def test_bad_feedback_is_rejected(self):
        with self.assertRaises(AssertionError):
            trial._row((8, 9), answer_override=19)


if __name__ == "__main__":
    unittest.main()
