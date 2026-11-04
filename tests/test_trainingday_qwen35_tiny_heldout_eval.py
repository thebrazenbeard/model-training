"""Dependency-free split and completion-mask falsifiers for a CPU experiment."""

from __future__ import annotations

import unittest

from trainingday import qwen35_tiny_heldout_eval as trial


class HeldoutControlSourceTests(unittest.TestCase):
    def test_predeclared_replication_seeds_include_original_five(self):
        self.assertEqual(trial.SEEDS, tuple(range(35032, 35052)))

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

    def test_top1_probe_never_supplies_answer_as_completion(self):
        for seed in trial.SEEDS:
            train, heldout = trial._pairs(seed)
            for pair in train + heldout:
                prefix, expected = trial._completion_prefix(pair)
                self.assertEqual(len(prefix), 5)
                self.assertEqual(prefix[-1], trial.EQUAL)
                self.assertEqual(expected, trial.NUMBER_OFFSET + sum(pair))
                self.assertEqual(prefix[2:4], [trial.NUMBER_OFFSET + x for x in pair])

    def test_commuted_heldout_uses_only_unseen_reversed_prompts(self):
        for seed in trial.SEEDS:
            train, heldout = trial._pairs(seed)
            training_prefixes = {tuple(trial._completion_prefix(pair)[0]) for pair in train}
            shifted = [pair for pair in heldout if pair[0] != pair[1]]
            self.assertTrue(shifted)
            for pair in shifted:
                canonical, canonical_expected = trial._completion_prefix(pair)
                commuted, commuted_expected = trial._completion_prefix(pair, commuted=True)
                self.assertNotEqual(canonical, commuted)
                self.assertEqual(canonical_expected, commuted_expected)
                self.assertEqual(commuted[2:4], canonical[3:1:-1])
                self.assertNotIn(tuple(commuted), training_prefixes)
        with self.assertRaises(AssertionError):
            trial._completion_prefix((4, 4), commuted=True)

    def test_top1_metric_rejects_empty_probe(self):
        with self.assertRaises(AssertionError):
            trial._answer_top1(object(), [], object())


if __name__ == "__main__":
    unittest.main()
