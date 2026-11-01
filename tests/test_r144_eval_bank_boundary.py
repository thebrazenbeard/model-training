"""CPU-only R18 source receipt versus private-evaluation boundary checks."""

import unittest

from trainingday.r18_benchmark import BenchmarkError, _training_index


class R18PrivateBankBoundaryTests(unittest.TestCase):
    def test_source_plan_requires_explicit_non_holdout_boolean(self):
        row = {"id": "selected-1", "split": "train", "training_use": "ADMITTED",
               "evaluation_holdout": False, "prompt": "Compute two plus two.",
               "answer": "4"}
        self.assertEqual(len(_training_index([row],
            "ADMITTED_SOURCE_ONLY_NOT_TRAINED")), 1)
        for bad in (None, 0, 1, "false", "true", True):
            with self.subTest(bad=bad):
                with self.assertRaises(BenchmarkError):
                    _training_index([dict(row, evaluation_holdout=bad)],
                                    "ADMITTED_SOURCE_ONLY_NOT_TRAINED")
        missing = dict(row)
        missing.pop("evaluation_holdout")
        with self.assertRaises(BenchmarkError):
            _training_index([missing], "ADMITTED_SOURCE_ONLY_NOT_TRAINED")


if __name__ == "__main__":
    unittest.main()
