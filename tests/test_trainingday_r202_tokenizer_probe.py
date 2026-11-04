"""Source tests for an opt-in pinned-local tokenizer probe (NOT Qwen fit)."""

import unittest

from trainingday import cognitive_curriculum
from trainingday import r202_tokenizer_probe as candidate


class TokenizerProbeSourceTests(unittest.TestCase):
    def test_stats_are_deterministic_and_bounded(self):
        # A deliberately fake word-count encoder tests aggregation mechanics.
        # It is NOT an actual Qwen tokenizer or model-compatibility receipt.
        def fake_encoder(value):
            return list(range(len(value.split())))

        data = candidate.profile_counts(fake_encoder)
        self.assertEqual(data["row_count"], 1536)
        self.assertEqual(data["corpus_sha256"],
                         cognitive_curriculum.source_commitment()["train_sha256"])
        self.assertIs(data["authorizes_training"], False)
        self.assertIs(data["omits_chat_template_and_special_tokens"], True)
        for key in ("prompt", "prompt_plus_answer"):
            global_stats = data["overall"][key]
            self.assertGreater(global_stats["min"], 0)
            self.assertGreaterEqual(global_stats["max"], global_stats["p95_nearest_rank"])
            self.assertGreaterEqual(global_stats["p95_nearest_rank"], global_stats["median"])
            self.assertEqual(set(global_stats["exceeds_budget"]), {"64", "128", "256"})
        self.assertEqual(set(data["families"]), set(cognitive_curriculum.FAMILIES))
        self.assertEqual(data, candidate.profile_counts(fake_encoder))

    def test_invalid_encoder_and_non_train_rejected(self):
        with self.assertRaises(candidate.TokenizerProbeError):
            candidate.profile_counts(lambda _: ["wrong"])
        with self.assertRaises(candidate.TokenizerProbeError):
            candidate.profile_counts(lambda _: [1], rows=[])
        row = dict(cognitive_curriculum.train_row("math_logic", 0))
        row["split"] = "dev"
        with self.assertRaises(candidate.TokenizerProbeError):
            candidate.profile_counts(lambda _: [1], rows=[row])


if __name__ == "__main__":
    unittest.main()
