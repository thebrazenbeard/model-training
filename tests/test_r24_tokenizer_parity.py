"""Public/synthetic R18 token-preparation witness, no model or corpus."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from trainingday.r24_tokenizer_parity import inspect_tokenizer_parity


class SyntheticTokenizer:
    eos_token = "!"
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True
        return "<u>" + messages[0]["content"] + "</u><a>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) for ch in text]}


class ContextualTailTokenizer(SyntheticTokenizer):
    """Synthetic prefix-stable tokenizer whose suffix depends on context."""

    def __call__(self, text, *, add_special_tokens):
        ids = super().__call__(text, add_special_tokens=add_special_tokens)["input_ids"]
        pivot = text.find("</u><a>")
        if pivot >= 0:
            start = pivot + len("</u><a>")
            for i in range(start, len(ids)):
                ids[i] += 1000
        return {"input_ids": ids}


class BoundaryMergeTokenizer(SyntheticTokenizer):
    """Synthetic merge erases the leftmost prefix token at the seam."""

    def __call__(self, text, *, add_special_tokens):
        ids = super().__call__(text, add_special_tokens=add_special_tokens)["input_ids"]
        if "</u><a>X" in text:
            # Prefix was separately encoded without the next X, and is no
            # longer a token-prefix of this joined encoding.
            ids[len(text.split("</u><a>")[0]) + len("</u><a>") - 1] = 9919
        return {"input_ids": ids}


class InflatingSuffixTokenizer(SyntheticTokenizer):
    """Joined completion is longer while the separately encoded prefix stays stable."""

    def __call__(self, text, *, add_special_tokens):
        ids = super().__call__(text, add_special_tokens=add_special_tokens)["input_ids"]
        marker = "</u><a>"
        start = text.find(marker)
        if start >= 0 and len(text) > start + len(marker):
            ids.extend([7] * 40)
        return {"input_ids": ids}


class MissingEosTokenizer(SyntheticTokenizer):
    eos_token = None

    def __getattribute__(self, name):
        if name == "eos_token":
            raise AttributeError("synthetic eos_token unavailable")
        return super().__getattribute__(name)


class OverflowTokenizer(SyntheticTokenizer):
    def __call__(self, text, *, add_special_tokens):
        raise OverflowError("synthetic tokenizer overflow")


class SourceMutatingTokenizer(SyntheticTokenizer):
    def __init__(self, source_rows):
        self.source_rows = source_rows
        self.mutated = False

    def __call__(self, text, *, add_special_tokens):
        if not self.mutated:
            self.mutated = True
            self.source_rows.pop()
        return super().__call__(text, add_special_tokens=add_special_tokens)


def row(value="2", *, key="synthetic-a", prompt="What is 1+1?"):
    return {"id": key, "prompt": prompt, "response": value}


class TestR24TokenizerParity(unittest.TestCase):
    def test_runners_use_identical_joined_tokens_with_contextual_suffix(self):
        from trainingday.r18_mezo import tokenize_cases
        from trainingday.r18_qlora import _tensorize

        tokenizer = ContextualTailTokenizer()
        selected = [row()]
        mezo, _ = tokenize_cases(tokenizer, selected, 128)
        qlora = _tensorize(tokenizer, selected)
        self.assertEqual(mezo[0]["input_ids"], qlora[0]["input_ids"])
        self.assertEqual(mezo[0]["labels"], qlora[0]["labels"])
        self.assertEqual(mezo[0]["target_tokens"], qlora[0]["target_tokens"])

    def test_malformed_preparer_output_is_structured_hold(self):
        with patch("trainingday.r24_tokenizer_parity._tensorize", return_value=[{}]):
            result = inspect_tokenizer_parity(SyntheticTokenizer(), [row()], 128)
        self.assertEqual(result["reason"], "MALFORMED_PREPARER_OUTPUT")
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["authorizes_training"])

    def test_mutated_caller_cases_do_not_shrink_attested_scope(self):
        source = [row(), row("17", key="synthetic-b")]
        result = inspect_tokenizer_parity(SourceMutatingTokenizer(source), source, 128)
        self.assertEqual(len(source), 1)  # The adversarial side effect did happen.
        self.assertEqual(result["example_count"], 2)
        self.assertEqual(result["supervised_target_tokens"], 5)
        self.assertEqual(result["status"], "PASS_SOURCE_PARITY")

    def test_mismatched_preparer_counts_return_structured_hold(self):
        with patch("trainingday.r24_tokenizer_parity._tensorize", return_value=[]):
            result = inspect_tokenizer_parity(SyntheticTokenizer(), [row()], 128)
        self.assertEqual(result["reason"], "PREPARER_EXAMPLE_COUNT_MISMATCH")
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["authorizes_training"])

    def test_missing_eos_token_returns_structured_hold(self):
        result = inspect_tokenizer_parity(MissingEosTokenizer(), [row()], 128)
        self.assertEqual(result["reason"], "MEZO_PREPARATION_REJECTED")
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["authorizes_training"])

    def test_tokenizer_overflow_returns_structured_hold(self):
        result = inspect_tokenizer_parity(OverflowTokenizer(), [row()], 128)
        self.assertEqual(result["reason"], "MEZO_PREPARATION_REJECTED")
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["authorizes_training"])

    def test_identical_char_tokenizer_matches_full_ids_and_supervised_masks(self):
        result = inspect_tokenizer_parity(SyntheticTokenizer(),
                                          [row(), row("17", key="synthetic-b")], 128)
        self.assertEqual(result["status"], "PASS_SOURCE_PARITY")
        self.assertEqual(result["mismatched_examples"], 0)
        self.assertEqual(result["supervised_target_tokens"], 5)
        self.assertFalse(result["authorizes_training"])
        self.assertFalse(result["physical_tokenizer_attested"])
        self.assertFalse(result["corpus_membership_attested"])
        self.assertFalse(result["sampling_schedule_attested"])

    def test_prefix_stable_contextual_suffix_now_matches_both_runners(self):
        result = inspect_tokenizer_parity(ContextualTailTokenizer(), [row()], 128)
        self.assertEqual(result["reason"], "MATCHED_SUPPLIED_TOKEN_IDS_AND_LABELS")
        self.assertEqual(result["mismatched_examples"], 0)
        self.assertEqual(result["status"], "PASS_SOURCE_PARITY")
        self.assertFalse(result["authorizes_training"])

    def test_joined_prefix_token_merge_fails_both_boundary_checks(self):
        result = inspect_tokenizer_parity(BoundaryMergeTokenizer(), [row("X")], 128)
        self.assertEqual(result["reason"], "MEZO_PREPARATION_REJECTED")
        self.assertEqual(result["status"], "HOLD")

    def test_overlong_example_fails_mezo_no_truncation_guard(self):
        result = inspect_tokenizer_parity(SyntheticTokenizer(), [row("123")], 8)
        self.assertEqual(result["reason"], "MEZO_PREPARATION_REJECTED")

    def test_qlora_only_expansion_fails_no_truncation_guard(self):
        tokenizer = SyntheticTokenizer()
        from trainingday.r18_mezo import tokenize_cases
        from trainingday.r18_qlora import _tensorize
        prepared, _ = tokenize_cases(tokenizer, [row()], 128)
        qlora = _tensorize(tokenizer, [row()])
        inflated = [dict(qlora[0],
                         input_ids=qlora[0]["input_ids"] + [7] * 40,
                         labels=qlora[0]["labels"] + [-100] * 40)]
        with patch("trainingday.r24_tokenizer_parity._tensorize", return_value=inflated):
            result = inspect_tokenizer_parity(tokenizer, [row()], len(prepared[0]["input_ids"]))
        self.assertEqual(result["reason"], "QLORA_NO_TRUNCATION_LIMIT_EXCEEDED")
        self.assertEqual(result["status"], "HOLD")

    def test_invalid_or_duplicate_inputs_fail_closed(self):
        for examples, max_seq in [([], 128), ([row(), row()], 128),
                                  ([row()], 1), ([row("")], 128),
                                  ([{"id": "a", "prompt": "p"}], 128),
                                  ([row()], True)]:
            with self.subTest(examples=examples, max_seq=max_seq):
                result = inspect_tokenizer_parity(SyntheticTokenizer(), examples, max_seq)
                self.assertEqual(result["reason"], "INVALID_INPUT")
                self.assertEqual(result["status"], "HOLD")

    def test_report_contains_no_plain_prompt_response_or_token_ids(self):
        report = inspect_tokenizer_parity(SyntheticTokenizer(),
                                          [row("SECRET_SENTINEL")], 128)
        self.assertNotIn("SECRET_SENTINEL", str(report))
        self.assertNotIn("What is 1+1?", str(report))
        self.assertNotIn("input_ids", str(report))


if __name__ == "__main__":
    unittest.main()
