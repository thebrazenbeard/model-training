"""Source-only R18 train/eval prefix parity; never loads a model or GPU."""

import unittest

from trainingday import r18_eval, r18_qlora
from trainingday.numeric_prompt import numeric_question, r18_generation_prefix


class StrictTokenizer:
    eos_token = "!"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize,
                            add_generation_prompt, enable_thinking):
        self.calls.append((messages, tokenize, add_generation_prompt,
                           enable_thinking))
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"

    def __call__(self, content, **kwargs):
        return {"input_ids": [ord(c) for c in content]}


class R18ChatPrefixParityTests(unittest.TestCase):
    def test_train_and_eval_reference_one_strict_prefix_helper(self):
        self.assertIs(r18_qlora.r18_generation_prefix, r18_generation_prefix)
        self.assertIs(r18_eval.r18_generation_prefix, r18_generation_prefix)

    def test_strict_nonthinking_template_prefix_and_target_mask(self):
        tokenizer = StrictTokenizer()
        prompt = numeric_question("2 + 2?")
        expected = r18_generation_prefix(tokenizer, prompt)
        rows = r18_qlora._tensorize(
            tokenizer, [{"id": "synthetic", "prompt": prompt, "response": "4"}])
        self.assertEqual(tokenizer.calls[0],
                         ([{"role": "user", "content": prompt}], False, True, False))
        self.assertEqual(tokenizer.calls, tokenizer.calls[:1] * 2)
        self.assertEqual(rows[0]["input_ids"][:len(expected)],
                         [ord(c) for c in expected])
        self.assertEqual(rows[0]["labels"][:len(expected)],
                         [-100] * len(expected))
        self.assertEqual(rows[0]["labels"][-2:], [ord("4"), ord("!")])

    def test_no_silent_fallback_when_disable_thinking_is_unsupported(self):
        class LegacyTokenizer(StrictTokenizer):
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt):
                return "LEGACY:" + messages[0]["content"]

        with self.assertRaisesRegex(RuntimeError, "non-thinking"):
            r18_qlora._tensorize(LegacyTokenizer(), [
                {"id": "synthetic", "prompt": numeric_question("2+2?"),
                 "response": "4"}])

    def test_empty_or_nontext_prefix_is_rejected(self):
        for value in (None, b"bytes", ""):
            class MalformedTokenizer(StrictTokenizer):
                def apply_chat_template(self, messages, **kwargs):
                    return value

            with self.subTest(value=value), self.assertRaisesRegex(
                    RuntimeError, "invalid generation prefix"):
                r18_generation_prefix(MalformedTokenizer(), numeric_question("2+2?"))


if __name__ == "__main__":
    unittest.main()
