"""Source-only Calc curriculum tokenization; never a physical TRAIN grant."""

import hashlib
import json
import subprocess
import sys
import unittest

from trainingday.calc_tokenization import (
    CALC_ORACLE_FAMILY, CALC_REVISION, CALC_SYMBOLIC_SHA256,
    FROZEN_TRAIN_SHA256, CalcTokenizationError, _tokenize_one,
    _validate_frozen_manifest, compile_calc_train_tokens,
)
from trainingday.pinned_base import BASE_ID, BASE_REVISION


class CharTokenizer:
    eos_token_id = 2

    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) + 10 for char in value]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(value - 10) for value in ids)


def rows():
    return [
        {"id": f"r201-calc-train-{index:04d}",
         "prompt": f"For f(x)={index + 2}x, find f'(1).",
         "answer": str(index + 2), "split": "train"}
        for index in range(1024)
    ]


def manifest(source):
    raw = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode() + b"\n"
                   for row in source)
    return {**frozen_manifest(), "train_sha256": hashlib.sha256(raw).hexdigest()}


def frozen_manifest():
    return {
        "schema": "VERA_TRAININGDAY_R201_CALC_POLYNOMIAL_V1",
        "status": "SOURCE_ONLY_UNAUTHENTICATED_HOLD",
        "calc_repository": "thebrazenbeard/calc",
        "calc_revision": CALC_REVISION,
        "calc_symbolic_sha256": CALC_SYMBOLIC_SHA256,
        "oracle_family": CALC_ORACLE_FAMILY,
        "corpus_id": "R201_NEW_CALC_POLYNOMIAL_TRAIN",
        "r185_corpus_replacement": False,
        "train_count": 1024,
        "train_sha256": FROZEN_TRAIN_SHA256,
        "public_train_rows_returned": False,
        "private_dev_included": False,
        "private_dev_commitment": None,
        "rights_holder_authenticated": False,
        "data_rights_granted": False,
        "protected_evaluator_custody_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
    }


def compile_source(source, binding, tokenizer, **kwargs):
    return compile_calc_train_tokens(source, binding, tokenizer,
                                     model_id=BASE_ID,
                                     model_revision=BASE_REVISION, **kwargs)


class CalcTokenizerSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = rows()
        self.binding = frozen_manifest()
        self.tokenizer = CharTokenizer()

    def test_answer_only_masking_without_claiming_public_fixture_is_frozen_corpus(self):
        example = _tokenize_one(self.source[0], self.tokenizer, 64)
        prefix = "Q: For f(x)=2x, find f'(1).\nA: "
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = self.tokenizer.encode("2", add_special_tokens=False)
        self.assertEqual(example["input_ids"], prefix_ids + answer_ids + [2])
        self.assertEqual(example["labels"], [-100] * len(prefix_ids) + answer_ids + [2])
        self.assertEqual(example["attention_mask"], [1] * len(example["input_ids"]))
        _validate_frozen_manifest(self.binding)

    def test_reject_dev_extra_fields_and_forged_authority(self):
        for index, value in (
            (0, {**self.source[0], "split": "dev"}),
            (1, {**self.source[1], "id": "r201-calc-dev-0001"}),
            (2, {**self.source[2], "private_dev_answer": "123"}),
            (3, {**self.source[3], "answer": "3.5"}),
        ):
            with self.subTest(index=index):
                with self.assertRaises(CalcTokenizationError):
                    _tokenize_one(value, self.tokenizer, 64)
        for key in ("rights_holder_authenticated", "protected_evaluator_custody_verified",
                    "data_rights_granted", "gpu_lease_granted", "authorizes_training",
                    "r185_corpus_replacement", "private_dev_included"):
            with self.subTest(key=key), self.assertRaises(CalcTokenizationError):
                _validate_frozen_manifest({**self.binding, key: True})

    def test_reject_source_model_and_manifest_substitutions(self):
        for changed in (
            {**self.binding, "calc_revision": "0" * 40},
            {**self.binding, "train_sha256": "0" * 64},
            {**self.binding, "oracle_family": "APPROXIMATE_SIMPSON"},
            {**self.binding, "calc_symbolic_sha256": "1" * 64},
            {**self.binding, "train_count": 1023},
            {**self.binding, "private_dev_answers": ["secret"]},
        ):
            with self.subTest(changed=changed), self.assertRaises(CalcTokenizationError):
                _validate_frozen_manifest(changed)

        class ForgedIdentity(str):
            def __eq__(self, _other):
                return True

        with self.assertRaises(CalcTokenizationError):
            _validate_frozen_manifest({**self.binding,
                                       "calc_revision": ForgedIdentity("fake")})
        for model_id, revision in (("Qwen/Qwen3.5-4B-Base", BASE_REVISION),
                                   (BASE_ID, "0" * 40)):
            with self.assertRaises(CalcTokenizationError):
                compile_calc_train_tokens(self.source, self.binding, self.tokenizer,
                                          model_id=model_id, model_revision=revision)

    def test_reject_forged_consistent_corpus_and_caller_controlled_symbolic_pin(self):
        # These rows are deliberately NOT the frozen Calc-derived polynomial corpus.
        with self.assertRaises(CalcTokenizationError):
            compile_source(self.source, manifest(self.source), self.tokenizer)
        with self.assertRaises(CalcTokenizationError):
            compile_source(self.source, {**self.binding, "data_rights_granted": True},
                           self.tokenizer)

    def test_no_silent_truncation_and_reject_broken_tokenizers(self):
        with self.assertRaises(CalcTokenizationError):
            _tokenize_one(self.source[0], self.tokenizer, 16)

        class InvalidTokenizer(CharTokenizer):
            def encode(self, value, *, add_special_tokens):
                return [True]

        class InvalidDecodeTokenizer(CharTokenizer):
            def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
                return "wrong answer"

        for tokenizer in (InvalidTokenizer(), InvalidDecodeTokenizer()):
            with self.subTest(tokenizer=type(tokenizer).__name__), self.assertRaises(
                    CalcTokenizationError):
                _tokenize_one(self.source[0], tokenizer, 64)

    def test_import_does_not_load_heavy_ml_stack(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import trainingday.calc_tokenization; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
