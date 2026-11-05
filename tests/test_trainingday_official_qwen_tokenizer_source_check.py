"""Offline negative tests for the pinned public tokenizer CI entry point.

The genuine raw Qwen tokenizer is exercised separately by GitHub CI; this
module intentionally needs neither its bytes nor the tokenizers wheel.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trainingday import official_qwen_tokenizer_source_check as candidate
from trainingday.calc_tokenization import FROZEN_TRAIN_SHA256, SCHEMA as COMPILER_SCHEMA
from trainingday.pinned_base import BASE_ID, BASE_REVISION


class CharRawTokenizer:
    def token_to_id(self, token):
        return 248046 if token == "<|im_end|>" else None

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(char) + 10 for char in text])

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(chr(value - 10) for value in ids)


def source_and_compiled():
    tokenizer = candidate._RawTokenizerAdapter(CharRawTokenizer())
    rows = [
        {"id": f"r201-calc-train-{index:04d}", "prompt": f"f'({index % 9})=?",
         "answer": str(-index if index % 2 else index), "split": "train"}
        for index in range(1024)
    ]
    examples = []
    for row in rows:
        prefix = f"Q: {row['prompt']}\nA:"
        completion = " " + row["answer"]
        first = tokenizer.encode(prefix, add_special_tokens=False)
        second = tokenizer.encode(completion, add_special_tokens=False)
        sequence = first + second + [tokenizer.eos_token_id]
        examples.append({
            "id": row["id"], "input_ids": sequence,
            "labels": [-100] * len(first) + second + [tokenizer.eos_token_id],
            "attention_mask": [1] * len(sequence),
        })
    compiled = {
        "schema": COMPILER_SCHEMA,
        "status": "TOKENIZATION_SOURCE_ONLY_UNAUTHENTICATED_HOLD",
        "base_model": BASE_ID, "base_revision": BASE_REVISION,
        "train_count": 1024, "train_sha256": FROZEN_TRAIN_SHA256,
        "max_sequence_length": 64, "examples": examples,
        "tokenizer_authenticated": False, "private_dev_access": False,
        "authorizes_training": False, "optimizer_steps_observed": 0,
    }
    return tokenizer, rows, compiled


class OfficialTokenizerSourceCheckTests(unittest.TestCase):
    def test_import_is_lightweight(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; "
             "import trainingday.official_qwen_tokenizer_source_check; "
             "assert 'torch' not in sys.modules; "
             "assert 'transformers' not in sys.modules; "
             "assert 'tokenizers' not in sys.modules"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_pinned_byte_digest_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tokenizer.json"
            data = b'{"public":"fixture"}'
            path.write_bytes(data)
            with patch.object(candidate, "TOKENIZER_SHA256", hashlib.sha256(data).hexdigest()):
                self.assertEqual(candidate._read_verified_tokenizer(path), data)
                path.write_bytes(data + b" ")
                with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                            "SHA256 mismatch"):
                    candidate._read_verified_tokenizer(path)
            with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                        "SHA256 mismatch"):
                candidate._read_verified_tokenizer(path)
            absent = Path(folder) / "absent.json"
            with self.assertRaises(candidate.TokenizerSourceCheckError):
                candidate._read_verified_tokenizer(absent)
            link = Path(folder) / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                        "symlink rejected"):
                candidate._read_verified_tokenizer(link)

    def test_version_and_eos_must_match_even_with_parseable_json(self):
        class FakeTokenizer:
            @staticmethod
            def from_str(value):
                self_value = value
                assert self_value == "{}"
                return CharRawTokenizer()

        module = SimpleNamespace(Tokenizer=FakeTokenizer)
        with patch.object(candidate, "version", return_value="0.21.0"):
            with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                        "version mismatch"):
                candidate._load_tokenizer(b"{}")
        with patch.object(candidate, "version", return_value="0.22.1"), \
                patch.dict(sys.modules, {"tokenizers": module}):
            loaded = candidate._load_tokenizer(b"{}")
            self.assertEqual(loaded.eos_token_id, 248046)
            self.assertEqual(loaded.decode(loaded.encode(" -4", add_special_tokens=False),
                                           skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False), " -4")

        class WrongEOS(CharRawTokenizer):
            def token_to_id(self, token):
                return 151645  # Qwen3, not the pinned Qwen3.5 tokenizer.

        with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                    "EOS mapping mismatch"):
            candidate._RawTokenizerAdapter(WrongEOS())

    def test_all_rows_canonical_and_fail_closed_on_tampering(self):
        tokenizer, rows, compiled = source_and_compiled()
        result = candidate._audit_compiled(rows, compiled, tokenizer)
        self.assertEqual(result["negative_answer_rows"], 512)
        self.assertLessEqual(result["max_observed_tokens"], 64)
        for flag in ("authorizes_training", "tokenizer_authenticated", "private_dev_access"):
            with self.subTest(flag=flag):
                forged = {**compiled, flag: True}
                with self.assertRaises(candidate.TokenizerSourceCheckError):
                    candidate._audit_compiled(rows, forged, tokenizer)
        with self.assertRaises(candidate.TokenizerSourceCheckError):
            candidate._audit_compiled(rows, {**compiled, "schema": "V1"}, tokenizer)
        with self.assertRaises(candidate.TokenizerSourceCheckError):
            candidate._audit_compiled(rows, {**compiled, "optimizer_steps_observed": 1},
                                      tokenizer)
        damaged = [dict(example) for example in compiled["examples"]]
        damaged[500]["labels"] = [-100] * len(damaged[500]["labels"])
        with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                    "mask/EOS mismatch"):
            candidate._audit_compiled(rows, {**compiled, "examples": damaged}, tokenizer)
        with self.assertRaisesRegex(candidate.TokenizerSourceCheckError,
                                    "coverage"):
            candidate._audit_compiled(rows, {**compiled,
                                             "examples": compiled["examples"][:-1]}, tokenizer)

    def test_cli_fails_nonzero_without_pinned_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            error = io.StringIO()
            with redirect_stderr(error):
                code = candidate.main(["--tokenizer", str(Path(folder) / "missing.json")])
            self.assertEqual(code, 1)
            self.assertIn("source HOLD", error.getvalue())


if __name__ == "__main__":
    unittest.main()
