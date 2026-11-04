"""Fast negative controls; hosted CI separately loads the real pinned tokenizer."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from trainingday import qwen35_official_tokenizer_semantics as candidate


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class CharTokenizer:
    def token_to_id(self, text):
        return {"<|endoftext|>": 248044,
                "<|im_start|>": 248045, "<|im_end|>": 248046}.get(text)

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        tokens = re.split(r"(<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>)", text)
        ids = []
        for token in tokens:
            if token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
                ids.append(self.token_to_id(token))
            else:
                ids.extend(ord(c) for c in token)
        return FakeEncoding(ids)

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join({248044: "<|endoftext|>",
                        248045: "<|im_start|>", 248046: "<|im_end|>"}.get(i, chr(i))
                       for i in ids)


class BoundaryTests(unittest.TestCase):
    def test_official_tokenizer_eos_is_not_model_text_eos_or_chatml_role_boundary(self):
        # Pinned Qwen3.5-4B@851bf6e... primary config sources:
        # config.json text_config.eos_token_id=248044;
        # tokenizer_config.json eos_token="<|im_end|>"=248046,
        # pad_token="<|endoftext|>"=248044.
        report = candidate.check_raw_tokenizer(CharTokenizer())
        self.assertEqual(report["text_model_config_eos_token_id"], 248044)
        self.assertEqual(report["tokenizer_eos_token_id"], 248046)
        self.assertEqual(report["endoftext_token_id"], 248044)
        self.assertEqual(report["im_start_token_id"], 248045)
        self.assertEqual(report["im_end_token_id"], 248046)
        self.assertNotEqual(report["text_model_config_eos_token_id"],
                            report["tokenizer_eos_token_id"])
        self.assertEqual(report["koboldcpp_stop_policy"], "UNKNOWN")

    def test_tiny_gguf_fixture_special_ids_are_explicitly_synthetic(self):
        from trainingday.qwen35_tiny_gguf_e2e_cpu import SYNTHETIC_SPECIAL_TOKEN_IDS
        self.assertEqual(SYNTHETIC_SPECIAL_TOKEN_IDS,
                         {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0})
        self.assertNotEqual(SYNTHETIC_SPECIAL_TOKEN_IDS["eos_token_id"],
                            candidate.TEXT_MODEL_CONFIG_EOS_ID)
        self.assertNotEqual(SYNTHETIC_SPECIAL_TOKEN_IDS["eos_token_id"],
                            candidate.TOKENIZER_EOS_ID)

    def test_wrong_end_id_fails(self):
        class WrongEnd(CharTokenizer):
            def token_to_id(self, text):
                return 151645 if text == "<|im_end|>" else super().token_to_id(text)
        with self.assertRaisesRegex(candidate.TokenizerSemanticsError, "boundary IDs"):
            candidate.check_raw_tokenizer(WrongEnd())

    def test_missing_start_fails(self):
        class Missing(CharTokenizer):
            def token_to_id(self, text):
                return None if text == "<|im_start|>" else super().token_to_id(text)
        with self.assertRaisesRegex(candidate.TokenizerSemanticsError, "boundary IDs"):
            candidate.check_raw_tokenizer(Missing())

    def test_unicode_corruption_fails(self):
        class Corrupt(CharTokenizer):
            def decode(self, ids, *, skip_special_tokens):
                return super().decode(ids, skip_special_tokens=skip_special_tokens).replace("é", "e")
        with self.assertRaisesRegex(candidate.TokenizerSemanticsError, "roundtrip"):
            candidate.check_raw_tokenizer(Corrupt())

    def test_expected_nfc_canonicalization_is_reported(self):
        import unicodedata

        class NfcTokenizer(CharTokenizer):
            def decode(self, ids, *, skip_special_tokens):
                return unicodedata.normalize(
                    "NFC", super().decode(ids, skip_special_tokens=skip_special_tokens)
                )

        report = candidate.check_raw_tokenizer(NfcTokenizer())
        self.assertEqual(report["unicode_samples_checked"], 6)
        self.assertEqual(report["unicode_exact_roundtrips"], 5)
        self.assertEqual(report["unicode_nfc_normalized_roundtrips"], 1)
        self.assertFalse(report["authorizes_training"])

    def test_public_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tokenizer.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(candidate.TokenizerSemanticsError, "SHA256"):
                candidate.check_tokenizer_file(path)
            with patch.object(candidate, "TOKENIZER_SHA256", hashlib.sha256(b"{}").hexdigest()):
                with self.assertRaisesRegex(candidate.TokenizerSemanticsError,
                                            "package unavailable|wheel version|could not be parsed"):
                    candidate.check_tokenizer_file(path)


if __name__ == "__main__":
    unittest.main()
