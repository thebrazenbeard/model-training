"""Pure offline tests; no tokenizer, host, GPU or model weights required."""

import unittest
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trainingday.dialogue_curriculum import candidate_rows
from trainingday.dialogue_token_budget_probe import (
    _bins, _token_count, _verify_public_candidate, _verify_tokenizer_snapshot,
    EXPECTED_TOKENIZER_FILES, main,
)


class TokenBudgetProbeTests(unittest.TestCase):
    def test_batch_encoding_counts_input_ids_not_mapping_keys(self):
        # Host Transformers returns a two-key BatchEncoding. len(mapping)=2
        # would silently fabricate tiny 2-token sequences without this guard.
        result = {"input_ids": list(range(37)), "attention_mask": [1] * 37}
        self.assertEqual(len(result), 2)
        self.assertEqual(_token_count(result), 37)

    def test_list_and_single_batch(self):
        self.assertEqual(_token_count([1, 2, 3]), 3)
        self.assertEqual(_token_count({"input_ids": [[1, 2, 3, 4]]}), 4)

    def test_reject_malformed_encoded_output(self):
        for malformed in ({}, {"input_ids": []}, {"input_ids": [[1], [2]]},
                          {"input_ids": ["1", "2"]}):
            with self.assertRaises((KeyError, ValueError, TypeError)):
                _token_count(malformed)

    def test_aggregate_budgets(self):
        b = _bins([24, 64, 65, 96, 128, 257])
        self.assertEqual(b["median"], 80.5)
        self.assertEqual(b["within_budget"],
                         {"64": 2, "96": 4, "128": 5, "256": 5})
        self.assertEqual(b["over_budget"]["256"], 1)

    def test_frozen_exact_candidate_no_substitution(self):
        rows = candidate_rows()
        _verify_public_candidate(rows)
        rows[0]["messages"][-1]["content"] += " A harmless but unapproved edit."
        with self.assertRaisesRegex(ValueError, "candidate content hash"):
            _verify_public_candidate(rows)

    def test_unpinned_tokenizer_file_bytes_rejected_before_import(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
                (path / name).write_text("not the pinned bytes", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer file bytes"):
                _verify_tokenizer_snapshot(path)

    def test_receipt_revision_mismatch_is_rejected_even_if_hashes_match(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for name in EXPECTED_TOKENIZER_FILES:
                (path / name).write_text("stub", encoding="utf-8")
            receipt = {"repo": "Qwen/Qwen3.5-4B", "revision": "0" * 40,
                       "files": [{"file": key, "sha256": digest} for key, digest in
                                 EXPECTED_TOKENIZER_FILES.items()]}
            (path / "TRAININGDAY_PUBLIC_SOURCE_RECEIPT.json").write_text(
                json.dumps(receipt), encoding="utf-8")
            with patch("trainingday.dialogue_token_budget_probe._digest",
                       side_effect=lambda p: EXPECTED_TOKENIZER_FILES[p.name]):
                with self.assertRaisesRegex(ValueError, "model/revision mismatch"):
                    _verify_tokenizer_snapshot(path)

    def test_bad_private_system_file_never_prints_path_or_traceback(self):
        private_path = "DO_NOT_LEAK_PRIVATE_PREFIX_LOCATION_TEST"
        out = io.StringIO()
        with patch.object(sys, "argv", ["probe", "--snapshot", "nonexistent",
                                        "--system-file", private_path]):
            with redirect_stdout(out), self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(json.loads(out.getvalue()),
                         {"status": "NOT_MEASURED", "error_type": "FileNotFoundError"})
        self.assertNotIn(private_path, out.getvalue())


if __name__ == "__main__":
    unittest.main()
