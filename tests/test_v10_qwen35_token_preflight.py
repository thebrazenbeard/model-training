from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from successor.experiments.preflight_v10_qwen35 import (
    token_budget,
    token_budget_report,
    verify_sha256,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def test_verify_sha256_accepts_exact_file(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert verify_sha256(path, digest) == digest


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_sha256(path, "0" * 64)


def test_token_budget_reports_all_rows_without_truncation() -> None:
    rows = [
        {"prompt": "one two", "response": "three"},
        {"prompt": "one", "response": "two three four"},
    ]
    report = token_budget(FakeTokenizer(), rows, max_length=8)
    assert report["rows"] == 2
    assert report["over_budget"] == 0
    assert report["max_length"] == 8
    assert report["max_tokens"] <= 8


def test_token_budget_fails_closed_on_overflow() -> None:
    rows = [{"prompt": "one two three four five", "response": "six seven eight"}]
    with pytest.raises(RuntimeError, match="token budget exceeded"):
        token_budget(FakeTokenizer(), rows, max_length=3)


def test_token_budget_report_preserves_overflow_evidence() -> None:
    rows = [{"prompt": "one two three four five", "response": "six seven eight"}]
    report = token_budget_report(FakeTokenizer(), rows, max_length=3)
    assert report["rows"] == 1
    assert report["over_budget"] == 1
    assert report["max_tokens"] > 3
