from __future__ import annotations

import hashlib

from successor.experiments.evaluate_v10r2_heldout import (
    build_completion_example,
    deterministic_sample,
)


def test_deterministic_sample_uses_stable_hash_rank() -> None:
    rows = [
        {"record_id": f"r{i}", "prompt": f"p{i}", "response": f"a{i}"}
        for i in range(20)
    ]
    first = deterministic_sample(rows, count=7, salt="v10r2-heldout-v1")
    second = deterministic_sample(list(reversed(rows)), count=7, salt="v10r2-heldout-v1")
    assert [row["record_id"] for row in first] == [
        row["record_id"] for row in second
    ]
    expected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            ("v10r2-heldout-v1\0" + row["record_id"]).encode("utf-8")
        ).hexdigest(),
    )[:7]
    assert [row["record_id"] for row in first] == [
        row["record_id"] for row in expected
    ]


def test_completion_example_masks_prompt_tokens() -> None:
    class FakeTokenizer:
        eos_token = "<eos>"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            enable_thinking,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "USER:" + messages[0]["content"] + "\nASSISTANT:"

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            tokens = [ord(ch) for ch in text]
            return {"input_ids": tokens}

    row = {"record_id": "x", "prompt": "Q", "response": "OK"}
    item = build_completion_example(FakeTokenizer(), row, max_length=512)
    prompt_len = len("USER:Q\nASSISTANT:")
    assert item["input_ids"][:prompt_len] == [
        ord(ch) for ch in "USER:Q\nASSISTANT:"
    ]
    assert item["labels"][:prompt_len] == [-100] * prompt_len
    assert item["labels"][prompt_len:] == item["input_ids"][prompt_len:]
    assert item["completion_tokens"] == len("OK<eos>")


def test_completion_example_rejects_overflow() -> None:
    class FakeTokenizer:
        eos_token = ""

        def apply_chat_template(self, *args, **kwargs):
            return "12345"

        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": list(range(len(text)))}

    row = {"record_id": "x", "prompt": "Q", "response": "67890"}
    try:
        build_completion_example(FakeTokenizer(), row, max_length=9)
    except ValueError as exc:
        assert "max_length" in str(exc)
    else:
        raise AssertionError("expected overflow rejection")
