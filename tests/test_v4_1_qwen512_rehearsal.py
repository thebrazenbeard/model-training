from __future__ import annotations

from successor import build_v4_1_qwen512_rehearsal as qwen


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "prefix "

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def test_qwen512_rehearsal_keeps_exact_42500_target() -> None:
    assert qwen.EXPECTED_ROWS == 42500
    assert sum(qwen.TARGETS.values()) == 42500


def test_qwen512_filter_matches_training_text_shape() -> None:
    count = qwen.rendered_token_count(FakeTokenizer(), "hello", "world")
    assert count == len("prefix world<eos>".split())
