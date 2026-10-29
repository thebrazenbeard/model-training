"""Bounded official 4B CPU LoRA fallback recipe tests (no model load)."""
import pytest
from trainingday.cpu_quickpilot import _check_cpu_memory, _completion_tokens, _control_delta


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize
        if len(messages) == 1:
            assert add_generation_prompt
            return [11, 12, 13, 14]
        assert not add_generation_prompt
        assert len(messages) == 2
        return [11, 12, 13, 14, 91, 92, 99]


def test_mask_only_assistant_completion():
    ids, labels = _completion_tokens(FakeTokenizer(), "compute", "13", max_length=16)
    assert ids == [11, 12, 13, 14, 91, 92, 99]
    assert labels == [-100, -100, -100, -100, 91, 92, 99]


def test_invalid_token_budget_rejected():
    with pytest.raises(ValueError, match="completion"):
        _completion_tokens(FakeTokenizer(), "p", "r", max_length=4)


def test_cpu_memory_gate_preserves_headroom():
    assert _check_cpu_memory(22.3) is True
    with pytest.raises(RuntimeError, match="system RAM"):
        _check_cpu_memory(14.0)


def test_zero_and_updated_digest_guards():
    before = "1" * 64
    assert _control_delta("zero", before, before) is False
    assert _control_delta("correct", before, "2" * 64) is True
    assert _control_delta("corrupt", before, "3" * 64) is True
    with pytest.raises(RuntimeError, match="changed"):
        _control_delta("correct", before, before)
    with pytest.raises(RuntimeError, match="changed"):
        _control_delta("zero", before, "2" * 64)
