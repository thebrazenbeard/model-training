"""Versioned strict numeric question shared by training and evaluation.

Changing a question's instructions after training confounds skill and formatting.
"""
from __future__ import annotations

from trainingday.admission import AdmissionError

NUMERIC_QUESTION_CONTRACT = "R15_STRICT_INTEGER_V1"
NUMERIC_SUFFIX = "\nReply with only the final integer. No steps or other words."


def numeric_question(prompt: str) -> str:
    if type(prompt) is not str or not prompt.strip() or len(prompt) > 4000:
        raise AdmissionError("invalid numeric question")
    if prompt.endswith(NUMERIC_SUFFIX):
        raise AdmissionError("numeric question already formatted")
    return prompt + NUMERIC_SUFFIX


def r18_generation_prefix(tokenizer, numeric_prompt: str) -> str:
    """Use one strict non-thinking Qwen chat prefix for R18 train and eval.

    A tokenizer which cannot accept the explicit thinking-mode contract is
    incompatible; silently retrying without that option can teach/evaluate
    under different assistant prefixes. This checks callback behavior only,
    not physical tokenizer identity or host custody.
    """
    if type(numeric_prompt) is not str or not numeric_prompt:
        raise RuntimeError("R18 numeric prompt invalid")
    try:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": numeric_prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, AttributeError) as exc:
        raise RuntimeError(
            "R18 tokenizer lacks explicit non-thinking chat-template contract") from exc
    if type(prefix) is not str or not prefix:
        raise RuntimeError("R18 tokenizer returned invalid generation prefix")
    return prefix
