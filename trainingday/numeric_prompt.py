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
