"""Source-level guardrails for the R15 matched numeric prompt contract."""
import pytest

from trainingday.admission import AdmissionError
from trainingday.numeric_prompt import (
    NUMERIC_QUESTION_CONTRACT, NUMERIC_SUFFIX, numeric_question,
)
from trainingday.train_neural import _control_rows
from trainingday.eval_heldout import numeric_question as eval_numeric_question


def test_shared_numeric_question_is_identical_for_trainer_and_evaluator():
    rows = [
        {"id": "one", "family": "arithmetic", "prompt": "Compute 2+3.", "answer": 5},
        {"id": "two", "family": "arithmetic", "prompt": "Compute 2+4.", "answer": 6},
    ]
    correct = _control_rows(rows, "correct")
    corrupt = _control_rows(rows, "corrupt")
    assert NUMERIC_QUESTION_CONTRACT == "R15_STRICT_INTEGER_V1"
    assert eval_numeric_question is numeric_question
    assert [r["prompt"] for r in correct] == [r["prompt"] for r in corrupt]
    assert all(r["prompt"] == numeric_question(x["prompt"]) for r, x in zip(correct, rows))
    assert correct[0]["response"] == 5 and corrupt[0]["response"] != 5


@pytest.mark.parametrize("prompt", ["", " ", None, 7])
def test_numeric_question_rejects_invalid_inputs(prompt):
    with pytest.raises(AdmissionError):
        numeric_question(prompt)


def test_numeric_question_rejects_double_suffix():
    with pytest.raises(AdmissionError):
        numeric_question("Compute 2+3." + NUMERIC_SUFFIX)
