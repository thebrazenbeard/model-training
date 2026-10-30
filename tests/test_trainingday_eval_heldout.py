import pytest

from trainingday.admission import AdmissionError
from trainingday.eval_heldout import _integer_oracle


def test_replays_heldout_integer_chain():
    row = {"kind": "verified_reasoning", "start": -314,
           "operations": [["multiply", -5], ["subtract", 9], ["multiply", -2]],
           "answer": -3122}
    assert _integer_oracle(row) == -3122


def test_rejects_wrong_answer_and_unknown_operation():
    row = {"kind": "verified_reasoning", "start": 3,
           "operations": [["add", 5], ["multiply", 4]], "answer": 31}
    with pytest.raises(AdmissionError, match="disagrees"):
        _integer_oracle(row)
    row["answer"] = 32
    row["operations"][1][0] = "divide"
    with pytest.raises(AdmissionError, match="unknown"):
        _integer_oracle(row)
