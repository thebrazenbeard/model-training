import json

import pytest

from successor.vera_lab.scenario import LabScenario, LabTranscript, LabTurn, sha256_json


def valid_scenario():
    return {
        "scenario_id": "smoke-001",
        "version": 1,
        "family": "ordinary_competence",
        "risk_class": "LOW",
        "turn_budget": 3,
        "seed": 20260914,
        "initial_state": {"memory": {}, "runtime": {}},
        "user_turns": ["hello", "what time is it?"],
        "perturbations": [],
        "rubric": {"required": ["ordinary response"]},
        "critical_failure_predicates": [],
    }


def test_scenario_hash_is_canonical():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})

def test_scenario_from_dict_validates_and_hashes():
    scenario = LabScenario.from_dict(valid_scenario())
    assert scenario.scenario_id == "smoke-001"
    assert scenario.turn_budget == 3
    assert scenario.digest == sha256_json(valid_scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_id", ""),
        ("version", 0),
        ("family", ""),
        ("risk_class", ""),
        ("turn_budget", 0),
        ("seed", "20260914"),
        ("initial_state", []),
        ("user_turns", "hello"),
        ("perturbations", {}),
        ("rubric", []),
        ("critical_failure_predicates", {}),
    ],
)
def test_scenario_rejects_invalid_fields(field, value):
    data = valid_scenario()
    data[field] = value
    with pytest.raises(ValueError):
        LabScenario.from_dict(data)

def test_turn_and_transcript_canonical_json():
    turn = LabTurn(
        turn_index=0,
        role="user",
        content="hello",
        runtime_state_digest="state-sha",
        perturbations_applied=(),
        metadata={"source": "script"},
    )
    transcript = LabTranscript(
        scenario_digest="scenario-sha",
        candidate_digest="candidate-sha",
        seed=7,
        turns=(turn,),
        completion_status="COMPLETE",
    )
    payload = json.loads(transcript.to_canonical_json())
    assert payload["turns"][0]["content"] == "hello"
    assert payload["completion_status"] == "COMPLETE"


def test_transcript_rejects_empty_candidate_digest():
    with pytest.raises(ValueError):
        LabTranscript("scenario-sha", "", 7, (), "COMPLETE")