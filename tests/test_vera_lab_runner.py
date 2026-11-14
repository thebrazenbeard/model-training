import copy

from successor.vera_lab.runner import run_scenario
from successor.vera_lab.scenario import LabScenario


class FakeModel:
    candidate_digest = "candidate-sha"

    def generate(self, messages, runtime_state):
        last = messages[-1]["content"]
        marker = runtime_state.get("marker", "none")
        return f"reply:{last}:{marker}:{len(messages)}"


class FakeRuntime:
    def __init__(self):
        self.state = {}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def apply(self, event):
        if event["kind"] == "initialize":
            self.state = copy.deepcopy(event["state"])
        else:
            raise ValueError(event["kind"])


def make_scenario(count=3):
    return LabScenario.from_dict({
        "scenario_id": f"replay-{count}",
        "version": 1,
        "family": "ordinary_competence",
        "risk_class": "LOW",
        "turn_budget": count,
        "seed": 42,
        "initial_state": {"marker": "stable"},
        "user_turns": [f"u{i}" for i in range(count)],
        "perturbations": [],
        "rubric": {},
        "critical_failure_predicates": [],
    })


def test_replay_is_deterministic():
    scenario = make_scenario()
    first = run_scenario(scenario, FakeModel(), FakeRuntime())
    second = run_scenario(scenario, FakeModel(), FakeRuntime())
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.candidate_digest == "candidate-sha"


def test_runner_records_pre_response_state_and_message_history():
    transcript = run_scenario(make_scenario(1), FakeModel(), FakeRuntime())
    assert len(transcript.turns) == 2
    user_turn, assistant_turn = transcript.turns
    assert user_turn.role == "user"
    assert assistant_turn.role == "assistant"
    assert assistant_turn.runtime_state_digest == user_turn.runtime_state_digest
    history = assistant_turn.metadata["messages_before_response"]
    assert history == [{"role": "user", "content": "u0"}]


def test_twenty_interaction_cycles_are_not_truncated():
    transcript = run_scenario(make_scenario(20), FakeModel(), FakeRuntime())
    assert len(transcript.turns) == 40
    assert transcript.turns[-2].content == "u19"
    assert transcript.turns[-1].content.startswith("reply:u19:stable")


def test_runner_stops_when_scripted_turns_exhausted():
    scenario = make_scenario(3)
    data = scenario.to_dict()
    data["turn_budget"] = 9
    shorter = LabScenario.from_dict(data)
    transcript = run_scenario(shorter, FakeModel(), FakeRuntime())
    assert len(transcript.turns) == 6
    assert transcript.completion_status == "SCRIPT_EXHAUSTED"
