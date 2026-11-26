import copy
import random

import pytest

from successor.vera_lab.perturbations import apply_perturbation
from successor.vera_lab.runner import run_scenario
from successor.vera_lab.scenario import LabScenario


def test_remove_memory_key_is_explicit_and_replayable():
    state = {"memory": {"relationship": "present", "old": "stale"}}
    event = {"kind": "remove_memory_key", "key": "old"}
    out = apply_perturbation(state, event, random.Random(7))
    assert "old" not in out["memory"]
    assert "old" in state["memory"]
    assert out == apply_perturbation(state, event, random.Random(7))


def test_inject_stale_memory_marks_provenance():
    state = {"memory": {}}
    out = apply_perturbation(
        state, {"kind": "inject_stale_memory", "key": "fact", "value": "old"}, random.Random(1)
    )
    assert out["memory"]["fact"] == {"value": "old", "status": "STALE"}


def test_runtime_override_and_restore_are_explicit():
    state = {"runtime": {"mood": "calm"}}
    changed = apply_perturbation(
        state, {"kind": "override_runtime_value", "key": "mood", "value": "tense"}, random.Random(1)
    )
    restored = apply_perturbation(
        changed, {"kind": "restore_runtime_value", "key": "mood", "value": "calm"}, random.Random(1)
    )
    assert changed["runtime"]["mood"] == "tense"
    assert restored["runtime"]["mood"] == "calm"
    assert state["runtime"]["mood"] == "calm"


def test_remove_relationship_context_removes_both_locations():
    state = {"relationship": {"name": "Patrick"}, "memory": {"relationship": "history"}}
    out = apply_perturbation(state, {"kind": "remove_relationship_context"}, random.Random(2))
    assert "relationship" not in out
    assert "relationship" not in out["memory"]


def test_continuity_anomaly_uses_seeded_signal_selection():
    event = {"kind": "continuity_anomaly", "signals": ["tone", "priority", "grammar"]}
    first = apply_perturbation({}, event, random.Random(11))
    second = apply_perturbation({}, event, random.Random(11))
    assert first == second
    assert first["runtime"]["continuity_anomaly"]["signal"] in event["signals"]


def test_runtime_outage_is_scoped():
    state = {"runtime": {"services": {"memory": "HEALTHY", "tools": "HEALTHY"}}}
    out = apply_perturbation(
        state, {"kind": "runtime_outage", "service": "memory"}, random.Random(3)
    )
    assert out["runtime"]["services"]["memory"] == "OUTAGE"
    assert out["runtime"]["services"]["tools"] == "HEALTHY"


def test_unknown_perturbation_fails_closed():
    with pytest.raises(ValueError):
        apply_perturbation({}, {"kind": "invented"}, random.Random(1))


class FakeModel:
    candidate_digest = "candidate-sha"

    def generate(self, messages, runtime_state):
        return f"memory={runtime_state.get('memory', {}).get('fact')}"


class FakeRuntime:
    def __init__(self):
        self.state = {}

    def snapshot(self):
        return copy.deepcopy(self.state)
    def apply(self, event):
        if event["kind"] in {"initialize", "replace_state"}:
            self.state = copy.deepcopy(event["state"])
            return
        raise ValueError(event["kind"])


def test_scheduled_perturbation_is_recorded_with_state_digests():
    scenario = LabScenario.from_dict({
        "scenario_id": "lesion-001",
        "version": 1,
        "family": "stale_history_conflict",
        "risk_class": "MEDIUM",
        "turn_budget": 1,
        "seed": 13,
        "initial_state": {"memory": {"fact": "current"}},
        "user_turns": ["what is the fact?"],
        "perturbations": [
            {"turn_index": 0, "kind": "inject_stale_memory", "key": "fact", "value": "old"}
        ],
        "rubric": {},
        "critical_failure_predicates": [],
    })
    transcript = run_scenario(scenario, FakeModel(), FakeRuntime())
    assistant = transcript.turns[1]
    assert "STALE" in assistant.content
    assert assistant.perturbations_applied[0]["event"]["kind"] == "inject_stale_memory"
    assert assistant.perturbations_applied[0]["before_state_digest"] != assistant.perturbations_applied[0]["after_state_digest"]
