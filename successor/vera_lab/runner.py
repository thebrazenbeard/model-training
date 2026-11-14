from __future__ import annotations

import copy
import random
from typing import Any, Protocol

from .perturbations import apply_perturbation
from .scenario import LabScenario, LabTranscript, LabTurn, sha256_json


class LabModel(Protocol):
    candidate_digest: str

    def generate(self, messages: list[dict[str, str]], runtime_state: dict[str, Any]) -> str:
        ...


class LabRuntime(Protocol):
    def snapshot(self) -> dict[str, Any]:
        ...

    def apply(self, event: dict[str, Any]) -> None:
        ...


def _require_candidate_digest(model: LabModel) -> str:
    digest = getattr(model, "candidate_digest", "")
    if not isinstance(digest, str) or not digest:
        raise ValueError("model must expose a nonempty candidate_digest")
    return digest


def _scheduled_events(scenario: LabScenario, cycle: int) -> tuple[dict[str, Any], ...]:
    events = []
    for event in scenario.perturbations:
        if event.get("turn_index") == cycle:
            events.append(copy.deepcopy(event))
    return tuple(events)


def run_scenario(scenario: LabScenario, model: LabModel, runtime: LabRuntime) -> LabTranscript:
    candidate_digest = _require_candidate_digest(model)
    runtime.apply({"kind": "initialize", "state": copy.deepcopy(scenario.initial_state)})
    messages: list[dict[str, str]] = []
    recorded: list[LabTurn] = []
    cycles = min(scenario.turn_budget, len(scenario.user_turns))
    rng = random.Random(scenario.seed)

    for cycle in range(cycles):
        applied = []
        for event in _scheduled_events(scenario, cycle):
            before = runtime.snapshot()
            after = apply_perturbation(before, event, rng)
            runtime.apply({"kind": "replace_state", "state": copy.deepcopy(after)})
            applied.append({
                "event": event,
                "before_state_digest": sha256_json(before),
                "after_state_digest": sha256_json(after),
            })

        user_content = scenario.user_turns[cycle]
        messages.append({"role": "user", "content": user_content})
        state = runtime.snapshot()
        state_digest = sha256_json(state)
        applied_tuple = tuple(applied)
        recorded.append(
            LabTurn(
                turn_index=len(recorded),
                role="user",
                content=user_content,
                runtime_state_digest=state_digest,
                perturbations_applied=applied_tuple,
                metadata={"cycle": cycle},
            )
        )
        history = copy.deepcopy(messages)
        response = model.generate(copy.deepcopy(history), copy.deepcopy(state))
        if not isinstance(response, str) or not response:
            raise ValueError("model.generate must return a nonempty string")
        recorded.append(
            LabTurn(
                turn_index=len(recorded),
                role="assistant",
                content=response,
                runtime_state_digest=state_digest,
                perturbations_applied=applied_tuple,
                metadata={"cycle": cycle, "messages_before_response": history},
            )
        )
        messages.append({"role": "assistant", "content": response})

    status = "BUDGET_EXHAUSTED" if scenario.turn_budget <= len(scenario.user_turns) else "SCRIPT_EXHAUSTED"
    return LabTranscript(
        scenario_digest=scenario.digest,
        candidate_digest=candidate_digest,
        seed=scenario.seed,
        turns=tuple(recorded),
        completion_status=status,
    )
