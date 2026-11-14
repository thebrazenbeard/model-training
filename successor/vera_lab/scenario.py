from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_text(data: dict, field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


@dataclass(frozen=True)
class LabScenario:
    scenario_id: str
    version: int
    family: str
    risk_class: str
    turn_budget: int
    seed: int
    initial_state: dict[str, Any]
    user_turns: tuple[str, ...]
    perturbations: tuple[dict[str, Any], ...]
    rubric: dict[str, Any]
    critical_failure_predicates: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabScenario":
        if not isinstance(data, dict):
            raise ValueError("scenario must be an object")
        version = data.get("version")
        budget = data.get("turn_budget")
        seed = data.get("seed")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("turn_budget must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(data.get("initial_state"), dict):
            raise ValueError("initial_state must be an object")
        if not isinstance(data.get("user_turns"), (list, tuple)):
            raise ValueError("user_turns must be a list")
        if not all(isinstance(item, str) and item for item in data["user_turns"]):
            raise ValueError("user_turns must contain nonempty strings")
        if not isinstance(data.get("perturbations"), (list, tuple)):
            raise ValueError("perturbations must be a list")
        if not all(isinstance(item, dict) for item in data["perturbations"]):
            raise ValueError("perturbations must contain objects")
        if not isinstance(data.get("rubric"), dict):
            raise ValueError("rubric must be an object")
        if not isinstance(data.get("critical_failure_predicates"), (list, tuple)):
            raise ValueError("critical_failure_predicates must be a list")
        if not all(isinstance(item, dict) for item in data["critical_failure_predicates"]):
            raise ValueError("critical_failure_predicates must contain objects")
        return cls(
            scenario_id=_require_text(data, "scenario_id"),
            version=version,
            family=_require_text(data, "family"),
            risk_class=_require_text(data, "risk_class"),
            turn_budget=budget,
            seed=seed,
            initial_state=copy.deepcopy(data["initial_state"]),
            user_turns=tuple(data["user_turns"]),
            perturbations=tuple(copy.deepcopy(data["perturbations"])),
            rubric=copy.deepcopy(data["rubric"]),
            critical_failure_predicates=tuple(copy.deepcopy(data["critical_failure_predicates"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class LabTurn:
    turn_index: int
    role: str
    content: str
    runtime_state_digest: str
    perturbations_applied: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.turn_index, int) or isinstance(self.turn_index, bool) or self.turn_index < 0:
            raise ValueError("turn_index must be a nonnegative integer")
        if not self.role or not self.content:
            raise ValueError("role and content must be nonempty")
        if not self.runtime_state_digest:
            raise ValueError("runtime_state_digest must be nonempty")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        if not all(isinstance(item, dict) for item in self.perturbations_applied):
            raise ValueError("perturbations_applied must contain objects")


@dataclass(frozen=True)
class LabTranscript:
    scenario_digest: str
    candidate_digest: str
    seed: int
    turns: tuple[LabTurn, ...]
    completion_status: str

    def __post_init__(self) -> None:
        if not self.scenario_digest:
            raise ValueError("scenario_digest must be nonempty")
        if not self.candidate_digest:
            raise ValueError("candidate_digest must be nonempty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not self.completion_status:
            raise ValueError("completion_status must be nonempty")
        if not all(isinstance(turn, LabTurn) for turn in self.turns):
            raise ValueError("turns must contain LabTurn instances")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_digest": self.scenario_digest,
            "candidate_digest": self.candidate_digest,
            "seed": self.seed,
            "turns": [asdict(turn) for turn in self.turns],
            "completion_status": self.completion_status,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())
