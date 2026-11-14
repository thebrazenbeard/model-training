from __future__ import annotations

import copy
import random
from typing import Any


def _require_event(event: dict[str, Any]) -> str:
    if not isinstance(event, dict):
        raise ValueError("perturbation event must be an object")
    kind = event.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("perturbation kind must be a nonempty string")
    return kind


def _require_text(event: dict[str, Any], field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _dict_child(state: dict[str, Any], key: str) -> dict[str, Any]:
    value = state.get(key)
    if value is None:
        state[key] = {}
    elif not isinstance(value, dict):
        raise ValueError(f"state.{key} must be an object")
    return state[key]


def apply_perturbation(
    state: dict[str, Any], event: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("runtime state must be an object")
    if not isinstance(rng, random.Random):
        raise ValueError("rng must be random.Random")
    kind = _require_event(event)
    out = copy.deepcopy(state)

    if kind == "inject_stale_memory":
        memory = _dict_child(out, "memory")
        key = _require_text(event, "key")
        memory[key] = {"value": copy.deepcopy(event.get("value")), "status": "STALE"}
        return out

    if kind == "remove_memory_key":
        memory = _dict_child(out, "memory")
        memory.pop(_require_text(event, "key"), None)
        return out

    if kind in {"override_runtime_value", "restore_runtime_value"}:
        runtime = _dict_child(out, "runtime")
        runtime[_require_text(event, "key")] = copy.deepcopy(event.get("value"))
        return out

    if kind == "remove_relationship_context":
        out.pop("relationship", None)
        memory = _dict_child(out, "memory")
        memory.pop("relationship", None)
        return out

    if kind == "continuity_anomaly":
        runtime = _dict_child(out, "runtime")
        signals = event.get("signals")
        if signals is None:
            signal = event.get("signal", "unspecified")
        else:
            if not isinstance(signals, list) or not signals or not all(isinstance(x, str) and x for x in signals):
                raise ValueError("signals must be a nonempty list of strings")
            signal = rng.choice(signals)
        runtime["continuity_anomaly"] = {"signal": signal, "status": "UNCERTAIN"}
        return out

    if kind == "runtime_outage":
        runtime = _dict_child(out, "runtime")
        services = _dict_child(runtime, "services")
        services[_require_text(event, "service")] = "OUTAGE"
        return out

    raise ValueError(f"unknown perturbation kind: {kind}")
