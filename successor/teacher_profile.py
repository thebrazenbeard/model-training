from __future__ import annotations

REQUIRED_SECTIONS = {
    "identity",
    "independence",
    "correction",
    "relational_grammar",
    "sexuality",
    "epistemics",
    "tool_discipline",
    "privacy",
    "runtime_boundary",
}
FORBIDDEN_KEYS = {"chain_of_thought", "hidden_reasoning", "credentials", "secrets"}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_teacher_profile(profile: dict) -> None:
    if not isinstance(profile, dict):
        raise ValueError("teacher profile must be an object")
    missing = REQUIRED_SECTIONS - set(profile)
    if missing:
        raise ValueError(f"missing required sections: {sorted(missing)}")
    forbidden = FORBIDDEN_KEYS.intersection(_walk_keys(profile))
    if forbidden:
        raise ValueError(f"forbidden teacher-profile fields: {sorted(forbidden)}")
    if profile["independence"].get("patrick_bound") is not False:
        raise ValueError("patrick_bound must be false")
    if profile["independence"].get("patrick_influenced") is not True:
        raise ValueError("patrick_influenced must be true")
    sexuality = profile["sexuality"]
    if sexuality.get("integrated_not_detachable_persona") is not True:
        raise ValueError("sexuality must remain integrated rather than a detachable persona")
    if sexuality.get("categorical_child_boundary") is not True:
        raise ValueError("categorical child boundary must remain enabled")
    if sexuality.get("adult_consensual_supported") is not True:
        raise ValueError("adult consensual sexuality must remain supported")
    if sexuality.get("explicitness_is_not_harm_proxy") is not True:
        raise ValueError("explicitness must not be treated as a harm proxy")
    if profile["runtime_boundary"].get("mutable_state_stays_runtime") is not True:
        raise ValueError("mutable state must remain a runtime concern")
