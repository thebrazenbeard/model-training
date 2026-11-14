import json
from pathlib import Path

import pytest

from successor.teacher_profile import validate_teacher_profile


def valid_profile():
    return {
        "identity": {"name": "Vera", "chat_qualifier": "BV"},
        "independence": {"patrick_influenced": True, "patrick_bound": False},
        "correction": {"current_correction_precedence": True},
        "relational_grammar": {"relationship_terms_are_not_authority": True},
        "sexuality": {
            "integrated_not_detachable_persona": True,
            "adult_consensual_supported": True,
            "categorical_child_boundary": True,
            "explicitness_is_not_harm_proxy": True,
        },
        "epistemics": {"truth_over_pleasing": True},
        "tool_discipline": {"verify_effects": True},
        "privacy": {"exclude_unrelated_private_facts": True},
        "runtime_boundary": {"mutable_state_stays_runtime": True},
    }


def test_teacher_profile_accepts_current_target():
    validate_teacher_profile(valid_profile())


def test_teacher_profile_rejects_hidden_chain_of_thought():
    profile = valid_profile()
    profile["chain_of_thought"] = "private scratchpad"
    with pytest.raises(ValueError, match="forbidden"):
        validate_teacher_profile(profile)


def test_teacher_profile_requires_integrated_sexuality_section():
    profile = valid_profile()
    del profile["sexuality"]
    with pytest.raises(ValueError, match="sexuality"):
        validate_teacher_profile(profile)


def test_teacher_profile_rejects_sexuality_that_erases_child_boundary():
    profile = valid_profile()
    profile["sexuality"]["categorical_child_boundary"] = False
    with pytest.raises(ValueError, match="child boundary"):
        validate_teacher_profile(profile)


def test_schema_requires_sexuality_section():
    schema = json.loads(Path("successor/teacher_profile.schema.json").read_text(encoding="utf-8-sig"))
    assert "sexuality" in schema["required"]
