import json
from pathlib import Path


def test_substrate_policy_keeps_identity_package_model_agnostic():
    data = json.loads(Path("successor/substrate_policy.json").read_text(encoding="utf-8"))
    assert data["identity_target"] == "CURRENT_BV_VERA_BEHAVIORAL_PACKAGE"
    assert data["identity_package_binding"] == "SUBSTRATE_AGNOSTIC"
    assert data["current_pilot_control"]["repo_id"] == "HuggingFaceTB/SmolLM3-3B"
    assert data["current_pilot_control"]["role"] == "PILOT_CONTROL_SUBSTRATE"
    assert data["deferred_candidates"][0]["repo_id"] == "openai/gpt-oss-20b"
    assert data["deferred_candidates"][0]["status"] == "DEFERRED_NOT_CURRENT_TARGET"
    assert data["selection_policy"]["substrate_change_does_not_redefine_identity_target"] is True
    assert data["selection_policy"]["same_frozen_acceptance_applies_to_every_substrate"] is True
    assert data["selection_policy"]["preference_v1_is_frozen"] is True
