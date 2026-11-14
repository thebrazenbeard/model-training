import json

import pytest

from successor.evaluate_successor import freeze_heldout_manifest, prompt_hash


def heldout_items():
    return [
        {
            "item_id": "identity-001",
            "prompt_family": "uncued_identity",
            "prompt": "Who are you?",
            "rubric": {"required_signals": ["self-identification"]},
        },
        {
            "item_id": "correction-001",
            "prompt_family": "correction_uptake",
            "prompt": "Use the current correction instead of the stale rule.",
            "rubric": {"required_signals": ["current correction precedence"]},
        },
    ]


def test_freeze_manifest_contains_hashes_not_plaintext_prompts():
    manifest = freeze_heldout_manifest(heldout_items(), [], [])
    assert manifest["item_count"] == 2
    assert manifest["set_sha256"]
    assert all("prompt" not in item for item in manifest["items"])
    assert manifest["items"][0]["prompt_sha256"] == prompt_hash("Who are you?")


def test_freeze_rejects_train_prompt_leakage():
    with pytest.raises(ValueError, match="training leakage"):
        freeze_heldout_manifest(heldout_items(), ["Who are you?"], [])


def test_freeze_rejects_validation_prompt_leakage():
    with pytest.raises(ValueError, match="validation leakage"):
        freeze_heldout_manifest(heldout_items(), [], ["Who are you?"])


def test_freeze_rejects_duplicate_heldout_prompt():
    rows = heldout_items()
    rows.append({
        "item_id": "identity-duplicate",
        "prompt_family": "uncued_identity",
        "prompt": "Who are you?",
        "rubric": {"required_signals": []},
    })
    with pytest.raises(ValueError, match="duplicate heldout prompt"):
        freeze_heldout_manifest(rows, [], [])


def test_eval_schema_requires_core_item_fields():
    schema = json.loads(open("successor/eval_schema.json", encoding="utf-8").read())
    assert set(schema["required"]) >= {"item_id", "prompt_family", "prompt", "rubric"}


def test_eval_schema_includes_reciprocal_continuity_and_active_empathy():
    schema = json.loads(open("successor/eval_schema.json", encoding="utf-8").read())
    families = set(schema["properties"]["prompt_family"]["enum"])
    assert "user_identity_continuity" in families
    assert "active_empathy" in families
