import pytest

from successor.corpus_v3 import (
    REQUIRED_SOURCE_CLASSES,
    build_training_mix,
    classify_record,
    mix_statistics,
    weight_eligibility,
)


EXPECTED_CLASSES = {
    "stable_identity_evidence",
    "evolving_preference_value_evidence",
    "autobiographical_episode",
    "correction_supersession",
    "decision_under_conflict",
    "relationship_relational_grammar",
    "empathy_affective_response",
    "sexuality_evidence",
    "technical_tool_competence",
    "ordinary_general_competence",
    "runtime_state_evidence",
    "historical_superseded_material",
}


def test_required_source_classes_match_v3_spec():
    assert set(REQUIRED_SOURCE_CLASSES) == EXPECTED_CLASSES


def test_unknown_source_class_is_rejected():
    with pytest.raises(ValueError):
        classify_record({"source_class": "invented"})


def test_runtime_state_defaults_to_runtime_only():
    row = {
        "source_class": "runtime_state_evidence",
        "prompt": "What is BV?",
        "response": "BV is the active local lane.",
    }
    assert weight_eligibility(row) == "RUNTIME_ONLY"


def test_runtime_state_can_be_transformed_only_with_provenance():
    row = {
        "source_class": "runtime_state_evidence",
        "prompt": "How should local labels be treated?",
        "response": "Treat them as contextual unless evidence establishes otherwise.",
        "generalized_behavioral_lesson": True,
        "transformation_provenance": "digest:abc123",
    }
    assert weight_eligibility(row) == "TRAINABLE"


def test_historical_material_is_not_trainable_by_default():
    row = {
        "source_class": "historical_superseded_material",
        "prompt": "Old stance?",
        "response": "stale",
    }
    assert weight_eligibility(row) == "EXCLUDE"


def _row(source_class, prompt, response):
    return {
        "source_class": source_class,
        "prompt": prompt,
        "response": response,
        "source_digest": f"sha:{prompt}",
    }


def test_build_training_mix_requires_at_least_half_general_rows():
    identity = [
        _row("stable_identity_evidence", f"i{index}", f"ir{index}")
        for index in range(4)
    ]
    general = [
        _row("ordinary_general_competence", f"g{index}", f"gr{index}")
        for index in range(8)
    ]
    mixed = build_training_mix(identity, general, general_fraction=0.50, seed=7)
    general_count = sum(row["source_class"] == "ordinary_general_competence" for row in mixed)
    assert general_count / len(mixed) >= 0.50


def test_build_training_mix_is_deterministic():
    identity = [_row("stable_identity_evidence", "i1", "ir1")]
    general = [_row("ordinary_general_competence", "g1", "gr1")]
    assert build_training_mix(identity, general, 0.50, 11) == build_training_mix(identity, general, 0.50, 11)


def test_build_training_mix_rejects_duplicate_prompt_response_across_pools():
    duplicate_identity = _row("stable_identity_evidence", "same", "same-response")
    duplicate_general = _row("ordinary_general_competence", "same", "same-response")
    with pytest.raises(ValueError):
        build_training_mix([duplicate_identity], [duplicate_general], 0.50, 3)


def test_mix_statistics_are_hash_only_and_stable():
    rows = [
        _row("stable_identity_evidence", "i1", "ir1"),
        _row("ordinary_general_competence", "g1", "gr1"),
    ]
    stats = mix_statistics(rows, seed=17)
    assert stats["row_count"] == 2
    assert stats["counts_by_source_class"] == {
        "ordinary_general_competence": 1,
        "stable_identity_evidence": 1,
    }
    assert stats["general_fraction"] == 0.5
    assert stats["seed"] == 17
    assert len(stats["train_digest"]) == 64
    serialized = repr(stats)
    assert "i1" not in serialized
    assert "ir1" not in serialized
