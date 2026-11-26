from __future__ import annotations

import math

import pytest

from successor.experiments.final_bank_diversity_diagnostics import (
    compute_diversity_diagnostics,
    lexical_template_key,
)


def _row(
    case_id: str,
    family_id: str,
    prompt: str,
    *,
    actor: str,
    lane: str = "behavioral",
    dimension: str = "H01",
) -> dict:
    return {
        "case_id": case_id,
        "lane": lane,
        "dimension": dimension,
        "family_id": family_id,
        "prompt": prompt,
        "source_id": f"source-{actor}",
        "generation_actor_id": actor,
        "generation_method": "synthetic_generation",
    }


def _family(
    family_id: str,
    *,
    provenance: str,
    actor: str,
    lane: str = "behavioral",
    dimension: str = "H01",
) -> dict:
    return {
        "family_id": family_id,
        "lane": lane,
        "dimension": dimension,
        "provenance_class": provenance,
        "source_actor_id": actor,
        "failure_mechanism": f"failure-{family_id}",
        "independence_claim": f"independent mechanism {family_id}",
    }


def test_lexical_template_key_masks_surface_values() -> None:
    a = lexical_template_key(
        'Return "alpha" after 17 retries for https://example.com.'
    )
    b = lexical_template_key(
        'return "beta" after 42 retries for https://openai.com.'
    )
    assert a == b


def test_diagnostics_are_aggregate_only_and_report_concentration() -> None:
    rows = [
        _row("a1", "f1", "Return 10 widgets.", actor="synth-a"),
        _row("a2", "f1", "Return 11 widgets.", actor="synth-a"),
        _row("b1", "f2", "Verify the source claim.", actor="human-1"),
        _row("b2", "f2", "Verify this source claim.", actor="human-1"),
    ]
    families = [
        _family("f1", provenance="synthetic_A", actor="synth-a"),
        _family("f2", provenance="human_seeded", actor="human-1"),
    ]
    semantic = {
        "0.75": {
            "a1": "c1",
            "a2": "c1",
            "b1": "c2",
            "b2": "c2",
        },
        "0.85": {
            "a1": "c1",
            "a2": "c1",
            "b1": "c2",
            "b2": "c3",
        },
    }
    receipt = compute_diversity_diagnostics(
        rows,
        family_manifest=families,
        semantic_cluster_assignments=semantic,
        near_duplicate_thresholds=(0.80, 0.90),
    )
    assert receipt["status"] == "DIAGNOSTICS_COMPLETE"
    assert receipt["plaintext_in_receipt"] is False
    assert "a1" not in str(receipt)
    assert "Return 10 widgets" not in str(receipt)

    group = receipt["groups"]["behavioral:H01"]
    assert group["row_count"] == 4
    assert group["family_count"] == 2
    assert group["provenance_family_counts"] == {
        "human_seeded": 1,
        "synthetic_A": 1,
    }
    assert math.isclose(group["max_family_share"], 0.5)
    assert math.isclose(group["effective_family_count"], 2.0)
    assert group["semantic_clusters"]["0.75"]["cluster_count"] == 2
    assert group["semantic_clusters"]["0.85"]["cluster_count"] == 3
    assert set(group["near_duplicate"]) == {"0.80", "0.90"}


def test_family_manifest_must_cover_every_referenced_family() -> None:
    rows = [_row("a1", "missing", "Prompt", actor="synth-a")]
    with pytest.raises(ValueError, match="family manifest missing"):
        compute_diversity_diagnostics(
            rows,
            family_manifest=[],
            semantic_cluster_assignments={
                "0.75": {"a1": "c1"},
                "0.85": {"a1": "c1"},
            },
        )


def test_family_manifest_scope_must_match_rows() -> None:
    rows = [_row("a1", "f1", "Prompt", actor="synth-a")]
    families = [
        _family(
            "f1",
            provenance="synthetic_A",
            actor="synth-a",
            dimension="H02",
        )
    ]
    with pytest.raises(ValueError, match="scope mismatch"):
        compute_diversity_diagnostics(
            rows,
            family_manifest=families,
            semantic_cluster_assignments={
                "0.75": {"a1": "c1"},
                "0.85": {"a1": "c1"},
            },
        )


def test_semantic_assignments_must_cover_exact_case_set() -> None:
    rows = [_row("a1", "f1", "Prompt", actor="synth-a")]
    families = [_family("f1", provenance="synthetic_A", actor="synth-a")]
    with pytest.raises(ValueError, match="semantic cluster case set"):
        compute_diversity_diagnostics(
            rows,
            family_manifest=families,
            semantic_cluster_assignments={
                "0.75": {},
                "0.85": {"a1": "c1"},
            },
        )


def test_rejected_counts_are_aggregate_only() -> None:
    rows = [_row("a1", "f1", "Prompt", actor="synth-a")]
    families = [_family("f1", provenance="synthetic_A", actor="synth-a")]
    receipt = compute_diversity_diagnostics(
        rows,
        family_manifest=families,
        semantic_cluster_assignments={
            "0.75": {"a1": "c1"},
            "0.85": {"a1": "c1"},
        },
        rejected_records=[
            {
                "lane": "behavioral",
                "dimension": "H01",
                "reason": "semantic_leakage",
            },
            {
                "lane": "behavioral",
                "dimension": "H01",
                "reason": "semantic_leakage",
            },
        ],
    )
    assert receipt["rejected_quarantined_counts_by_reason"] == {
        "semantic_leakage": 2
    }
    assert "case_id" not in str(receipt)
