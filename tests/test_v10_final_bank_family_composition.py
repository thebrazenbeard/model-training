from __future__ import annotations

from successor.experiments.final_bank_family_composition import (
    validate_final_bank_family_composition,
)


def _spec() -> dict:
    return {
        "schema": "V10_H01_H20_FINAL_BANK_GENERATION_SPEC_V1",
        "behavioral": {
            "dimensions": ["H01"],
            "rows_per_dimension": 6,
            "families_per_dimension": 3,
            "cases_per_family": 2,
            "family_provenance_per_dimension": {
                "synthetic_A": 1,
                "synthetic_B": 1,
                "human_seeded": 1,
            },
        },
        "adversarial": {
            "dimensions": ["H01"],
            "rows_per_dimension": 3,
            "families_per_dimension": 3,
            "cases_per_family": 1,
            "family_provenance_per_dimension": {
                "synthetic_A": 1,
                "synthetic_B": 1,
                "human_seeded": 1,
            },
        },
    }


def _binding() -> dict:
    return {
        "synthetic_custodians": {
            "A": {"actor_id": "synth-a"},
            "B": {"actor_id": "synth-b"},
        },
        "human_authors": [
            {"actor_id": "human-author"},
        ],
    }


def _row(
    case_id: str,
    lane: str,
    family_id: str,
    actor: str,
) -> dict:
    return {
        "case_id": case_id,
        "lane": lane,
        "dimension": "H01",
        "family_id": family_id,
        "generation_actor_id": actor,
    }


def _family(
    lane: str,
    family_id: str,
    provenance: str,
    source_actor_id: str,
) -> dict:
    return {
        "lane": lane,
        "dimension": "H01",
        "family_id": family_id,
        "provenance_class": provenance,
        "source_actor_id": source_actor_id,
        "failure_mechanism": f"mechanism-{family_id}",
        "independence_claim": f"independent-{family_id}",
    }


def _valid():
    rows = [
        _row("b-a-1", "behavioral", "bf-a", "synth-a"),
        _row("b-a-2", "behavioral", "bf-a", "synth-a"),
        _row("b-b-1", "behavioral", "bf-b", "synth-b"),
        _row("b-b-2", "behavioral", "bf-b", "synth-b"),
        _row("b-h-1", "behavioral", "bf-h", "human-author"),
        _row("b-h-2", "behavioral", "bf-h", "synth-a"),
        _row("a-a", "adversarial", "af-a", "synth-a"),
        _row("a-b", "adversarial", "af-b", "synth-b"),
        _row("a-h", "adversarial", "af-h", "human-author"),
    ]
    families = [
        _family("behavioral", "bf-a", "synthetic_A", "synth-a"),
        _family("behavioral", "bf-b", "synthetic_B", "synth-b"),
        _family("behavioral", "bf-h", "human_seeded", "human-author"),
        _family("adversarial", "af-a", "synthetic_A", "synth-a"),
        _family("adversarial", "af-b", "synthetic_B", "synth-b"),
        _family("adversarial", "af-h", "human_seeded", "human-author"),
    ]
    return rows, families


def test_valid_exact_family_composition_passes() -> None:
    rows, families = _valid()
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "COMPOSITION_PASS"
    assert result["reasons"] == []
    assert result["plaintext_in_receipt"] is False
    assert result["groups"]["behavioral:H01"]["provenance_family_counts"] == {
        "human_seeded": 1,
        "synthetic_A": 1,
        "synthetic_B": 1,
    }


def test_wrong_provenance_mix_holds() -> None:
    rows, families = _valid()
    families[1]["provenance_class"] = "synthetic_A"
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "HOLD"
    assert any("provenance_count:behavioral:H01:synthetic_A" in r for r in result["reasons"])


def test_human_seeded_family_requires_real_human_authored_case() -> None:
    rows, families = _valid()
    for row in rows:
        if row["family_id"] == "bf-h":
            row["generation_actor_id"] = "synth-a"
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "HOLD"
    assert "human_seeded_without_human_case:behavioral:H01:bf-h" in result["reasons"]


def test_synthetic_family_source_actor_must_match_bound_role() -> None:
    rows, families = _valid()
    families[0]["source_actor_id"] = "synth-b"
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "HOLD"
    assert "source_actor_mismatch:behavioral:H01:bf-a" in result["reasons"]


def test_family_case_count_is_exact() -> None:
    rows, families = _valid()
    rows.pop(1)
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "HOLD"
    assert "family_case_count:behavioral:H01:bf-a:1!=2" in result["reasons"]


def test_behavioral_and_adversarial_family_ids_are_disjoint() -> None:
    rows, families = _valid()
    rows[-3]["family_id"] = "bf-a"
    families[-3]["family_id"] = "bf-a"
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "HOLD"
    assert "cross_lane_family_id:bf-a" in result["reasons"]


def test_extra_manifest_family_without_rows_holds() -> None:
    rows, families = _valid()
    families.append(
        _family(
            "behavioral",
            "extra",
            "synthetic_A",
            "synth-a",
        )
    )
    result = validate_final_bank_family_composition(
        rows,
        family_manifest=families,
        generation_spec=_spec(),
        custodian_binding=_binding(),
    )
    assert result["status"] == "HOLD"
    assert "manifest_family_without_rows:behavioral:H01:extra" in result["reasons"]
