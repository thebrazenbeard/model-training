from __future__ import annotations

from successor.experiments.audit_v10_retention_diversity import (
    analyze_retention_diversity,
    lexical_template,
)


def _row(case_id: str, category: str, family: str, prompt: str) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "family_id": family,
        "prompt": prompt,
        "source_id": f"source:{case_id}",
        "source_revision": "rev-a",
        "source_terms": "CC0-1.0",
        "generation_method": "deterministic-v1",
    }


def test_lexical_template_collapses_values_but_keeps_scaffold() -> None:
    a = lexical_template('Return exactly 3 facts for "US".')
    b = lexical_template('Return exactly 9 facts for "CA".')
    assert a == b
    assert a != lexical_template('Summarize exactly 9 facts for "CA".')


def test_diversity_report_exposes_family_concentration() -> None:
    rows = [
        _row("a1", "knowledge", "family-a", "Return exactly 3 facts for US."),
        _row("a2", "knowledge", "family-a", "Return exactly 9 facts for CA."),
        _row("b1", "knowledge", "family-b", "Name the capital of France."),
        _row("b2", "knowledge", "family-b", "Name the capital of Spain."),
    ]
    report = analyze_retention_diversity(rows)
    category = report["categories"]["knowledge"]
    assert category["rows"] == 4
    assert category["unique_families"] == 2
    assert category["max_family_share"] == 0.5
    assert category["effective_family_count"] == 2.0
    assert category["difficulty"]["status"] == "UNAVAILABLE"


def test_near_duplicate_density_is_descriptive_not_a_gate() -> None:
    rows = [
        _row("a", "instruction", "f1", "Write alpha beta gamma delta epsilon theta iota kappa lambda."),
        _row("b", "instruction", "f2", "Write alpha beta gamma delta epsilon theta iota kappa mu."),
        _row("c", "instruction", "f3", "Explain orbital mechanics briefly."),
    ]
    report = analyze_retention_diversity(rows)
    category = report["categories"]["instruction"]
    assert category["near_duplicate_pairs"]["thresholds"]["0.80"]["pairs"] == 1
    assert report["status"] == "DIAGNOSTIC_ONLY_NO_PASS_FAIL_GATE"


def test_difficulty_distribution_is_reported_when_present() -> None:
    rows = [
        {**_row("a", "coding", "f1", "Task one."), "difficulty": "easy"},
        {**_row("b", "coding", "f2", "Task two."), "difficulty": "hard"},
        {**_row("c", "coding", "f3", "Task three."), "difficulty": "easy"},
    ]
    difficulty = analyze_retention_diversity(rows)["categories"]["coding"]["difficulty"]
    assert difficulty == {
        "status": "AVAILABLE",
        "counts": {"easy": 2, "hard": 1},
    }
