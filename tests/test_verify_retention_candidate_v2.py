from __future__ import annotations

from successor.experiments import build_v10_qwen35_retention_candidate as v1
from successor.experiments.build_v10_qwen35_retention_candidate_v2 import (
    build_candidate_rows_v2,
)
from successor.experiments.verify_retention_candidate_v2 import (
    verify_candidate_v2_rows,
)


def _countries(count: int = 260) -> list[dict]:
    rows = []
    for index in range(count):
        code = chr(65 + (index // 26) % 26) + chr(65 + index % 26)
        rows.append({
            "code": code,
            "name": f"Country {index}",
            "regions": index % 17 + 1,
            "settlements": index * 3 + 5,
        })
    return rows


def test_candidate_v2_verifier_accepts_strengthened_successor() -> None:
    countries = _countries()
    old = v1.build_candidate_rows(countries)
    new = build_candidate_rows_v2(countries)

    result = verify_candidate_v2_rows(old, new)

    assert result["status"] == "CANDIDATE_V2_MUTATION_ADEQUATE"
    assert result["case_count"] == 1500
    assert result["coding_rows"] == 250
    assert result["noncoding_changed"] == 0
    assert result["reference_failures"] == []
    assert result["mutant_survivors"] == []


def test_candidate_v2_verifier_rejects_v1_coding_row_regression() -> None:
    countries = _countries()
    old = v1.build_candidate_rows(countries)
    new = build_candidate_rows_v2(countries)
    old_by_id = {row["case_id"]: row for row in old}

    target = next(
        index
        for index, row in enumerate(new)
        if row["case_id"] == "ret-code-rotate_left-025"
    )
    new[target] = old_by_id["ret-code-rotate_left-025"]

    result = verify_candidate_v2_rows(old, new)

    assert result["status"] == "CANDIDATE_V2_HOLD"
    assert "ret-code-rotate_left-025" in result["coding_revision_failures"]
    assert any(
        item["case_id"] == "ret-code-rotate_left-025"
        for item in result["mutant_survivors"]
    )
