from __future__ import annotations

import json
from pathlib import Path

from successor.experiments.run_retention_audit_arch_v2_mechanical import (
    run_mechanical_audit,
)


ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "successor/evaluation/v10_qwen35/retention_candidate_v1.jsonl"


def _rows(case_ids: set[str]) -> list[dict]:
    found = []
    for line in CANDIDATE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["case_id"] in case_ids:
            found.append(row)
    assert {row["case_id"] for row in found} == case_ids
    return found


def test_mechanical_runner_binds_qualification_and_all_rows() -> None:
    rows = _rows({
        "ret-code-chunk_list-037",
        "ret-if-forbidden_character-043",
    })

    result = run_mechanical_audit(rows, expected_count=2)

    assert result["status"] == "MECHANICAL_VALIDATED"
    assert result["qualification"]["status"] == "VALIDATORS_QUALIFIED"
    assert result["reviewed"] == 2
    assert result["row_status_counts"] == {"MECHANICAL_VALID": 2}


def test_mechanical_runner_holds_on_count_mismatch() -> None:
    rows = _rows({"ret-code-chunk_list-037"})

    result = run_mechanical_audit(rows, expected_count=2)

    assert result["status"] == "MECHANICAL_HOLD"
    assert "row_count_mismatch" in result["reasons"]
