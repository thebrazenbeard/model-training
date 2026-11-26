from __future__ import annotations

import pytest

from successor.experiments.build_v10_retention_family_audit_v5 import (
    SELECTION_NAMESPACE,
    select_disjoint_family_audit_sample,
)


def _rows() -> list[dict]:
    rows = []
    for family in ("a", "b"):
        for i in range(12):
            rows.append({
                "case_id": f"{family}-{i:02d}",
                "family_id": family,
            })
    return rows


def test_v5_selection_is_exact_and_disjoint() -> None:
    rows = _rows()
    excluded = {"a-00", "a-01", "a-02", "b-00", "b-01", "b-02"}
    selected = select_disjoint_family_audit_sample(
        rows,
        excluded_case_ids=excluded,
        per_family=5,
        candidate_sha256="c" * 64,
    )
    assert len(selected) == 10
    assert not ({row["case_id"] for row in selected} & excluded)
    assert {row["family_id"] for row in selected} == {"a", "b"}
    for family in ("a", "b"):
        assert sum(row["family_id"] == family for row in selected) == 5


def test_v5_selection_is_deterministic() -> None:
    rows = _rows()
    kwargs = {
        "excluded_case_ids": {"a-00", "b-00"},
        "per_family": 5,
        "candidate_sha256": "d" * 64,
    }
    first = select_disjoint_family_audit_sample(rows, **kwargs)
    second = select_disjoint_family_audit_sample(list(reversed(rows)), **kwargs)
    assert [row["case_id"] for row in first] == [
        row["case_id"] for row in second
    ]


def test_v5_namespace_is_frozen() -> None:
    assert SELECTION_NAMESPACE == "RETENTION_FAMILY_AUDIT_V5_DISJOINT_20261002"


def test_v5_selection_fails_if_disjoint_pool_too_small() -> None:
    rows = _rows()
    excluded = {
        row["case_id"]
        for row in rows
        if row["family_id"] == "a" and row["case_id"] != "a-11"
    }
    with pytest.raises(ValueError, match="family a"):
        select_disjoint_family_audit_sample(
            rows,
            excluded_case_ids=excluded,
            per_family=5,
            candidate_sha256="e" * 64,
        )


def test_v5_selection_rejects_duplicate_case_ids() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate case_id"):
        select_disjoint_family_audit_sample(
            rows,
            excluded_case_ids=set(),
            per_family=5,
            candidate_sha256="f" * 64,
        )
