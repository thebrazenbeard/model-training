from __future__ import annotations

import json

from successor.experiments import build_v10_qwen35_retention_candidate as v1
from successor.experiments.build_v10_qwen35_retention_candidate_v2 import (
    CODING_CONTRACT_REVISION_V2,
    CODING_GENERATOR_ID_V2,
    _build_coding_v2,
    _coding_spec_v2,
)
from successor.experiments.retention_code_mutants_v2 import mutant_sources
from successor.experiments.retention_graders import (
    reference_coding_source,
    run_python_test_contract,
)


FAMILIES = tuple(v1.CODING_FAMILIES)


def _passes(source: str, tests: list[str]) -> bool:
    try:
        run_python_test_contract(source, tests)
        return True
    except Exception:
        return False


def test_v2_coding_lane_preserves_selected_case_ids() -> None:
    old = v1._build_coding(250)
    new = _build_coding_v2(250)

    assert len(old) == len(new) == 250
    assert [row["case_id"] for row in new] == [
        row["case_id"] for row in old
    ]


def test_v2_all_coding_rows_use_new_revision_and_generator() -> None:
    rows = _build_coding_v2(250)

    assert all(
        row["source_revision"] == CODING_CONTRACT_REVISION_V2
        for row in rows
    )
    assert all(
        row["generation_actor_id"] == CODING_GENERATOR_ID_V2
        for row in rows
    )
    assert all(
        CODING_CONTRACT_REVISION_V2 in row["source_id"]
        for row in rows
    )


def test_v2_rotate_left_and_chunk_contracts_are_explicit() -> None:
    rotate = _coding_spec_v2("rotate_left", 25)
    chunk = _coding_spec_v2("chunk_list", 14)

    assert "modulo" in rotate["task"].lower()
    assert "positive" in chunk["task"].lower()

    rotate_tests = "\n".join(rotate["tests"])
    chunk_tests = "\n".join(chunk["tests"])
    assert ", 6)" in rotate_tests
    assert ", 4)" in rotate_tests
    assert ", 1)" in chunk_tests
    assert "[0, 1, 2, 3, 4], 2" in chunk_tests


def test_v2_every_reference_passes_every_row_contract() -> None:
    for row in _build_coding_v2(250):
        grader = row["grader_contract"]
        source = reference_coding_source(
            grader["family"],
            grader["function"],
        )
        assert _passes(source, list(grader["tests"])), row["case_id"]


def test_v2_every_row_kills_all_frozen_family_mutants() -> None:
    for row in _build_coding_v2(250):
        grader = row["grader_contract"]
        mutants = mutant_sources(
            grader["family"],
            grader["function"],
        )
        assert len(mutants) >= 2
        surviving = [
            name
            for name, source in mutants.items()
            if _passes(source, list(grader["tests"]))
        ]
        assert surviving == [], (row["case_id"], surviving)


def test_v2_source_hash_binds_strengthened_tests() -> None:
    row = next(
        item
        for item in _build_coding_v2(250)
        if item["family_id"] == "generated-code:rotate_left"
    )
    grader = row["grader_contract"]
    source_record = {
        "family": grader["family"],
        "variant": int(row["case_id"].rsplit("-", 1)[1]),
        "function": grader["function"],
        "signature": row["prompt"].splitlines()[0].removeprefix(
            "Implement this Python function exactly: "
        ),
        "task": row["prompt"].splitlines()[1].removeprefix("Contract: "),
        "tests": list(grader["tests"]),
    }

    assert row["source_hash"] == v1.sha256_json(source_record)


def test_v2_strengthened_tests_are_not_single_mutant_patches() -> None:
    counts = {}
    for family in FAMILIES:
        spec = _coding_spec_v2(family, 7)
        counts[family] = len(spec["tests"])

    assert all(count >= 4 for count in counts.values())
    assert counts["rotate_left"] >= 5
    assert counts["chunk_list"] >= 5


def _synthetic_countries(count: int = 260) -> list[dict]:
    countries = []
    for index in range(count):
        first = chr(ord("A") + (index // 26) % 26)
        second = chr(ord("A") + index % 26)
        code = first + second
        countries.append({
            "code": code,
            "name": f"Country {index:03d}",
            "regions": index % 17 + 1,
            "settlements": index * 3 + 5,
        })
    return countries


def test_v2_full_candidate_rows_replace_only_coding_lane() -> None:
    from successor.experiments.build_v10_qwen35_retention_candidate_v2 import (
        build_candidate_rows_v2,
    )

    countries = _synthetic_countries()
    old_rows = v1.build_candidate_rows(countries)
    new_rows = build_candidate_rows_v2(countries)

    assert len(old_rows) == len(new_rows) == 1500
    assert [row["case_id"] for row in new_rows] == [
        row["case_id"] for row in old_rows
    ]

    old_by_id = {row["case_id"]: row for row in old_rows}
    changed = []
    for row in new_rows:
        old = old_by_id[row["case_id"]]
        if row["category"] == "coding":
            assert row != old
            changed.append(row["case_id"])
        else:
            assert row == old

    assert len(changed) == 250


def test_v2_keeps_existing_python_grader_runtime_id() -> None:
    rows = _build_coding_v2(250)
    assert all(
        row["grader_contract"]["grader_id"] == "python_unit_tests_v1"
        for row in rows
    )
    assert all(
        row["grader_contract"]["grader_version"] == "2"
        for row in rows
    )
