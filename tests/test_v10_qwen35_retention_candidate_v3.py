from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from successor.experiments.build_v10_qwen35_retention_candidate_v3 import (
    CHUNK_CONTRACT_REVISION_V3,
    CHUNK_GENERATOR_ID_V3,
    CHUNK_TASK_V3,
    repair_candidate_v3_rows,
)
from successor.experiments.retention_audit_arch_v2_validators import (
    resolve_source_evidence,
    validate_row_mechanically,
)
from successor.experiments.retention_code_mutants_v2 import mutant_sources
from successor.experiments.retention_graders import (
    reference_coding_source,
    run_python_test_contract,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "successor/evaluation/v10_qwen35"
V2_PATH = EVAL / "retention_candidate_v2.jsonl"
PACKETS = [
    EVAL / "retention_family_audit_v1v4.packet.jsonl",
    EVAL / "retention_family_audit_v5.packet.jsonl",
    EVAL / "retention_arch_v2.semantic.packet.jsonl",
    EVAL / "retention_arch_v6.semantic.packet.jsonl",
]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _consumed_ids() -> set[str]:
    used: set[str] = set()
    for path in PACKETS:
        used.update(row["case_id"] for row in _read_jsonl(path))
    return used


def _passes(source: str, tests: list[str]) -> bool:
    try:
        run_python_test_contract(source, tests)
        return True
    except Exception:
        return False


def test_v3_replenishes_only_consumed_coding_rows() -> None:
    v2 = _read_jsonl(V2_PATH)
    consumed = _consumed_ids()
    v3 = repair_candidate_v3_rows(v2, consumed_case_ids=consumed)

    assert len(v3) == 1500
    assert Counter(row["category"] for row in v3) == Counter(
        row["category"] for row in v2
    )
    assert Counter(row["family_id"] for row in v3) == Counter(
        row["family_id"] for row in v2
    )

    old_by_id = {row["case_id"]: row for row in v2}
    new_by_id = {row["case_id"]: row for row in v3}
    removed = set(old_by_id) - set(new_by_id)
    added = set(new_by_id) - set(old_by_id)

    assert len(removed) == 12
    assert len(added) == 12
    assert removed <= consumed
    assert not (added & consumed)

    affected = {
        "generated-code:chunk_list": 2,
        "generated-code:count_vowels": 1,
        "generated-code:digit_sum": 1,
        "generated-code:flatten_once": 5,
        "generated-code:reverse_words": 3,
    }
    assert Counter(old_by_id[cid]["family_id"] for cid in removed) == Counter(
        affected
    )
    assert Counter(new_by_id[cid]["family_id"] for cid in added) == Counter(
        affected
    )

    for cid in set(old_by_id) & set(new_by_id):
        old = old_by_id[cid]
        new = new_by_id[cid]
        if old["family_id"] == "generated-code:chunk_list":
            assert old != new
        else:
            assert old == new


def test_v3_chunk_contract_is_explicitly_conditional() -> None:
    v3 = repair_candidate_v3_rows(
        _read_jsonl(V2_PATH),
        consumed_case_ids=_consumed_ids(),
    )
    rows = [
        row for row in v3
        if row["family_id"] == "generated-code:chunk_list"
    ]

    assert len(rows) == 23
    for row in rows:
        assert row["source_revision"] == CHUNK_CONTRACT_REVISION_V3
        assert row["generation_actor_id"] == CHUNK_GENERATOR_ID_V3
        assert CHUNK_CONTRACT_REVISION_V3 in row["source_id"]
        assert f"Contract: {CHUNK_TASK_V3}" in row["prompt"]
        assert "if elements remain after the full-size chunks" in row["prompt"]


def test_v3_restores_five_fresh_rows_per_family() -> None:
    consumed = _consumed_ids()
    v3 = repair_candidate_v3_rows(
        _read_jsonl(V2_PATH),
        consumed_case_ids=consumed,
    )
    total = Counter(row["family_id"] for row in v3)
    fresh = Counter(
        row["family_id"]
        for row in v3
        if row["case_id"] not in consumed
    )

    assert len(total) == 26
    assert all(fresh[family_id] >= 5 for family_id in total)


def test_v3_changed_coding_rows_are_mechanically_valid() -> None:
    v2 = _read_jsonl(V2_PATH)
    v3 = repair_candidate_v3_rows(
        v2,
        consumed_case_ids=_consumed_ids(),
    )
    old_by_id = {row["case_id"]: row for row in v2}

    changed = [
        row for row in v3
        if old_by_id.get(row["case_id"]) != row
    ]
    assert len(changed) == 33

    for row in changed:
        evidence = resolve_source_evidence(row)
        result = validate_row_mechanically(row, source_evidence=evidence)
        assert result["status"] == "MECHANICAL_VALID", (
            row["case_id"],
            result,
        )


def test_v3_all_coding_rows_remain_mutation_adequate() -> None:
    v3 = repair_candidate_v3_rows(
        _read_jsonl(V2_PATH),
        consumed_case_ids=_consumed_ids(),
    )
    coding = [row for row in v3 if row["category"] == "coding"]

    assert len(coding) == 250
    for row in coding:
        grader = row["grader_contract"]
        family = grader["family"]
        function_name = grader["function"]
        tests = list(grader["tests"])

        assert _passes(
            reference_coding_source(family, function_name),
            tests,
        ), row["case_id"]

        survivors = [
            name
            for name, source in mutant_sources(
                family,
                function_name,
            ).items()
            if _passes(source, tests)
        ]
        assert survivors == [], (row["case_id"], survivors)


def test_v3_full_preflight_is_structure_ready() -> None:
    from successor.experiments import build_v10_qwen35_retention_candidate as v1
    from successor.experiments.build_v10_qwen35_retention_candidate_v3 import (
        build_candidate_v3,
    )

    rows, manifest = build_candidate_v3(
        parent_rows=_read_jsonl(V2_PATH),
        consumed_case_ids=_consumed_ids(),
        exclusion_hashes=v1.load_exclusion_hashes(
            ROOT
            / "successor/experiments/V10_QWEN35_EXCLUSION_PROMPT_HASHES_V2.txt"
        ),
    )
    assert len(rows) == 1500
    assert manifest["preflight"]["status"] == "STRUCTURE_READY"
