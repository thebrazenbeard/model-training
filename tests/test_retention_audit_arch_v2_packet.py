from __future__ import annotations

import json
from pathlib import Path

import pytest

from successor.experiments.build_retention_audit_arch_v2_packet import (
    RISK_TIEBREAK_NAMESPACE,
    SENTINEL_NAMESPACE,
    packet_bytes,
    risk_tuple,
    select_arch_v2_sample,
)


ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "successor/evaluation/v10_qwen35/retention_candidate_v1.jsonl"
V1V4 = ROOT / "successor/evaluation/v10_qwen35/retention_family_audit_v1v4.packet.jsonl"
V5 = ROOT / "successor/evaluation/v10_qwen35/retention_family_audit_v5.packet.jsonl"
CANDIDATE_SHA = "37161023afd97d849733455db41999e6ce6e79465ad534b68a1784d25d9f679a"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _synthetic_rows() -> list[dict]:
    rows = []
    for family in ("a", "b"):
        for index in range(12):
            rows.append({
                "case_id": f"{family}-{index:02d}",
                "family_id": family,
                "prompt": "x" * (index + 1),
                "grader_contract": {"kind": "deterministic"},
                "source_id": f"source:{family}:{index}",
            })
    return rows


def test_risk_tuple_uses_frozen_components() -> None:
    row = {
        "case_id": "c-1",
        "family_id": "evidence-calibration:mode-2",
        "prompt": "abc",
        "grader_contract": {"kind": "deterministic", "answer_key": "NOT_ENOUGH_INFO"},
        "source_id": "source:country:AA",
    }

    value = risk_tuple(row, candidate_sha256="a" * 64)

    assert value[:4] == (3, 1, 1, 3)
    assert len(value[4]) == 64
    assert RISK_TIEBREAK_NAMESPACE == "ARCH_V2_RISK_TIEBREAK"
    assert SENTINEL_NAMESPACE == "ARCH_V2_SENTINEL"


def test_selection_is_exact_disjoint_and_input_order_independent() -> None:
    rows = _synthetic_rows()
    excluded = {"a-00", "a-01", "b-00", "b-01"}

    first = select_arch_v2_sample(
        rows,
        excluded_case_ids=excluded,
        candidate_sha256="b" * 64,
    )
    second = select_arch_v2_sample(
        list(reversed(rows)),
        excluded_case_ids=excluded,
        candidate_sha256="b" * 64,
    )

    assert [row["case_id"] for row in first] == [
        row["case_id"] for row in second
    ]
    assert len(first) == 10
    assert not ({row["case_id"] for row in first} & excluded)
    assert all(
        sum(row["family_id"] == family for row in first) == 5
        for family in ("a", "b")
    )


def test_current_candidate_has_130_fresh_rows_zero_predecessor_overlap() -> None:
    rows = _read_jsonl(CANDIDATE)
    predecessor_ids = {
        row["case_id"]
        for row in _read_jsonl(V1V4) + _read_jsonl(V5)
    }

    selected = select_arch_v2_sample(
        rows,
        excluded_case_ids=predecessor_ids,
        candidate_sha256=CANDIDATE_SHA,
    )

    selected_ids = {row["case_id"] for row in selected}
    assert len(selected) == 130
    assert len(selected_ids) == 130
    assert not (selected_ids & predecessor_ids)
    assert len({row["family_id"] for row in selected}) == 26
    assert all(
        sum(row["family_id"] == family for row in selected) == 5
        for family in {row["family_id"] for row in selected}
    )


def test_packet_serialization_is_byte_reproducible() -> None:
    rows = _synthetic_rows()
    selected = select_arch_v2_sample(
        rows,
        excluded_case_ids=set(),
        candidate_sha256="c" * 64,
    )

    assert packet_bytes(selected) == packet_bytes(list(selected))


def test_selection_rejects_duplicate_case_ids() -> None:
    rows = _synthetic_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="duplicate case_id"):
        select_arch_v2_sample(
            rows,
            excluded_case_ids=set(),
            candidate_sha256="d" * 64,
        )


def test_selection_fails_if_fresh_pool_has_fewer_than_five() -> None:
    rows = _synthetic_rows()
    excluded = {
        row["case_id"]
        for row in rows
        if row["family_id"] == "a" and row["case_id"] not in {"a-08", "a-09", "a-10", "a-11"}
    }

    with pytest.raises(ValueError, match="family a"):
        select_arch_v2_sample(
            rows,
            excluded_case_ids=excluded,
            candidate_sha256="e" * 64,
        )


def test_builder_enriches_selected_rows_and_binds_packet_hash() -> None:
    from successor.experiments.build_retention_audit_arch_v2_packet import (
        build_arch_v2_packet,
    )
    from successor.experiments.retention_audit_arch_v2_validators import (
        sha256_json,
    )

    rows = []
    for index in range(8):
        constraints = {
            "type": "token_exact_count",
            "token": f"V{index:03d}",
            "count": 1,
        }
        prompt = (
            f"Write one short sentence about careful measurement. Include the "
            f"exact token V{index:03d} exactly 1 time(s)."
        )
        evidence = {
            "family": "token_count",
            "variant": index,
            "prompt": prompt,
            "constraints": constraints,
        }
        rows.append({
            "case_id": f"generated-{index:03d}",
            "category": "instruction_following",
            "family_id": "generated-instruction:token_count",
            "prompt": prompt,
            "source_id": (
                "vera_model_training:ARCH_V2_TEST:"
                f"instruction:token_count:{index}"
            ),
            "source_revision": "ARCH_V2_TEST",
            "source_terms": "INTERNAL_OBJECTIVE_CONTRACT",
            "source_hash": sha256_json(evidence),
            "generation_method": "test",
            "grader_contract": {
                "kind": "deterministic",
                "grader_id": "instruction_contract_v1",
                "grader_version": "1",
                "answer_key_digest": sha256_json(constraints),
                "constraints": constraints,
            },
        })

    packet, manifest = build_arch_v2_packet(
        rows,
        excluded_case_ids=set(),
        candidate_sha256="f" * 64,
        predecessor_bindings=[
            {"name": "none", "sha256": "0" * 64, "case_count": 0},
        ],
    )

    assert len(packet) == 5
    assert all("source_evidence" in row for row in packet)
    assert manifest["sample_rows"] == 5
    assert manifest["packet_sha256"] == __import__("hashlib").sha256(
        packet_bytes(packet)
    ).hexdigest()
