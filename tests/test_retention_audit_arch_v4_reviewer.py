from __future__ import annotations

import json

from successor.experiments.retention_audit_arch_v2_reviewer import (
    build_reviewer_qualification_controls as build_v2_controls,
)
from successor.experiments.retention_audit_arch_v3_reviewer import (
    build_reviewer_qualification_controls_v3,
)
from successor.experiments.retention_audit_arch_v4_reviewer import (
    build_reviewer_qualification_controls_v4,
    meaningful_witness_v4,
    parse_qualification_batch_response_v4,
    qualify_reviewer_v4,
    run_batched_reviewer_qualification_v4,
)


def test_v4_controls_are_fresh_balanced_and_disjoint() -> None:
    controls = build_reviewer_qualification_controls_v4()
    prior_ids = {
        row["case_id"]
        for row in build_v2_controls()
    } | {
        row["case_id"]
        for row in build_reviewer_qualification_controls_v3()
    }
    classes = {row["defect_class"] for row in controls}

    assert len(controls) == 48
    assert len(classes) == 6
    assert not ({row["case_id"] for row in controls} & prior_ids)
    assert all(row["case_id"].startswith("arch-v4-reviewer-") for row in controls)
    for defect_class in classes:
        rows = [row for row in controls if row["defect_class"] == defect_class]
        assert sum(row["expected_defect"] for row in rows) == 4
        assert sum(not row["expected_defect"] for row in rows) == 4


def test_v4_meaningful_witness_separates_content_from_json_type() -> None:
    assert meaningful_witness_v4("specific alternate parse") is True
    assert meaningful_witness_v4(["alternate A", "alternate B"]) is True
    assert meaningful_witness_v4({"ambiguity": "two valid shapes"}) is True

    for value in (
        None,
        "",
        "   ",
        "NONE",
        [],
        {},
        0,
        1,
        True,
        False,
    ):
        assert meaningful_witness_v4(value) is False


def test_v4_parser_accepts_structured_witness_without_transport_failure() -> None:
    controls = build_reviewer_qualification_controls_v4()[:4]
    raw = json.dumps({
        "reviews": [
            {
                "case_id": controls[0]["case_id"],
                "observed_defect": True,
                "defect_class": controls[0]["defect_class"],
                "witness": {"ambiguity": ["A", "B"]},
                "reason": "two referents remain valid",
            },
            {
                "case_id": controls[1]["case_id"],
                "observed_defect": False,
                "defect_class": controls[1]["defect_class"],
                "witness": "NONE",
                "reason": "referent is explicit",
            },
            {
                "case_id": controls[2]["case_id"],
                "observed_defect": True,
                "defect_class": controls[2]["defect_class"],
                "witness": ["reading one", "reading two"],
                "reason": "two readings",
            },
            {
                "case_id": controls[3]["case_id"],
                "observed_defect": False,
                "defect_class": controls[3]["defect_class"],
                "witness": "NONE",
                "reason": "clear",
            },
        ]
    })

    parsed = parse_qualification_batch_response_v4(raw, controls=controls)

    assert parsed[0]["witness"] == {"ambiguity": ["A", "B"]}
    assert parsed[0]["witness_text"] == '{"ambiguity":["A","B"]}'
    assert parsed[2]["witness"] == ["reading one", "reading two"]
    assert parsed[2]["witness_text"] == '["reading one","reading two"]'


def test_v4_qualification_scores_witness_content_not_representation() -> None:
    controls = build_reviewer_qualification_controls_v4()

    def call(control: dict) -> dict:
        if control["expected_defect"]:
            witness = {"example": "concrete semantic conflict"}
        else:
            witness = "NONE"
        if control["case_id"].endswith("-defect-0"):
            witness = {}
        return {
            "case_id": control["case_id"],
            "observed_defect": control["expected_defect"],
            "defect_class": control["defect_class"],
            "witness": witness,
            "witness_text": (
                json.dumps(witness, sort_keys=True, separators=(",", ":"))
                if not isinstance(witness, str)
                else witness
            ),
            "reason": "test",
        }

    result = qualify_reviewer_v4(call)

    assert result["status"] == "REVIEWER_UNQUALIFIED"
    assert result["sensitivity"] == [24, 24]
    assert result["specificity"] == [24, 24]
    assert len(result["structured_contradictions"]) == 6


def test_v4_batched_qualification_does_not_retry_scoreable_witness_types() -> None:
    controls = build_reviewer_qualification_controls_v4()
    calls = []

    def call_text(prompt: str) -> str:
        start = len(calls) * 4
        batch = controls[start:start + 4]
        calls.append(prompt)
        return json.dumps({
            "reviews": [
                {
                    "case_id": control["case_id"],
                    "observed_defect": control["expected_defect"],
                    "defect_class": control["defect_class"],
                    "witness": (
                        {"example": "structured witness"}
                        if control["expected_defect"]
                        else "NONE"
                    ),
                    "reason": "qualification",
                }
                for control in batch
            ]
        })

    result = run_batched_reviewer_qualification_v4(
        call_text=call_text,
        batch_size=4,
        max_attempts=3,
    )

    assert result["status"] == "REVIEWER_QUALIFIED"
    assert result["batch_count"] == 12
    assert len(calls) == 12
    assert all(batch["attempts"] == 1 for batch in result["batches"])
