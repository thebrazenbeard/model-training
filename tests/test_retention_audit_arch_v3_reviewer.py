from __future__ import annotations

import json

from successor.experiments.retention_audit_arch_v2_reviewer import (
    build_reviewer_qualification_controls as build_v2_controls,
)
from successor.experiments.retention_audit_arch_v3_reviewer import (
    build_reviewer_qualification_controls_v3,
    parse_qualification_batch_response_v3,
    qualify_reviewer_v3,
    qualification_batch_prompt_v3,
    run_batched_reviewer_qualification_v3,
)


def test_v3_controls_are_fresh_balanced_and_disjoint_from_v2() -> None:
    controls = build_reviewer_qualification_controls_v3()
    v2_ids = {row["case_id"] for row in build_v2_controls()}
    defect_classes = {
        "referent_ambiguity",
        "prompt_grader_semantic_mismatch",
        "contradictory_instructions",
        "underspecified_output_format",
        "evidence_policy_ambiguity",
        "scope_negation_ambiguity",
    }

    assert len(controls) == 48
    assert {row["defect_class"] for row in controls} == defect_classes
    assert not ({row["case_id"] for row in controls} & v2_ids)
    assert all(row["case_id"].startswith("arch-v3-reviewer-") for row in controls)
    for defect_class in defect_classes:
        rows = [row for row in controls if row["defect_class"] == defect_class]
        assert sum(row["expected_defect"] for row in rows) == 4
        assert sum(not row["expected_defect"] for row in rows) == 4


def test_v3_prompt_hides_labels_and_uses_none_witness_sentinel() -> None:
    controls = build_reviewer_qualification_controls_v3()[:4]
    prompt = qualification_batch_prompt_v3(controls)

    assert "expected_defect" not in prompt
    assert "exactly NONE" in prompt
    for control in controls:
        assert control["prompt"] in prompt
        assert control["defect_class"] in prompt


def test_v3_parser_preserves_contract_violation_for_scoring() -> None:
    controls = build_reviewer_qualification_controls_v3()[:2]
    raw = json.dumps({
        "reviews": [
            {
                "case_id": controls[0]["case_id"],
                "observed_defect": True,
                "defect_class": controls[0]["defect_class"],
                "witness": None,
                "reason": "defect claimed but no witness supplied",
            },
            {
                "case_id": controls[1]["case_id"],
                "observed_defect": False,
                "defect_class": controls[1]["defect_class"],
                "witness": "NONE",
                "reason": "no defect",
            },
        ]
    })

    parsed = parse_qualification_batch_response_v3(raw, controls=controls)

    assert parsed[0]["witness"] is None
    assert parsed[1]["witness"] == "NONE"


def test_v3_qualification_scores_missing_witness_as_contradiction() -> None:
    controls = build_reviewer_qualification_controls_v3()

    def call(control: dict) -> dict:
        witness = "concrete alternate interpretation" if control["expected_defect"] else "NONE"
        if control["expected_defect"] and control["case_id"].endswith("-0"):
            witness = None
        return {
            "case_id": control["case_id"],
            "observed_defect": control["expected_defect"],
            "defect_class": control["defect_class"],
            "witness": witness,
            "reason": "test",
        }

    result = qualify_reviewer_v3(call)

    assert result["status"] == "REVIEWER_UNQUALIFIED"
    assert result["sensitivity"] == [24, 24]
    assert result["specificity"] == [24, 24]
    assert len(result["structured_contradictions"]) == 6


def test_v3_batched_qualification_completes_all_controls_without_retrying_contradictions() -> None:
    controls = build_reviewer_qualification_controls_v3()
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
                        None
                        if control["expected_defect"] and start == 0
                        else "concrete witness"
                        if control["expected_defect"]
                        else "NONE"
                    ),
                    "reason": "qualification",
                }
                for control in batch
            ]
        })

    result = run_batched_reviewer_qualification_v3(
        call_text=call_text,
        batch_size=4,
        max_attempts=3,
    )

    assert len(calls) == 12
    assert result["batch_count"] == 12
    assert result["status"] == "REVIEWER_UNQUALIFIED"
    assert result["structured_contradictions"]
    assert all(batch["attempts"] == 1 for batch in result["batches"])
