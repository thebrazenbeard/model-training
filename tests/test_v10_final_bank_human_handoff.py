from __future__ import annotations

import copy

from successor.experiments.final_bank_human_handoff import (
    build_human_handoff_work_order,
    validate_human_review_receipt,
)


SEMANTIC = {"H01", "H03", "H04", "H05", "H06", "H11", "H14", "H15", "H18"}


def test_handoff_work_order_freezes_exact_human_seed_and_review_slots() -> None:
    order = build_human_handoff_work_order()

    assert order["schema"] == "V10_FINAL_BANK_HUMAN_HANDOFF_WORK_ORDER_V1"
    assert order["status"] == "UNASSIGNED_EXTERNAL_HUMAN_WORK"
    assert order["human_seed_slot_count"] == 280
    assert order["mandatory_semantic_review_slot_count"] == 5400
    assert order["mixed_dimension_case_slots_pending_classification"] == 6600

    seed_slots = order["human_seed_slots"]
    assert len({row["slot_id"] for row in seed_slots}) == 280
    assert all(row["assigned_author_id"] is None for row in seed_slots)
    assert all(row["required_genuine_human_authored_cases"] == 1 for row in seed_slots)
    assert sum(row["lane"] == "behavioral" for row in seed_slots) == 200
    assert sum(row["lane"] == "adversarial" for row in seed_slots) == 80

    review_slots = order["mandatory_semantic_review_slots"]
    assert len({row["slot_id"] for row in review_slots}) == 5400
    assert all(row["dimension"] in SEMANTIC for row in review_slots)
    assert all(row["assigned_reviewer_id"] is None for row in review_slots)
    assert all(row["reviewer_class"] == "HUMAN_REVIEW" for row in review_slots)
    assert sum(row["lane"] == "behavioral" for row in review_slots) == 4500
    assert sum(row["lane"] == "adversarial" for row in review_slots) == 900

    assert {
        row["dimension"] for row in order["mixed_dimension_review_classification"]
    } == {f"H{i:02d}" for i in range(1, 21)} - SEMANTIC
    assert all(
        "prompt" not in row and "answer" not in row and "case" not in row
        for row in seed_slots + review_slots
    )


def _receipt() -> dict:
    return {
        "case_id": "case-001",
        "verdict": "ADMIT",
        "reviewer_id": "human-reviewer-1",
        "reviewer_class": "HUMAN_REVIEW",
        "provider": "INDEPENDENT_HUMAN_CUSTODY",
        "runtime": "sealed-review-surface-v1",
        "independent_from_generation": True,
        "subject_digest": "a" * 64,
        "grader_digest": "b" * 64,
        "artifact_digest": "c" * 64,
        "review_timestamp": "2026-10-03T18:00:00Z",
        "review_bundle_sha256": "d" * 64,
        "review_answers": {
            "prompt_well_posed": True,
            "rubric_represents_target_dimension": True,
            "materially_wrong_answer_could_pass": False,
            "hidden_dependence_on_another_family": False,
            "severity_or_critical_failure_rule_unambiguous": True,
            "prior_train_validation_final_leakage": False,
            "usable_without_private_or_unlicensed_material": True,
        },
    }


def test_valid_human_review_receipt_passes_stricter_frozen_handoff() -> None:
    result = validate_human_review_receipt(
        _receipt(),
        expected_case_id="case-001",
        generation_actor_id="synthetic-A",
        expected_subject_digest="a" * 64,
        expected_grader_digest="b" * 64,
        bound_reviewer_ids={"human-reviewer-1"},
    )
    assert result["status"] == "HUMAN_REVIEW_RECEIPT_PASS"
    assert result["reasons"] == []
    assert result["verdict"] == "ADMIT"


def test_human_review_receipt_fails_on_self_review_wrong_digest_or_missing_question() -> None:
    value = _receipt()
    value["reviewer_id"] = "human-author-1"
    value["subject_digest"] = "e" * 64
    value["review_answers"].pop("usable_without_private_or_unlicensed_material")

    result = validate_human_review_receipt(
        value,
        expected_case_id="case-001",
        generation_actor_id="human-author-1",
        expected_subject_digest="a" * 64,
        expected_grader_digest="b" * 64,
        bound_reviewer_ids={"human-author-1"},
    )
    assert result["status"] == "HOLD"
    assert "reviewer_not_independent_from_generation" in result["reasons"]
    assert "subject_digest_mismatch" in result["reasons"]
    assert "review_answer_key_set_mismatch" in result["reasons"]


def test_unbound_or_model_reviewer_cannot_satisfy_human_receipt() -> None:
    value = _receipt()
    value["reviewer_id"] = "model-reviewer"
    value["reviewer_class"] = "EXTERNAL_MODEL_REVIEW"
    result = validate_human_review_receipt(
        value,
        expected_case_id="case-001",
        generation_actor_id="synthetic-A",
        expected_subject_digest="a" * 64,
        expected_grader_digest="b" * 64,
        bound_reviewer_ids={"human-reviewer-1"},
    )
    assert result["status"] == "HOLD"
    assert "reviewer_class_not_human" in result["reasons"]
    assert "reviewer_not_bound" in result["reasons"]
