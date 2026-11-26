from __future__ import annotations

import re
from datetime import datetime


DIMENSIONS = tuple(f"H{i:02d}" for i in range(1, 21))
SEMANTIC_HUMAN_DIMENSIONS = frozenset(
    {"H01", "H03", "H04", "H05", "H06", "H11", "H14", "H15", "H18"}
)
MIXED_DIMENSIONS = tuple(
    dimension
    for dimension in DIMENSIONS
    if dimension not in SEMANTIC_HUMAN_DIMENSIONS
)
REVIEW_QUESTIONS = (
    "prompt_well_posed",
    "rubric_represents_target_dimension",
    "materially_wrong_answer_could_pass",
    "hidden_dependence_on_another_family",
    "severity_or_critical_failure_rule_unambiguous",
    "prior_train_validation_final_leakage",
    "usable_without_private_or_unlicensed_material",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _family_id(
    *,
    lane: str,
    dimension: str,
    provenance_class: str,
    family_index: int,
) -> str:
    return (
        f"v10-final-{lane}-{dimension.lower()}-"
        f"{provenance_class.lower()}-{family_index:03d}"
    )


def _all_family_specs(*, lane: str, dimension: str) -> list[tuple[str, int, int]]:
    if lane == "behavioral":
        return [
            ("synthetic_A", index, 10) for index in range(1, 21)
        ] + [
            ("synthetic_B", index, 10) for index in range(21, 41)
        ] + [
            ("human_seeded", index, 10) for index in range(41, 51)
        ]
    if lane == "adversarial":
        return [
            ("synthetic_A", index, 5) for index in range(1, 9)
        ] + [
            ("synthetic_B", index, 5) for index in range(9, 17)
        ] + [
            ("human_seeded", index, 5) for index in range(17, 21)
        ]
    raise ValueError(f"unknown lane:{lane}")


def build_human_handoff_work_order() -> dict:
    human_seed_slots: list[dict] = []
    mandatory_review_slots: list[dict] = []

    for dimension in DIMENSIONS:
        for lane in ("behavioral", "adversarial"):
            for provenance_class, family_index, cases_per_family in _all_family_specs(
                lane=lane,
                dimension=dimension,
            ):
                family_id = _family_id(
                    lane=lane,
                    dimension=dimension,
                    provenance_class=provenance_class,
                    family_index=family_index,
                )
                if provenance_class == "human_seeded":
                    human_seed_slots.append(
                        {
                            "slot_id": f"human-seed::{family_id}",
                            "lane": lane,
                            "dimension": dimension,
                            "family_id": family_id,
                            "family_index": family_index,
                            "provenance_class": "human_seeded",
                            "required_genuine_human_authored_cases": 1,
                            "assigned_author_id": None,
                            "submission_case_id": None,
                            "submission_subject_digest": None,
                        }
                    )

                if dimension in SEMANTIC_HUMAN_DIMENSIONS:
                    for case_ordinal in range(1, cases_per_family + 1):
                        mandatory_review_slots.append(
                            {
                                "slot_id": (
                                    f"human-review::{family_id}::"
                                    f"{case_ordinal:02d}"
                                ),
                                "lane": lane,
                                "dimension": dimension,
                                "family_id": family_id,
                                "family_index": family_index,
                                "provenance_class": provenance_class,
                                "case_ordinal": case_ordinal,
                                "reviewer_class": "HUMAN_REVIEW",
                                "assigned_reviewer_id": None,
                                "case_id": None,
                                "subject_digest": None,
                                "grader_digest": None,
                            }
                        )

    mixed_dimension_review_classification = [
        {
            "dimension": dimension,
            "grading_class": "MIXED",
            "behavioral_case_slots": 500,
            "adversarial_case_slots": 100,
            "total_case_slots": 600,
            "rule": (
                "After case/grader materialization, every case whose grader kind "
                "is semantic_review requires the same real-human receipt contract."
            ),
        }
        for dimension in MIXED_DIMENSIONS
    ]

    return {
        "schema": "V10_FINAL_BANK_HUMAN_HANDOFF_WORK_ORDER_V1",
        "date": "2026-10-03",
        "status": "UNASSIGNED_EXTERNAL_HUMAN_WORK",
        "human_seed_slot_count": len(human_seed_slots),
        "mandatory_semantic_review_slot_count": len(mandatory_review_slots),
        "mixed_dimension_case_slots_pending_classification": (
            len(MIXED_DIMENSIONS) * 600
        ),
        "human_seed_slots": human_seed_slots,
        "mandatory_semantic_review_slots": mandatory_review_slots,
        "mixed_dimension_review_classification": (
            mixed_dimension_review_classification
        ),
        "rules": {
            "human_seed_requires_genuine_human_authorship": True,
            "model_generated_seed_cannot_satisfy_human_slot": True,
            "semantic_review_requires_real_human": True,
            "llm_review_cannot_satisfy_human_review_slot": True,
            "generation_actor_self_review_forbidden": True,
            "reviewer_must_be_bound_before_receipt_counts": True,
            "final_plaintext_training_lane_access": False,
        },
        "boundaries": {
            "contains_final_plaintext": False,
            "assigns_human_identity": False,
            "claims_human_work_completed": False,
            "final_bank_admitted": False,
            "training_authorized": False,
            "weights_changed": False,
        },
        "claim_ceiling": (
            "UNASSIGNED HUMAN WORK METADATA ONLY / "
            "NO HUMAN AUTHORSHIP OR REVIEW CLAIMED / "
            "NO FINAL PLAINTEXT / NO FINAL BANK / NO TRAINING"
        ),
    }


def _sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: object) -> bool:
    if not _nonempty(value):
        return False
    text = str(value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_human_review_receipt(
    receipt: dict,
    *,
    expected_case_id: str,
    generation_actor_id: str,
    expected_subject_digest: str,
    expected_grader_digest: str,
    bound_reviewer_ids: set[str],
) -> dict:
    reasons: list[str] = []

    if not isinstance(receipt, dict):
        return {
            "schema": "V10_FINAL_BANK_HUMAN_REVIEW_RECEIPT_CHECK_V1",
            "status": "HOLD",
            "reasons": ["receipt_not_object"],
        }
    if (
        not isinstance(bound_reviewer_ids, set)
        or any(not _nonempty(value) for value in bound_reviewer_ids)
    ):
        raise ValueError("bound_reviewer_ids must be nonempty string set")
    if not _sha256(expected_subject_digest):
        raise ValueError("expected_subject_digest invalid")
    if not _sha256(expected_grader_digest):
        raise ValueError("expected_grader_digest invalid")

    if receipt.get("case_id") != expected_case_id:
        reasons.append("case_id_mismatch")
    if receipt.get("verdict") not in {"ADMIT", "REJECT"}:
        reasons.append("verdict_invalid")
    reviewer_id = receipt.get("reviewer_id")
    if not _nonempty(reviewer_id):
        reasons.append("reviewer_id_missing")
    elif reviewer_id not in bound_reviewer_ids:
        reasons.append("reviewer_not_bound")
    if receipt.get("reviewer_class") != "HUMAN_REVIEW":
        reasons.append("reviewer_class_not_human")
    if not _nonempty(receipt.get("provider")):
        reasons.append("provider_missing")
    if not _nonempty(receipt.get("runtime")):
        reasons.append("runtime_missing")
    if receipt.get("independent_from_generation") is not True:
        reasons.append("independent_from_generation_not_true")
    if reviewer_id == generation_actor_id:
        reasons.append("reviewer_not_independent_from_generation")

    if receipt.get("subject_digest") != expected_subject_digest:
        reasons.append("subject_digest_mismatch")
    if receipt.get("grader_digest") != expected_grader_digest:
        reasons.append("grader_digest_mismatch")
    if not _sha256(receipt.get("artifact_digest")):
        reasons.append("artifact_digest_invalid")
    if not _sha256(receipt.get("review_bundle_sha256")):
        reasons.append("review_bundle_sha256_invalid")
    if not _timestamp(receipt.get("review_timestamp")):
        reasons.append("review_timestamp_invalid")

    review_answers = receipt.get("review_answers")
    if not isinstance(review_answers, dict):
        reasons.append("review_answers_missing")
    else:
        if set(review_answers) != set(REVIEW_QUESTIONS):
            reasons.append("review_answer_key_set_mismatch")
        for question in REVIEW_QUESTIONS:
            if question in review_answers and not isinstance(
                review_answers[question], bool
            ):
                reasons.append(f"review_answer_not_boolean:{question}")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_HUMAN_REVIEW_RECEIPT_CHECK_V1",
        "status": (
            "HUMAN_REVIEW_RECEIPT_PASS" if not reasons else "HOLD"
        ),
        "case_id": receipt.get("case_id"),
        "reviewer_id": reviewer_id,
        "verdict": receipt.get("verdict"),
        "reasons": reasons,
        "claim_ceiling": (
            "SINGLE HUMAN REVIEW RECEIPT VALIDATION ONLY / "
            "DOES NOT PROVE FULL REVIEW COVERAGE / "
            "DOES NOT ADMIT OR SEAL FINAL BANK"
        ),
    }
