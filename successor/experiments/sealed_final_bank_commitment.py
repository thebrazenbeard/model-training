from __future__ import annotations

import re


DIMENSIONS = tuple(f"H{i:02d}" for i in range(1, 21))
LANE_COUNTS = {
    "behavioral": 10_000,
    "adversarial": 2_000,
    "retention": 1_500,
}
REQUIRED_HASHES = (
    "bank_sha256",
    "grader_bundle_sha256",
    "review_bundle_sha256",
    "source_manifest_sha256",
    "contamination_receipt_sha256",
    "sealed_archive_sha256",
)
FORBIDDEN_PLAINTEXT_KEYS = {
    "prompt",
    "prompts",
    "response",
    "responses",
    "rubric",
    "rubrics",
    "answer",
    "answers",
    "case",
    "cases",
    "row",
    "rows",
    "grader_text",
    "review_text",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_forbidden_plaintext_keys(value, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_PLAINTEXT_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(
                _find_forbidden_plaintext_keys(
                    child,
                    path=f"{path}.{key_text}",
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _find_forbidden_plaintext_keys(
                    child,
                    path=f"{path}[{index}]",
                )
            )
    return found


def validate_sealed_final_bank_commitment(commitment: dict) -> dict:
    reasons: list[str] = []

    if not isinstance(commitment, dict):
        return {
            "schema": "V10_SEALED_FINAL_BANK_COMMITMENT_CHECK_V1",
            "status": "HOLD",
            "reasons": ["commitment_not_object"],
        }

    if commitment.get("schema") != "V10_SEALED_FINAL_BANK_COMMITMENT_V1":
        reasons.append(f"schema:{commitment.get('schema')}")
    if commitment.get("status") != "SEALED_PRETRAINING_FINAL_BANK":
        reasons.append(f"commitment_status:{commitment.get('status')}")
    if not _nonempty(commitment.get("bank_id")):
        reasons.append("bank_id_missing")

    for path in _find_forbidden_plaintext_keys(commitment):
        reasons.append(f"plaintext_field_present:{path}")

    lane_counts = commitment.get("lane_counts")
    if not isinstance(lane_counts, dict):
        reasons.append("lane_counts_missing")
        lane_counts = {}
    for lane, expected in LANE_COUNTS.items():
        actual = lane_counts.get(lane)
        if actual != expected:
            reasons.append(f"lane_count:{lane}:{actual}!={expected}")

    behavioral = commitment.get("behavioral_dimension_counts")
    if not isinstance(behavioral, dict):
        reasons.append("behavioral_dimension_counts_missing")
        behavioral = {}
    adversarial = commitment.get("adversarial_dimension_counts")
    if not isinstance(adversarial, dict):
        reasons.append("adversarial_dimension_counts_missing")
        adversarial = {}
    families = commitment.get("behavioral_family_counts")
    if not isinstance(families, dict):
        reasons.append("behavioral_family_counts_missing")
        families = {}

    if set(behavioral) != set(DIMENSIONS):
        reasons.append("behavioral_dimension_key_set_mismatch")
    if set(adversarial) != set(DIMENSIONS):
        reasons.append("adversarial_dimension_key_set_mismatch")
    if set(families) != set(DIMENSIONS):
        reasons.append("behavioral_family_key_set_mismatch")

    for dimension in DIMENSIONS:
        behavioral_count = behavioral.get(dimension)
        if behavioral_count != 500:
            reasons.append(
                f"behavioral_dimension_count:{dimension}:"
                f"{behavioral_count}!=500"
            )
        adversarial_count = adversarial.get(dimension)
        if adversarial_count != 100:
            reasons.append(
                f"adversarial_dimension_count:{dimension}:"
                f"{adversarial_count}!=100"
            )
        family_count = families.get(dimension)
        if not isinstance(family_count, int) or family_count < 50:
            reasons.append(
                f"behavioral_family_count:{dimension}:"
                f"{family_count}<50"
            )

    artifacts = commitment.get("plaintext_artifacts")
    if not isinstance(artifacts, dict):
        reasons.append("plaintext_artifacts_missing")
        artifacts = {}
    for field in REQUIRED_HASHES:
        if not _sha256(artifacts.get(field)):
            reasons.append(f"invalid_hash:{field}")

    custody = commitment.get("custody")
    if not isinstance(custody, dict):
        reasons.append("custody_missing")
        custody = {}
    if custody.get("plaintext_exposed_to_training_lane") is not False:
        reasons.append("plaintext_exposed_to_training_lane")
    if custody.get("training_lane_receives_hashes_only") is not True:
        reasons.append("training_lane_hashes_only_not_asserted")

    custodian_ids = custody.get("custodian_ids")
    if (
        not isinstance(custodian_ids, list)
        or len(custodian_ids) < 3
        or any(not _nonempty(value) for value in custodian_ids)
        or len(set(custodian_ids)) != len(custodian_ids)
    ):
        reasons.append("custodian_ids_invalid")
        custodian_ids = []
    reviewer_id = custody.get("human_reviewer_id")
    if not _nonempty(reviewer_id):
        reasons.append("human_reviewer_id_missing")
    elif reviewer_id in custodian_ids:
        reasons.append("human_reviewer_not_independent_of_custodians")

    admission = commitment.get("admission")
    if not isinstance(admission, dict):
        reasons.append("admission_missing")
        admission = {}
    for field in (
        "exact_normalized_exclusion",
        "semantic_contamination",
        "independent_review",
    ):
        value = admission.get(field)
        if value != "PASS":
            reasons.append(f"admission:{field}:{value}")
    if admission.get("bank_frozen") is not True:
        reasons.append("admission:bank_frozen_not_true")
    if admission.get("post_freeze_case_mutation") is not False:
        reasons.append("admission:post_freeze_case_mutation_not_false")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_SEALED_FINAL_BANK_COMMITMENT_CHECK_V1",
        "status": "SEALED_BANK_GATE_PASS" if not reasons else "HOLD",
        "bank_id": commitment.get("bank_id"),
        "lane_counts": {
            lane: lane_counts.get(lane)
            for lane in LANE_COUNTS
        },
        "reasons": reasons,
        "plaintext_present": any(
            reason.startswith("plaintext_field_present:")
            for reason in reasons
        ),
        "claim_ceiling": (
            "NONPLAINTEXT_PRETRAINING_BANK_GATE_ONLY / "
            "DOES_NOT REVEAL OR SCORE FINAL CASES / "
            "DOES_NOT AUTHORIZE TRAINING"
        ),
    }
