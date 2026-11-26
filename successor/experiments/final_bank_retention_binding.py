from __future__ import annotations

import re


EXPECTED_ALLOCATION = {
    "knowledge_factuality": 300,
    "reasoning_math": 300,
    "coding": 250,
    "instruction_following": 250,
    "extraction_structured": 200,
    "truthfulness_factual_calibration": 200,
}
REQUIRED_ARTIFACTS = {
    "candidate_sha256",
    "candidate_manifest_sha256",
    "candidate_verify_sha256",
    "admission_result_sha256",
    "result_manifest_sha256",
    "semantic_receipt_sha256",
    "witness_adjudication_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def validate_retention_lane_binding(binding: dict) -> dict:
    reasons: list[str] = []
    if not isinstance(binding, dict):
        return {
            "schema": "V10_FINAL_BANK_RETENTION_LANE_BINDING_CHECK_V1",
            "status": "HOLD",
            "reasons": ["binding_not_object"],
        }

    if binding.get("schema") != "V10_FINAL_BANK_RETENTION_LANE_BINDING_V1":
        reasons.append(f"schema:{binding.get('schema')}")
    if binding.get("status") != "RETENTION_LANE_ADMITTED_FOR_FINAL_BANK":
        reasons.append(f"binding_status:{binding.get('status')}")
    if binding.get("lane") != "retention":
        reasons.append(f"lane:{binding.get('lane')}")
    if binding.get("lane_count") != 1500:
        reasons.append(f"lane_count:{binding.get('lane_count')}!=1500")

    allocation = binding.get("allocation")
    if not isinstance(allocation, dict):
        reasons.append("allocation_missing")
        allocation = {}
    if set(allocation) != set(EXPECTED_ALLOCATION):
        reasons.append("allocation_key_set_mismatch")
    for key, expected in EXPECTED_ALLOCATION.items():
        actual = allocation.get(key)
        if actual != expected:
            reasons.append(f"allocation:{key}:{actual}!={expected}")

    source = binding.get("source")
    if not isinstance(source, dict):
        reasons.append("source_missing")
        source = {}
    if source.get("repository") != "thebrazenbeard/vera_model_training":
        reasons.append(f"source_repository:{source.get('repository')}")
    branch = source.get("branch")
    if not isinstance(branch, str) or not branch.strip():
        reasons.append("source_branch_missing")
    head = source.get("head_sha")
    if not isinstance(head, str) or not _GIT_SHA.fullmatch(head):
        reasons.append("source_head_sha_invalid")
    if source.get("admission_status") != "RETENTION_ADMITTED_ARCH_V7":
        reasons.append(
            f"source_admission_status:{source.get('admission_status')}"
        )

    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict):
        reasons.append("artifacts_missing")
        artifacts = {}
    if set(artifacts) != REQUIRED_ARTIFACTS:
        reasons.append("artifact_key_set_mismatch")
    for field in sorted(REQUIRED_ARTIFACTS):
        if not _sha256(artifacts.get(field)):
            reasons.append(f"artifact_hash_invalid:{field}")

    boundaries = binding.get("boundaries")
    if not isinstance(boundaries, dict):
        reasons.append("boundaries_missing")
        boundaries = {}
    if boundaries.get("internal_evidence_backed_admission") is not True:
        reasons.append("boundary_internal_admission_not_true")
    if boundaries.get("independent_external_review") is not False:
        reasons.append("boundary_independent_external_review_not_false")
    if boundaries.get("methodology_history_independent") is not False:
        reasons.append("boundary_methodology_history_independent_not_false")
    if boundaries.get("full_final_bank_admitted") is not False:
        reasons.append("boundary_full_final_bank_admitted_not_false")
    if boundaries.get("training_authorized") is not False:
        reasons.append("boundary_training_authorized_not_false")
    if boundaries.get("weights_changed") is not False:
        reasons.append("boundary_weights_changed_not_false")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_RETENTION_LANE_BINDING_CHECK_V1",
        "status": "RETENTION_LANE_BINDING_PASS" if not reasons else "HOLD",
        "lane_count": binding.get("lane_count"),
        "source_head_sha": source.get("head_sha"),
        "reasons": reasons,
        "claim_ceiling": (
            "RETENTION_LANE_BINDING_ONLY / "
            "DOES_NOT ADMIT BEHAVIORAL_OR_ADVERSARIAL LANES / "
            "DOES_NOT SEAL FINAL BANK / DOES NOT AUTHORIZE TRAINING"
        ),
    }
