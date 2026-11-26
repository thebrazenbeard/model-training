from __future__ import annotations

from collections import Counter

from successor.experiments.build_v10_retention_audit_packet import (
    select_family_audit_sample,
)
from successor.experiments.retention_admission_v3 import (
    EXPECTED_REVIEWER_BLOB_SHA256,
    EXPECTED_REVIEWER_MODEL,
    EXPECTED_REVIEWER_PROVIDER,
    _packet_case_fields_match,
    _valid_sha256,
    candidate_rows_sha256,
)
from successor.experiments.v10_qwen35_bank import preflight_retention_rows


_REQUIRED_BOOLEAN_FIELDS = (
    "prompt_well_posed",
    "grader_matches_prompt",
    "source_evidence_sufficient",
)


def _derived_verdict(review: dict) -> str:
    return (
        "ADMIT"
        if all(review.get(field) is True for field in _REQUIRED_BOOLEAN_FIELDS)
        else "REJECT"
    )


def validate_retention_admission_v4(
    *,
    rows: list[dict],
    exclusion_hashes: set[str],
    candidate_file_sha256: str,
    packet_rows: list[dict],
    packet_file_sha256: str,
    packet_manifest: dict,
    audit_rows: list[dict],
    review_output_sha256: str,
    audit_receipt: dict,
    semantic_binding: dict,
    expected_protocol_sha256: str,
    expected_allocation: dict[str, int] | None = None,
    expected_per_family: int = 5,
) -> dict:
    reasons: list[str] = []

    structure = preflight_retention_rows(
        rows,
        exclusion_hashes=exclusion_hashes,
        expected_allocation=expected_allocation,
    )
    if structure["status"] != "STRUCTURE_READY":
        reasons.extend(
            f"structure:{reason}"
            for reason in structure["reasons"]
        )

    candidate_sha = candidate_rows_sha256(rows)
    if candidate_file_sha256 != candidate_sha:
        reasons.append(
            "candidate_file_sha256_mismatch:"
            f"{candidate_file_sha256}!={candidate_sha}"
        )

    if packet_manifest.get("candidate_sha256") != candidate_sha:
        reasons.append("packet_candidate_sha256_mismatch")
    if packet_manifest.get("packet_sha256") != packet_file_sha256:
        reasons.append("packet_file_sha256_mismatch")
    if packet_manifest.get("per_family") != expected_per_family:
        reasons.append(
            f"packet_per_family:{packet_manifest.get('per_family')}!="
            f"{expected_per_family}"
        )

    candidate_by_id = {row["case_id"]: row for row in rows}
    candidate_families = {row["family_id"] for row in rows}
    expected_sample = select_family_audit_sample(
        rows,
        per_family=expected_per_family,
        candidate_sha256=candidate_sha,
    )
    expected_ids = {row["case_id"] for row in expected_sample}

    packet_by_id: dict[str, dict] = {}
    for packet in packet_rows:
        case_id = packet.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            reasons.append("packet_case_id_invalid")
            continue
        if case_id in packet_by_id:
            reasons.append(f"packet_duplicate_case_id:{case_id}")
        packet_by_id[case_id] = packet

    if set(packet_by_id) != expected_ids:
        reasons.append(
            "packet_case_set_mismatch:"
            f"{len(packet_by_id)}!={len(expected_ids)}"
        )
    if packet_manifest.get("sample_rows") != len(packet_rows):
        reasons.append("packet_sample_rows_mismatch")
    if packet_manifest.get("family_count") != len(candidate_families):
        reasons.append("packet_family_count_mismatch")

    for case_id, packet in packet_by_id.items():
        candidate = candidate_by_id.get(case_id)
        if candidate is None:
            reasons.append(f"packet_unknown_case:{case_id}")
            continue
        if not _packet_case_fields_match(candidate, packet):
            reasons.append(f"packet_subject_mismatch:{case_id}")

    if audit_receipt.get("candidate_sha256") != candidate_sha:
        reasons.append("audit_candidate_sha256_mismatch")
    if audit_receipt.get("packet_sha256") != packet_file_sha256:
        reasons.append("audit_packet_sha256_mismatch")
    if audit_receipt.get("protocol_sha256") != expected_protocol_sha256:
        reasons.append("audit_protocol_sha256_mismatch")
    if audit_receipt.get("review_output_sha256") != review_output_sha256:
        reasons.append("audit_output_sha256_mismatch")
    if audit_receipt.get("sample_rows") != len(packet_rows):
        reasons.append("audit_sample_rows_mismatch")
    if audit_receipt.get("family_count") != len(candidate_families):
        reasons.append("audit_family_count_mismatch")
    if audit_receipt.get("cases_per_family") != expected_per_family:
        reasons.append("audit_cases_per_family_mismatch")

    reviewer = audit_receipt.get("reviewer")
    if not isinstance(reviewer, dict):
        reasons.append("audit_reviewer_missing")
    else:
        if reviewer.get("provider") != EXPECTED_REVIEWER_PROVIDER:
            reasons.append("audit_reviewer_provider_mismatch")
        if reviewer.get("model") != EXPECTED_REVIEWER_MODEL:
            reasons.append("audit_reviewer_model_mismatch")
        if (
            reviewer.get("model_blob_sha256")
            != EXPECTED_REVIEWER_BLOB_SHA256
        ):
            reasons.append("audit_reviewer_blob_mismatch")
        if reviewer.get("runtime") != "ollama version is 0.34.2":
            reasons.append("audit_reviewer_runtime_mismatch")
        if reviewer.get("temperature") != 0:
            reasons.append("audit_temperature_mismatch")
        if reviewer.get("seed") != 20261001:
            reasons.append("audit_seed_mismatch")

    summary = audit_receipt.get("summary")
    if not isinstance(summary, dict):
        reasons.append("audit_summary_missing")
        summary = {}
    if summary.get("status") != "PASS":
        reasons.append(f"audit_summary_status:{summary.get('status')}")
    if summary.get("reviewed") != len(expected_ids):
        reasons.append("audit_summary_reviewed_mismatch")
    if summary.get("families_reviewed") != len(candidate_families):
        reasons.append("audit_summary_families_reviewed_mismatch")
    if summary.get("reject") != 0:
        reasons.append("audit_summary_reject_count_nonzero")
    if summary.get("admit") != len(expected_ids):
        reasons.append("audit_summary_admit_count_mismatch")
    if summary.get("reasons") not in ([], None):
        reasons.append("audit_summary_reasons_nonempty")

    audit_by_id: dict[str, dict] = {}
    for audit in audit_rows:
        case_id = audit.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            reasons.append("audit_case_id_invalid")
            continue
        if case_id in audit_by_id:
            reasons.append(f"audit_duplicate_case_id:{case_id}")
        audit_by_id[case_id] = audit

    if set(audit_by_id) != expected_ids:
        reasons.append(
            "audit_case_set_mismatch:"
            f"{len(audit_by_id)}!={len(expected_ids)}"
        )

    reviewed_families = Counter()
    for case_id, audit in audit_by_id.items():
        candidate = candidate_by_id.get(case_id)
        if candidate is None:
            reasons.append(f"audit_unknown_case:{case_id}")
            continue
        reviewed_families[candidate["family_id"]] += 1

        if audit.get("protocol_sha256") != expected_protocol_sha256:
            reasons.append(
                f"audit_row_protocol_sha256_mismatch:{case_id}"
            )
        if not _valid_sha256(
            audit.get("family_review_prompt_sha256")
        ):
            reasons.append(
                f"audit_prompt_sha256_invalid:{case_id}"
            )

        review = audit.get("review")
        if not isinstance(review, dict):
            reasons.append(f"audit_review_missing:{case_id}")
            continue
        if "verdict" in review:
            reasons.append(
                f"audit_review_verdict_field_present:{case_id}"
            )

        for field in _REQUIRED_BOOLEAN_FIELDS:
            if not isinstance(review.get(field), bool):
                reasons.append(
                    f"audit_review_boolean_invalid:{case_id}:{field}"
                )
        if not isinstance(review.get("issue_code"), str) or not review.get(
            "issue_code"
        ):
            reasons.append(f"audit_review_issue_code_invalid:{case_id}")
        if not isinstance(review.get("reason"), str) or not review.get(
            "reason", ""
        ).strip():
            reasons.append(f"audit_review_reason_invalid:{case_id}")
        if review.get("confidence") not in {"high", "medium", "low"}:
            reasons.append(f"audit_review_confidence_invalid:{case_id}")

        derived = _derived_verdict(review)
        if audit.get("verdict") != derived:
            reasons.append(
                f"audit_derived_verdict_mismatch:{case_id}"
            )
        if derived != "ADMIT":
            reasons.append(f"audit_non_admit:{case_id}")

    for family_id in sorted(candidate_families):
        count = reviewed_families[family_id]
        if count != expected_per_family:
            reasons.append(
                f"audit_family_count:{family_id}:{count}!="
                f"{expected_per_family}"
            )

    semantic_status = semantic_binding.get("status")
    if semantic_status != "PASS":
        reasons.append(
            f"semantic_binding_status:{semantic_status}"
        )
    if semantic_binding.get("target_sha256") != candidate_sha:
        reasons.append("semantic_binding_target_mismatch")
    semantic_reasons = semantic_binding.get("reasons")
    if semantic_reasons not in ([], None):
        reasons.append("semantic_binding_reasons_nonempty")

    if not _valid_sha256(expected_protocol_sha256):
        reasons.append("expected_protocol_sha256_invalid")
    if not _valid_sha256(packet_file_sha256):
        reasons.append("packet_file_sha256_invalid")
    if not _valid_sha256(review_output_sha256):
        reasons.append("review_output_sha256_invalid")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_QWEN35_RETENTION_ADMISSION_CHECK_V4",
        "status": (
            "RETENTION_ADMITTED_V4"
            if not reasons
            else "HOLD"
        ),
        "candidate_sha256": candidate_sha,
        "protocol_sha256": expected_protocol_sha256,
        "case_count": len(rows),
        "family_count": len(candidate_families),
        "reviewed_case_count": len(audit_by_id),
        "reviewed_family_count": len(
            [
                family
                for family in candidate_families
                if reviewed_families[family]
            ]
        ),
        "expected_per_family": expected_per_family,
        "structure_status": structure["status"],
        "reasons": reasons,
        "claim_ceiling": (
            "OBJECTIVE_RETENTION_LANE_ADMISSION_V4_ONLY / "
            "V3_PREDECESSOR_REMAINS_HOLD / "
            "NOT_BEHAVIORAL_OR_ADVERSARIAL_FINAL_BANK_ADMISSION / "
            "DETERMINISTIC_TRUTH_PLUS_FAMILY_LEVEL_EXTERNAL_MODEL_AUDIT"
        ),
    }
