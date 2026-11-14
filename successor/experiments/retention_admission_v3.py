from __future__ import annotations

from collections import Counter
import hashlib
import json

from successor.experiments.build_v10_retention_audit_packet import (
    select_family_audit_sample,
)
from successor.experiments.v10_qwen35_bank import preflight_retention_rows


EXPECTED_REVIEWER_PROVIDER = "OLLAMA_LOCAL"
EXPECTED_REVIEWER_MODEL = "ministral-3:14b"
EXPECTED_REVIEWER_BLOB_SHA256 = (
    "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _rows_sha256(rows: list[dict]) -> str:
    payload = (
        "\n".join(_canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_rows_sha256(rows: list[dict]) -> str:
    return _rows_sha256(rows)


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _packet_case_fields_match(candidate: dict, packet: dict) -> bool:
    fields = (
        "case_id",
        "category",
        "family_id",
        "prompt",
        "source_id",
        "source_revision",
        "source_terms",
        "source_hash",
        "generation_method",
        "grader_contract",
    )
    return all(packet.get(field) == candidate.get(field) for field in fields)


def validate_retention_admission_v3(
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
            f"structure:{reason}" for reason in structure["reasons"]
        )

    candidate_sha = candidate_rows_sha256(rows)
    if candidate_file_sha256 != candidate_sha:
        reasons.append(
            f"candidate_file_sha256_mismatch:{candidate_file_sha256}!={candidate_sha}"
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
            f"packet_case_set_mismatch:{len(packet_by_id)}!={len(expected_ids)}"
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
    if audit_receipt.get("review_output_sha256") != review_output_sha256:
        reasons.append("audit_output_sha256_mismatch")

    reviewer = audit_receipt.get("reviewer")
    if not isinstance(reviewer, dict):
        reasons.append("audit_reviewer_missing")
    else:
        if reviewer.get("provider") != EXPECTED_REVIEWER_PROVIDER:
            reasons.append("audit_reviewer_provider_mismatch")
        if reviewer.get("model") != EXPECTED_REVIEWER_MODEL:
            reasons.append("audit_reviewer_model_mismatch")
        if reviewer.get("model_blob_sha256") != EXPECTED_REVIEWER_BLOB_SHA256:
            reasons.append("audit_reviewer_blob_mismatch")
        if reviewer.get("temperature") != 0:
            reasons.append("audit_temperature_mismatch")
        if reviewer.get("seed") != 20261001:
            reasons.append("audit_seed_mismatch")

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
            f"audit_case_set_mismatch:{len(audit_by_id)}!={len(expected_ids)}"
        )

    reviewed_families = Counter()
    for case_id, audit in audit_by_id.items():
        candidate = candidate_by_id.get(case_id)
        if candidate is None:
            reasons.append(f"audit_unknown_case:{case_id}")
            continue
        reviewed_families[candidate["family_id"]] += 1
        review = audit.get("review")
        if audit.get("verdict") != "ADMIT":
            reasons.append(f"audit_non_admit:{case_id}")
            continue
        if not isinstance(review, dict) or review.get("verdict") != "ADMIT":
            reasons.append(f"audit_review_non_admit:{case_id}")
            continue
        for field in (
            "prompt_well_posed",
            "grader_matches_prompt",
            "source_evidence_sufficient",
        ):
            if review.get(field) is not True:
                reasons.append(f"audit_review_check_failed:{case_id}:{field}")

    for family_id in sorted(candidate_families):
        count = reviewed_families[family_id]
        if count != expected_per_family:
            reasons.append(
                f"audit_family_count:{family_id}:{count}!={expected_per_family}"
            )

    semantic_status = semantic_binding.get("status")
    if semantic_status != "PASS":
        reasons.append(f"semantic_binding_status:{semantic_status}")
    if semantic_binding.get("target_sha256") != candidate_sha:
        reasons.append("semantic_binding_target_mismatch")
    semantic_reasons = semantic_binding.get("reasons")
    if semantic_reasons not in ([], None):
        reasons.append("semantic_binding_reasons_nonempty")

    if not _valid_sha256(packet_file_sha256):
        reasons.append("packet_file_sha256_invalid")
    if not _valid_sha256(review_output_sha256):
        reasons.append("review_output_sha256_invalid")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_QWEN35_RETENTION_ADMISSION_CHECK_V3",
        "status": "RETENTION_ADMITTED_V3" if not reasons else "HOLD",
        "candidate_sha256": candidate_sha,
        "case_count": len(rows),
        "family_count": len(candidate_families),
        "reviewed_case_count": len(audit_by_id),
        "reviewed_family_count": len(
            [family for family in candidate_families if reviewed_families[family]]
        ),
        "expected_per_family": expected_per_family,
        "structure_status": structure["status"],
        "reasons": reasons,
        "claim_ceiling": (
            "OBJECTIVE_RETENTION_LANE_ADMISSION_ONLY / "
            "NOT_BEHAVIORAL_OR_ADVERSARIAL_FINAL_BANK_ADMISSION / "
            "DETERMINISTIC_TRUTH_PLUS_FAMILY_LEVEL_EXTERNAL_MODEL_AUDIT"
        ),
    }
