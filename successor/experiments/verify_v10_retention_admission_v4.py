from __future__ import annotations

import argparse
import json
from pathlib import Path

from successor.experiments.retention_admission_v4 import (
    validate_retention_admission_v4,
)
from successor.experiments.verify_v10_retention_admission_v3 import (
    _exclusion_hashes,
    _mismatch,
    _read_json,
    _read_jsonl,
    _sha256_file,
)
from successor.experiments.v10_qwen35_bank import RETENTION_ALLOCATION


def verify_retention_admission_files_v4(
    *,
    candidate_path: Path,
    candidate_manifest_path: Path,
    exclusion_path: Path,
    packet_path: Path,
    packet_manifest_path: Path,
    protocol_path: Path,
    runner_path: Path,
    audit_output_path: Path,
    audit_receipt_path: Path,
    semantic_path: Path,
    admission_policy_path: Path,
    evidence_binding_path: Path,
) -> dict:
    binding_reasons: list[str] = []

    binding = _read_json(evidence_binding_path)
    if (
        binding.get("schema")
        != "V10_QWEN35_RETENTION_ADMISSION_EVIDENCE_BINDING_V2"
    ):
        binding_reasons.append("evidence_binding_schema_mismatch")
    if binding.get("status") != "FROZEN_BEFORE_V4_AUDIT_RESULT":
        binding_reasons.append("evidence_binding_status_mismatch")

    policy = _read_json(admission_policy_path)
    candidate_manifest = _read_json(candidate_manifest_path)
    packet_manifest = _read_json(packet_manifest_path)
    protocol = _read_json(protocol_path)
    audit_receipt = _read_json(audit_receipt_path)
    semantic = _read_json(semantic_path)

    candidate_rows = _read_jsonl(candidate_path)
    packet_rows = _read_jsonl(packet_path)
    audit_rows = _read_jsonl(audit_output_path)
    exclusion_hashes = _exclusion_hashes(exclusion_path)

    actual = {
        "policy": _sha256_file(admission_policy_path),
        "candidate": _sha256_file(candidate_path),
        "exclusion": _sha256_file(exclusion_path),
        "packet": _sha256_file(packet_path),
        "packet_manifest": _sha256_file(packet_manifest_path),
        "protocol": _sha256_file(protocol_path),
        "runner": _sha256_file(runner_path),
        "audit_output": _sha256_file(audit_output_path),
        "semantic": _sha256_file(semantic_path),
    }

    policy_binding = binding.get("admission_policy", {})
    candidate_binding = binding.get("candidate", {})
    packet_binding = binding.get("family_audit_packet", {})
    audit_binding = binding.get("family_audit_v4", {})
    semantic_binding = binding.get("semantic_screen", {})
    exclusion_binding = binding.get("exclusion_registry", {})
    receipt_requirements = binding.get(
        "final_v4_receipt_requirements",
        {},
    )

    for label, actual_value, expected_value in (
        (
            "admission_policy_file_sha256_mismatch",
            actual["policy"],
            policy_binding.get("file_sha256"),
        ),
        (
            "candidate_file_sha256_mismatch",
            actual["candidate"],
            candidate_binding.get("file_sha256"),
        ),
        (
            "exclusion_file_sha256_mismatch",
            actual["exclusion"],
            exclusion_binding.get("file_sha256"),
        ),
        (
            "packet_file_sha256_mismatch",
            actual["packet"],
            packet_binding.get("file_sha256"),
        ),
        (
            "packet_manifest_file_sha256_mismatch",
            actual["packet_manifest"],
            packet_binding.get("manifest_file_sha256"),
        ),
        (
            "protocol_file_sha256_mismatch",
            actual["protocol"],
            audit_binding.get("protocol_file_sha256"),
        ),
        (
            "runner_file_sha256_mismatch",
            actual["runner"],
            audit_binding.get("runner_file_sha256"),
        ),
        (
            "semantic_file_sha256_mismatch",
            actual["semantic"],
            semantic_binding.get("file_sha256"),
        ),
    ):
        _mismatch(
            binding_reasons,
            label=label,
            actual=actual_value,
            expected=expected_value,
        )

    if policy.get("schema") != "V10_QWEN35_RETENTION_ADMISSION_V4":
        binding_reasons.append("admission_policy_schema_mismatch")
    if (
        policy.get("status")
        != "PREREGISTERED_BEFORE_V4_AUDIT_RESULT"
    ):
        binding_reasons.append("admission_policy_status_mismatch")

    policy_subject = policy.get("subject", {})
    _mismatch(
        binding_reasons,
        label="policy_candidate_sha256_mismatch",
        actual=policy_subject.get("expected_candidate_sha256"),
        expected=candidate_binding.get("file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="policy_expected_rows_mismatch",
        actual=policy_subject.get("expected_rows"),
        expected=candidate_binding.get("expected_rows"),
    )

    policy_audit = policy.get("family_audit_v4", {})
    _mismatch(
        binding_reasons,
        label="policy_protocol_sha256_mismatch",
        actual=policy_audit.get("protocol_sha256"),
        expected=audit_binding.get("protocol_file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="policy_runner_sha256_mismatch",
        actual=policy_audit.get("runner_sha256"),
        expected=audit_binding.get("runner_file_sha256"),
    )
    if policy_audit.get("v3_rows_reusable") is not False:
        binding_reasons.append("policy_v3_reuse_not_false")

    _mismatch(
        binding_reasons,
        label="candidate_manifest_data_sha256_mismatch",
        actual=candidate_manifest.get("data_sha256"),
        expected=actual["candidate"],
    )
    _mismatch(
        binding_reasons,
        label="candidate_manifest_case_count_mismatch",
        actual=candidate_manifest.get("case_count"),
        expected=len(candidate_rows),
    )
    _mismatch(
        binding_reasons,
        label="candidate_expected_rows_mismatch",
        actual=len(candidate_rows),
        expected=candidate_binding.get("expected_rows"),
    )
    if candidate_manifest.get("category_counts") != RETENTION_ALLOCATION:
        binding_reasons.append("candidate_allocation_mismatch")

    for label, actual_value, expected_value in (
        (
            "packet_manifest_candidate_sha256_mismatch",
            packet_manifest.get("candidate_sha256"),
            actual["candidate"],
        ),
        (
            "packet_manifest_packet_sha256_mismatch",
            packet_manifest.get("packet_sha256"),
            actual["packet"],
        ),
        (
            "packet_manifest_rows_mismatch",
            packet_manifest.get("sample_rows"),
            packet_binding.get("rows"),
        ),
        (
            "packet_manifest_families_mismatch",
            packet_manifest.get("family_count"),
            packet_binding.get("families"),
        ),
        (
            "packet_manifest_cases_per_family_mismatch",
            packet_manifest.get("per_family"),
            packet_binding.get("cases_per_family"),
        ),
    ):
        _mismatch(
            binding_reasons,
            label=label,
            actual=actual_value,
            expected=expected_value,
        )

    _mismatch(
        binding_reasons,
        label="protocol_candidate_sha256_mismatch",
        actual=protocol.get("candidate_sha256"),
        expected=actual["candidate"],
    )
    _mismatch(
        binding_reasons,
        label="protocol_packet_sha256_mismatch",
        actual=protocol.get("packet_sha256"),
        expected=actual["packet"],
    )
    _mismatch(
        binding_reasons,
        label="protocol_sample_rows_mismatch",
        actual=protocol.get("sample_rows"),
        expected=packet_binding.get("rows"),
    )
    _mismatch(
        binding_reasons,
        label="protocol_family_count_mismatch",
        actual=protocol.get("family_count"),
        expected=packet_binding.get("families"),
    )
    _mismatch(
        binding_reasons,
        label="protocol_cases_per_family_mismatch",
        actual=protocol.get("cases_per_family"),
        expected=packet_binding.get("cases_per_family"),
    )

    required_reviewer = audit_binding.get("required_reviewer", {})
    receipt_reviewer = audit_receipt.get("reviewer", {})
    for field in (
        "provider",
        "model",
        "model_blob_sha256",
        "runtime",
        "temperature",
        "seed",
    ):
        _mismatch(
            binding_reasons,
            label=f"audit_reviewer_{field}_mismatch",
            actual=receipt_reviewer.get(field),
            expected=required_reviewer.get(field),
        )

    for field, receipt_field in (
        ("candidate_sha256", "candidate_sha256"),
        ("packet_sha256", "packet_sha256"),
        ("protocol_sha256", "protocol_sha256"),
        ("sample_rows", "sample_rows"),
        ("family_count", "family_count"),
        ("cases_per_family", "cases_per_family"),
    ):
        _mismatch(
            binding_reasons,
            label=f"audit_receipt_{field}_mismatch",
            actual=audit_receipt.get(receipt_field),
            expected=receipt_requirements.get(field),
        )
    _mismatch(
        binding_reasons,
        label="audit_receipt_review_output_sha256_mismatch",
        actual=audit_receipt.get("review_output_sha256"),
        expected=actual["audit_output"],
    )

    summary = audit_receipt.get("summary", {})
    _mismatch(
        binding_reasons,
        label="audit_receipt_summary_status_mismatch",
        actual=summary.get("status"),
        expected=receipt_requirements.get("summary_status_required"),
    )
    _mismatch(
        binding_reasons,
        label="audit_receipt_reviewed_rows_mismatch",
        actual=summary.get("reviewed"),
        expected=receipt_requirements.get("reviewed_rows_required"),
    )
    _mismatch(
        binding_reasons,
        label="audit_receipt_reviewed_families_mismatch",
        actual=summary.get("families_reviewed"),
        expected=receipt_requirements.get("reviewed_families_required"),
    )
    _mismatch(
        binding_reasons,
        label="audit_receipt_reject_count_mismatch",
        actual=summary.get("reject"),
        expected=receipt_requirements.get("reject_count_required"),
    )

    for label, actual_value, expected_value in (
        (
            "semantic_status_mismatch",
            semantic.get("status"),
            semantic_binding.get("required_status"),
        ),
        (
            "semantic_target_sha256_mismatch",
            semantic.get("target_sha256"),
            semantic_binding.get("target_sha256"),
        ),
        (
            "semantic_threshold_mismatch",
            semantic.get("threshold"),
            semantic_binding.get("threshold"),
        ),
        (
            "semantic_model_archive_sha256_mismatch",
            semantic.get("model_archive_sha256"),
            semantic_binding.get("model_archive_sha256"),
        ),
        (
            "semantic_registry_sha256_mismatch",
            semantic.get("registry", {}).get("registry_sha256"),
            semantic_binding.get("exclusion_registry_sha256"),
        ),
    ):
        _mismatch(
            binding_reasons,
            label=label,
            actual=actual_value,
            expected=expected_value,
        )

    binding_reasons = sorted(set(binding_reasons))
    if binding_reasons:
        return {
            "schema": "V10_QWEN35_RETENTION_ADMISSION_FILE_CHECK_V2",
            "status": "HOLD",
            "binding_reasons": binding_reasons,
            "core_reasons": [],
            "effect": "READ_ONLY_ADMISSION_VERIFICATION",
        }

    core = validate_retention_admission_v4(
        rows=candidate_rows,
        exclusion_hashes=exclusion_hashes,
        candidate_file_sha256=actual["candidate"],
        packet_rows=packet_rows,
        packet_file_sha256=actual["packet"],
        packet_manifest=packet_manifest,
        audit_rows=audit_rows,
        review_output_sha256=actual["audit_output"],
        audit_receipt=audit_receipt,
        semantic_binding=semantic,
        expected_protocol_sha256=actual["protocol"],
        expected_allocation=RETENTION_ALLOCATION,
        expected_per_family=int(
            packet_binding.get("cases_per_family")
        ),
    )
    return {
        **core,
        "schema": "V10_QWEN35_RETENTION_ADMISSION_FILE_CHECK_V2",
        "binding_reasons": [],
        "core_reasons": list(core.get("reasons", [])),
        "evidence_hashes": actual,
        "effect": "READ_ONLY_ADMISSION_VERIFICATION",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--admission-policy", type=Path, required=True)
    parser.add_argument("--evidence-binding", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        result = verify_retention_admission_files_v4(
            candidate_path=args.candidate,
            candidate_manifest_path=args.candidate_manifest,
            exclusion_path=args.exclusion,
            packet_path=args.packet,
            packet_manifest_path=args.packet_manifest,
            protocol_path=args.protocol,
            runner_path=args.runner,
            audit_output_path=args.audit_output,
            audit_receipt_path=args.audit_receipt,
            semantic_path=args.semantic,
            admission_policy_path=args.admission_policy,
            evidence_binding_path=args.evidence_binding,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "V10_QWEN35_RETENTION_ADMISSION_FILE_CHECK_V2",
            "status": "HOLD",
            "binding_reasons": [f"admission_input_error:{exc}"],
            "core_reasons": [],
            "effect": "READ_ONLY_ADMISSION_VERIFICATION",
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "RETENTION_ADMITTED_V4" else 2


if __name__ == "__main__":
    raise SystemExit(main())
