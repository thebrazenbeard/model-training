from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.retention_admission_v3 import (
    validate_retention_admission_v3,
)
from successor.experiments.v10_qwen35_bank import RETENTION_ALLOCATION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"JSONL row {line_number} must be an object: {path}"
            )
        rows.append(value)
    return rows


def _exclusion_hashes(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _mismatch(
    reasons: list[str],
    *,
    label: str,
    actual,
    expected,
) -> None:
    if actual != expected:
        reasons.append(label)


def verify_retention_admission_files(
    *,
    candidate_path: Path,
    candidate_manifest_path: Path,
    exclusion_path: Path,
    packet_path: Path,
    packet_manifest_path: Path,
    protocol_path: Path,
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
        != "V10_QWEN35_RETENTION_ADMISSION_EVIDENCE_BINDING_V1"
    ):
        binding_reasons.append("evidence_binding_schema_mismatch")

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
        "audit_output": _sha256_file(audit_output_path),
        "semantic": _sha256_file(semantic_path),
    }

    policy_binding = binding.get("admission_policy", {})
    candidate_binding = binding.get("candidate", {})
    packet_binding = binding.get("family_audit_packet", {})
    protocol_binding = binding.get("family_audit_protocol", {})
    semantic_binding = binding.get("semantic_screen", {})
    exclusion_binding = binding.get("exclusion_registry", {})
    receipt_requirements = binding.get(
        "final_audit_receipt_requirements", {}
    )

    _mismatch(
        binding_reasons,
        label="admission_policy_file_sha256_mismatch",
        actual=actual["policy"],
        expected=policy_binding.get("file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="candidate_file_sha256_mismatch",
        actual=actual["candidate"],
        expected=candidate_binding.get("file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="exclusion_file_sha256_mismatch",
        actual=actual["exclusion"],
        expected=exclusion_binding.get("file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="packet_file_sha256_mismatch",
        actual=actual["packet"],
        expected=packet_binding.get("file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="packet_manifest_file_sha256_mismatch",
        actual=actual["packet_manifest"],
        expected=packet_binding.get("manifest_file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="protocol_file_sha256_mismatch",
        actual=actual["protocol"],
        expected=protocol_binding.get("file_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="semantic_file_sha256_mismatch",
        actual=actual["semantic"],
        expected=semantic_binding.get("file_sha256"),
    )

    if (
        policy.get("schema")
        != "V10_QWEN35_RETENTION_ADMISSION_V3"
    ):
        binding_reasons.append("admission_policy_schema_mismatch")
    if (
        policy.get("status")
        != "PREREGISTERED_BEFORE_FAMILY_AUDIT_RESULT"
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

    _mismatch(
        binding_reasons,
        label="packet_manifest_candidate_sha256_mismatch",
        actual=packet_manifest.get("candidate_sha256"),
        expected=actual["candidate"],
    )
    _mismatch(
        binding_reasons,
        label="packet_manifest_packet_sha256_mismatch",
        actual=packet_manifest.get("packet_sha256"),
        expected=actual["packet"],
    )
    _mismatch(
        binding_reasons,
        label="packet_manifest_rows_mismatch",
        actual=packet_manifest.get("sample_rows"),
        expected=packet_binding.get("rows"),
    )
    _mismatch(
        binding_reasons,
        label="packet_manifest_families_mismatch",
        actual=packet_manifest.get("family_count"),
        expected=packet_binding.get("families"),
    )
    _mismatch(
        binding_reasons,
        label="packet_manifest_cases_per_family_mismatch",
        actual=packet_manifest.get("per_family"),
        expected=packet_binding.get("cases_per_family"),
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

    required_reviewer = protocol_binding.get("required_reviewer", {})
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

    _mismatch(
        binding_reasons,
        label="audit_receipt_candidate_sha256_mismatch",
        actual=audit_receipt.get("candidate_sha256"),
        expected=receipt_requirements.get("candidate_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="audit_receipt_packet_sha256_mismatch",
        actual=audit_receipt.get("packet_sha256"),
        expected=receipt_requirements.get("packet_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="audit_receipt_protocol_sha256_mismatch",
        actual=audit_receipt.get("protocol_sha256"),
        expected=receipt_requirements.get("protocol_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="audit_receipt_review_output_sha256_mismatch",
        actual=audit_receipt.get("review_output_sha256"),
        expected=actual["audit_output"],
    )
    for field, receipt_field in (
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
        label="audit_receipt_summary_status_mismatch",
        actual=audit_receipt.get("summary", {}).get("status"),
        expected=receipt_requirements.get("summary_status_required"),
    )

    _mismatch(
        binding_reasons,
        label="semantic_status_mismatch",
        actual=semantic.get("status"),
        expected=semantic_binding.get("required_status"),
    )
    _mismatch(
        binding_reasons,
        label="semantic_target_sha256_mismatch",
        actual=semantic.get("target_sha256"),
        expected=semantic_binding.get("target_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="semantic_threshold_mismatch",
        actual=semantic.get("threshold"),
        expected=semantic_binding.get("threshold"),
    )
    _mismatch(
        binding_reasons,
        label="semantic_model_archive_sha256_mismatch",
        actual=semantic.get("model_archive_sha256"),
        expected=semantic_binding.get("model_archive_sha256"),
    )
    _mismatch(
        binding_reasons,
        label="semantic_registry_sha256_mismatch",
        actual=semantic.get("registry", {}).get("registry_sha256"),
        expected=semantic_binding.get(
            "exclusion_registry_sha256"
        ),
    )

    binding_reasons = sorted(set(binding_reasons))
    if binding_reasons:
        return {
            "schema": "V10_QWEN35_RETENTION_ADMISSION_FILE_CHECK_V1",
            "status": "HOLD",
            "binding_reasons": binding_reasons,
            "core_reasons": [],
            "effect": "READ_ONLY_ADMISSION_VERIFICATION",
        }

    core = validate_retention_admission_v3(
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
        expected_allocation=RETENTION_ALLOCATION,
        expected_per_family=int(
            packet_binding.get("cases_per_family")
        ),
    )
    return {
        **core,
        "schema": "V10_QWEN35_RETENTION_ADMISSION_FILE_CHECK_V1",
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
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--admission-policy", type=Path, required=True)
    parser.add_argument("--evidence-binding", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        result = verify_retention_admission_files(
            candidate_path=args.candidate,
            candidate_manifest_path=args.candidate_manifest,
            exclusion_path=args.exclusion,
            packet_path=args.packet,
            packet_manifest_path=args.packet_manifest,
            protocol_path=args.protocol,
            audit_output_path=args.audit_output,
            audit_receipt_path=args.audit_receipt,
            semantic_path=args.semantic,
            admission_policy_path=args.admission_policy,
            evidence_binding_path=args.evidence_binding,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "V10_QWEN35_RETENTION_ADMISSION_FILE_CHECK_V1",
            "status": "HOLD",
            "binding_reasons": [f"admission_input_error:{exc}"],
            "core_reasons": [],
            "effect": "READ_ONLY_ADMISSION_VERIFICATION",
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "RETENTION_ADMITTED_V3" else 2


if __name__ == "__main__":
    raise SystemExit(main())
