from __future__ import annotations

from collections import Counter
from typing import Any

from successor.experiments.retention_audit_arch_v2_reconcile import (
    DEFECT_CLASSES,
    reconcile_case,
)


def reconcile_case_v4(
    mechanical: dict,
    semantic: dict,
    *,
    witness_resolution: str | None = None,
) -> dict:
    if (
        semantic.get("observation")
        in {"AMBIGUITY_WITNESS", "CONTRACT_COUNTEREXAMPLE"}
        and semantic.get("witness_contract_valid") is False
    ):
        return {
            "case_id": mechanical.get("case_id"),
            "classification": "REVIEWER_DEFECT",
            "reason": "semantic_witness_contract_violation",
        }

    return reconcile_case(
        mechanical,
        semantic,
        witness_resolution=witness_resolution,
    )


def reconcile_audit_v4(
    mechanical_receipt: dict,
    semantic_receipt: dict,
    *,
    method_defects: list[str] | None = None,
    witness_resolutions: dict[str, str] | None = None,
    expected_semantic_case_ids: set[str] | None = None,
) -> dict:
    findings: list[dict[str, Any]] = []
    method_defects = list(method_defects or [])
    witness_resolutions = dict(witness_resolutions or {})

    for defect in method_defects:
        findings.append({
            "case_id": None,
            "classification": "AUDIT_METHOD_DEFECT",
            "reason": defect,
        })

    if mechanical_receipt.get("status") != "MECHANICAL_VALIDATED":
        rows = mechanical_receipt.get("rows", [])
        binding_rows = [
            row for row in rows
            if row.get("status") == "BINDING_DEFECT"
        ]
        bank_rows = [
            row for row in rows
            if row.get("status") == "BANK_DEFECT"
        ]
        for row in binding_rows:
            findings.append({
                "case_id": row.get("case_id"),
                "classification": "BINDING_DEFECT",
                "reason": "mechanical_binding_defect",
            })
        for row in bank_rows:
            findings.append({
                "case_id": row.get("case_id"),
                "classification": "BANK_DEFECT",
                "reason": "mechanically_proven_bank_defect",
            })
        if not binding_rows and not bank_rows:
            findings.append({
                "case_id": None,
                "classification": "AUDIT_METHOD_DEFECT",
                "reason": "mechanical_receipt_not_validated",
            })

    semantic_status = semantic_receipt.get("status")
    if semantic_status == "SEMANTIC_AUDIT_HOLD":
        reasons = semantic_receipt.get("reasons", [])
        transport = [
            reason for reason in reasons
            if isinstance(reason, str)
            and reason.startswith("transport_failure:")
        ]
        if transport:
            findings.append({
                "case_id": transport[0].split(":", 1)[1],
                "classification": "TRANSPORT_DEFECT",
                "reason": transport[0],
            })
        else:
            findings.append({
                "case_id": None,
                "classification": "AUDIT_METHOD_DEFECT",
                "reason": "semantic_audit_hold_without_transport_failure",
            })
    elif semantic_status == "SEMANTIC_AUDIT_COMPLETE":
        mechanical_by_id = {
            row.get("case_id"): row
            for row in mechanical_receipt.get("rows", [])
        }
        semantic_rows = semantic_receipt.get("rows", [])
        semantic_ids: set[str] = set()
        expected_ids = (
            set(expected_semantic_case_ids)
            if expected_semantic_case_ids is not None
            else None
        )
        for semantic in semantic_rows:
            case_id = semantic.get("case_id")
            semantic_ids.add(case_id)
            if expected_ids is not None and case_id not in expected_ids:
                findings.append({
                    "case_id": case_id,
                    "classification": "BINDING_DEFECT",
                    "reason": "semantic_case_outside_frozen_packet",
                })
                continue
            mechanical = mechanical_by_id.get(case_id)
            if mechanical is None:
                findings.append({
                    "case_id": case_id,
                    "classification": "BINDING_DEFECT",
                    "reason": "semantic_case_missing_from_mechanical_evidence",
                })
                continue
            result = reconcile_case_v4(
                mechanical,
                semantic,
                witness_resolution=witness_resolutions.get(case_id),
            )
            if result["classification"] is not None:
                findings.append(result)

        if expected_ids is not None:
            for case_id in sorted(expected_ids - semantic_ids):
                findings.append({
                    "case_id": case_id,
                    "classification": "BINDING_DEFECT",
                    "reason": "frozen_packet_case_missing_semantic_review",
                })
    else:
        findings.append({
            "case_id": None,
            "classification": "AUDIT_METHOD_DEFECT",
            "reason": f"unexpected_semantic_status:{semantic_status}",
        })

    counts = Counter(
        item["classification"]
        for item in findings
        if item.get("classification") in DEFECT_CLASSES
    )
    return {
        "schema": "RETENTION_AUDIT_ARCH_V4_RECONCILIATION_V1",
        "status": (
            "RETENTION_ADMITTED_ARCH_V4"
            if not counts
            else "RETENTION_HOLD_ARCH_V4"
        ),
        "defect_counts": dict(sorted(counts.items())),
        "findings": findings,
    }
