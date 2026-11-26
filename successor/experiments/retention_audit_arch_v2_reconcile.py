from __future__ import annotations

from collections import Counter
from typing import Any


DEFECT_CLASSES = {
    "BANK_DEFECT",
    "REVIEWER_DEFECT",
    "AUDIT_METHOD_DEFECT",
    "TRANSPORT_DEFECT",
    "BINDING_DEFECT",
    "UNRESOLVED",
}


def reconcile_case(
    mechanical: dict,
    semantic: dict,
    *,
    witness_resolution: str | None = None,
) -> dict:
    mechanical_case = mechanical.get("case_id")
    semantic_case = semantic.get("case_id")
    if mechanical_case != semantic_case:
        return {
            "case_id": mechanical_case,
            "classification": "BINDING_DEFECT",
            "reason": "case_id_mismatch",
        }

    mechanical_status = mechanical.get("status")
    if mechanical_status == "BINDING_DEFECT":
        return {
            "case_id": mechanical_case,
            "classification": "BINDING_DEFECT",
            "reason": "mechanical_binding_defect",
        }
    if mechanical_status == "BANK_DEFECT":
        return {
            "case_id": mechanical_case,
            "classification": "BANK_DEFECT",
            "reason": "mechanically_proven_bank_defect",
        }
    if mechanical_status != "MECHANICAL_VALID":
        return {
            "case_id": mechanical_case,
            "classification": "AUDIT_METHOD_DEFECT",
            "reason": f"unexpected_mechanical_status:{mechanical_status}",
        }

    observation = semantic.get("observation")
    mechanical_answer = mechanical.get("mechanical_facts", {}).get(
        "derived_answer"
    )
    semantic_answer = semantic.get("derived_answer")

    if observation == "DERIVED_ANSWER":
        if (
            mechanical_answer is not None
            and semantic_answer is not None
            and str(semantic_answer) != str(mechanical_answer)
        ):
            return {
                "case_id": mechanical_case,
                "classification": "REVIEWER_DEFECT",
                "reason": "derived_answer_conflicts_with_mechanical_fact",
            }
        return {
            "case_id": mechanical_case,
            "classification": None,
            "reason": "derived_answer_consistent",
        }

    if observation in {"AMBIGUITY_WITNESS", "CONTRACT_COUNTEREXAMPLE"}:
        if witness_resolution == "CONTRADICTED_BY_MECHANICAL":
            return {
                "case_id": mechanical_case,
                "classification": "REVIEWER_DEFECT",
                "reason": "semantic_witness_contradicted_by_mechanical_evidence",
            }
        if witness_resolution == "SUPPORTED_BY_MECHANICAL":
            return {
                "case_id": mechanical_case,
                "classification": "AUDIT_METHOD_DEFECT",
                "reason": (
                    "semantic_witness_claimed_supported_without_mechanical_"
                    "bank_defect"
                ),
            }
        return {
            "case_id": mechanical_case,
            "classification": "UNRESOLVED",
            "reason": "semantic_witness_requires_reconciliation",
        }

    if observation == "CANNOT_DETERMINE":
        return {
            "case_id": mechanical_case,
            "classification": "UNRESOLVED",
            "reason": "reviewer_cannot_determine",
        }

    if observation == "NO_SEMANTIC_DEFECT_FOUND":
        return {
            "case_id": mechanical_case,
            "classification": None,
            "reason": "no_semantic_defect_found",
        }

    return {
        "case_id": mechanical_case,
        "classification": "AUDIT_METHOD_DEFECT",
        "reason": f"unexpected_semantic_observation:{observation}",
    }


def reconcile_audit(
    mechanical_receipt: dict,
    semantic_receipt: dict,
    *,
    method_defects: list[str] | None = None,
    witness_resolutions: dict[str, str] | None = None,
    expected_semantic_case_ids: set[str] | None = None,
) -> dict:
    classifications: list[dict[str, Any]] = []
    method_defects = list(method_defects or [])
    witness_resolutions = dict(witness_resolutions or {})

    for defect in method_defects:
        classifications.append({
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
            classifications.append({
                "case_id": row.get("case_id"),
                "classification": "BINDING_DEFECT",
                "reason": "mechanical_binding_defect",
            })
        for row in bank_rows:
            classifications.append({
                "case_id": row.get("case_id"),
                "classification": "BANK_DEFECT",
                "reason": "mechanically_proven_bank_defect",
            })
        if not binding_rows and not bank_rows:
            classifications.append({
                "case_id": None,
                "classification": "AUDIT_METHOD_DEFECT",
                "reason": "mechanical_receipt_not_validated",
            })

    semantic_status = semantic_receipt.get("status")
    if semantic_status == "SEMANTIC_AUDIT_HOLD":
        reasons = semantic_receipt.get("reasons", [])
        transport_reasons = [
            reason for reason in reasons
            if isinstance(reason, str)
            and reason.startswith("transport_failure:")
        ]
        if transport_reasons:
            classifications.append({
                "case_id": transport_reasons[0].split(":", 1)[1],
                "classification": "TRANSPORT_DEFECT",
                "reason": transport_reasons[0],
            })
        else:
            classifications.append({
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
                classifications.append({
                    "case_id": case_id,
                    "classification": "BINDING_DEFECT",
                    "reason": "semantic_case_outside_frozen_packet",
                })
                continue
            mechanical = mechanical_by_id.get(case_id)
            if mechanical is None:
                classifications.append({
                    "case_id": case_id,
                    "classification": "BINDING_DEFECT",
                    "reason": "semantic_case_missing_from_mechanical_evidence",
                })
                continue
            result = reconcile_case(
                mechanical,
                semantic,
                witness_resolution=witness_resolutions.get(case_id),
            )
            if result["classification"] is not None:
                classifications.append(result)

        if expected_ids is not None:
            missing = expected_ids - semantic_ids
            for case_id in sorted(missing):
                classifications.append({
                    "case_id": case_id,
                    "classification": "BINDING_DEFECT",
                    "reason": "frozen_packet_case_missing_semantic_review",
                })
    else:
        classifications.append({
            "case_id": None,
            "classification": "AUDIT_METHOD_DEFECT",
            "reason": f"unexpected_semantic_status:{semantic_status}",
        })

    counts = Counter(
        item["classification"]
        for item in classifications
        if item.get("classification") in DEFECT_CLASSES
    )
    status = (
        "RETENTION_ADMITTED_ARCH_V2"
        if not counts
        else "RETENTION_HOLD_ARCH_V2"
    )
    return {
        "schema": "RETENTION_AUDIT_ARCH_V2_RECONCILIATION_V1",
        "status": status,
        "defect_counts": dict(sorted(counts.items())),
        "findings": classifications,
    }
