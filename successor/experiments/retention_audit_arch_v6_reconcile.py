from __future__ import annotations

from collections import Counter
from typing import Any

BLOCKING_CLASSES = {
    "BANK_DEFECT",
    "AUDIT_METHOD_DEFECT",
    "TRANSPORT_DEFECT",
    "BINDING_DEFECT",
    "UNRESOLVED",
}
NONBLOCKING_CLASSES = {"REVIEWER_DEFECT"}
ALLOWED_RESOLUTIONS = {
    "CONTRADICTED_BY_DETERMINISTIC_EVIDENCE",
    "BANK_DEFECT_CONFIRMED",
    "AUDIT_METHOD_DEFECT_CONFIRMED",
}


def _finding(
    case_id: str | None,
    classification: str | None,
    reason: str,
    *,
    evidence: str | None = None,
) -> dict:
    value = {
        "case_id": case_id,
        "classification": classification,
        "reason": reason,
    }
    if evidence is not None:
        value["evidence"] = evidence
    if classification is not None:
        value["blocking"] = classification in BLOCKING_CLASSES
    return value


def reconcile_case_v6(
    mechanical: dict,
    semantic: dict,
    *,
    witness_resolution: dict | None = None,
) -> dict:
    mechanical_case = mechanical.get("case_id")
    semantic_case = semantic.get("case_id")
    if mechanical_case != semantic_case:
        return _finding(
            mechanical_case,
            "BINDING_DEFECT",
            "case_id_mismatch",
        )

    mechanical_status = mechanical.get("status")
    if mechanical_status == "BINDING_DEFECT":
        return _finding(
            mechanical_case,
            "BINDING_DEFECT",
            "mechanical_binding_defect",
        )
    if mechanical_status == "BANK_DEFECT":
        return _finding(
            mechanical_case,
            "BANK_DEFECT",
            "mechanically_proven_bank_defect",
        )
    if mechanical_status != "MECHANICAL_VALID":
        return _finding(
            mechanical_case,
            "AUDIT_METHOD_DEFECT",
            f"unexpected_mechanical_status:{mechanical_status}",
        )

    if semantic.get("witness_contract_valid") is False:
        return _finding(
            mechanical_case,
            "REVIEWER_DEFECT",
            "semantic_witness_contract_violation",
        )

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
            return _finding(
                mechanical_case,
                "REVIEWER_DEFECT",
                "derived_answer_conflicts_with_deterministic_evidence",
            )
        return _finding(
            mechanical_case,
            None,
            "derived_answer_consistent",
        )

    if observation in {"AMBIGUITY_WITNESS", "CONTRACT_COUNTEREXAMPLE"}:
        if witness_resolution is None:
            return _finding(
                mechanical_case,
                "UNRESOLVED",
                "semantic_witness_requires_adjudication",
            )
        if not isinstance(witness_resolution, dict):
            return _finding(
                mechanical_case,
                "AUDIT_METHOD_DEFECT",
                "witness_resolution_not_object",
            )
        resolution = witness_resolution.get("resolution")
        evidence = witness_resolution.get("evidence")
        if (
            resolution not in ALLOWED_RESOLUTIONS
            or not isinstance(evidence, str)
            or not evidence.strip()
        ):
            return _finding(
                mechanical_case,
                "AUDIT_METHOD_DEFECT",
                "witness_resolution_invalid",
            )
        if resolution == "CONTRADICTED_BY_DETERMINISTIC_EVIDENCE":
            return _finding(
                mechanical_case,
                "REVIEWER_DEFECT",
                "semantic_witness_contradicted_by_deterministic_evidence",
                evidence=evidence,
            )
        if resolution == "BANK_DEFECT_CONFIRMED":
            return _finding(
                mechanical_case,
                "BANK_DEFECT",
                "semantic_witness_confirmed_bank_defect",
                evidence=evidence,
            )
        return _finding(
            mechanical_case,
            "AUDIT_METHOD_DEFECT",
            "semantic_witness_confirmed_audit_method_defect",
            evidence=evidence,
        )

    if observation == "CANNOT_DETERMINE":
        return _finding(
            mechanical_case,
            "UNRESOLVED",
            "reviewer_cannot_determine",
        )

    if observation == "NO_SEMANTIC_DEFECT_FOUND":
        return _finding(
            mechanical_case,
            None,
            "no_semantic_defect_found",
        )

    return _finding(
        mechanical_case,
        "AUDIT_METHOD_DEFECT",
        f"unexpected_semantic_observation:{observation}",
    )


def reconcile_audit_v6(
    mechanical_receipt: dict,
    semantic_receipt: dict,
    *,
    method_defects: list[str] | None = None,
    witness_resolutions: dict[str, dict] | None = None,
    expected_semantic_case_ids: set[str] | None = None,
) -> dict:
    findings: list[dict[str, Any]] = []
    witness_resolutions = dict(witness_resolutions or {})
    expected_ids = (
        set(expected_semantic_case_ids)
        if expected_semantic_case_ids is not None
        else None
    )

    for defect in method_defects or []:
        findings.append(
            _finding(None, "AUDIT_METHOD_DEFECT", str(defect))
        )

    mechanical_rows = mechanical_receipt.get("rows", [])
    mechanical_by_id = {
        row.get("case_id"): row
        for row in mechanical_rows
        if isinstance(row, dict)
    }

    if mechanical_receipt.get("status") != "MECHANICAL_VALIDATED":
        bad_rows = [
            row for row in mechanical_rows
            if isinstance(row, dict)
            and row.get("status") != "MECHANICAL_VALID"
        ]
        if bad_rows:
            for row in bad_rows:
                status = row.get("status")
                classification = (
                    "BINDING_DEFECT"
                    if status == "BINDING_DEFECT"
                    else "BANK_DEFECT"
                    if status == "BANK_DEFECT"
                    else "AUDIT_METHOD_DEFECT"
                )
                findings.append(
                    _finding(
                        row.get("case_id"),
                        classification,
                        f"mechanical_status:{status}",
                    )
                )
        else:
            findings.append(
                _finding(
                    None,
                    "AUDIT_METHOD_DEFECT",
                    "mechanical_receipt_not_validated",
                )
            )

    semantic_status = semantic_receipt.get("status")
    if semantic_status == "SEMANTIC_AUDIT_HOLD":
        reasons = semantic_receipt.get("reasons", [])
        transport = [
            reason for reason in reasons
            if isinstance(reason, str)
            and reason.startswith("transport_failure:")
        ]
        if transport:
            findings.append(
                _finding(
                    transport[0].split(":", 1)[1],
                    "TRANSPORT_DEFECT",
                    transport[0],
                )
            )
        else:
            findings.append(
                _finding(
                    None,
                    "AUDIT_METHOD_DEFECT",
                    "semantic_audit_hold_without_transport_failure",
                )
            )
    elif semantic_status == "SEMANTIC_AUDIT_COMPLETE":
        semantic_rows = semantic_receipt.get("rows", [])
        if not isinstance(semantic_rows, list):
            findings.append(
                _finding(
                    None,
                    "BINDING_DEFECT",
                    "semantic_rows_not_list",
                )
            )
            semantic_rows = []

        seen: set[str] = set()
        for semantic in semantic_rows:
            if not isinstance(semantic, dict):
                findings.append(
                    _finding(
                        None,
                        "BINDING_DEFECT",
                        "semantic_row_not_object",
                    )
                )
                continue
            case_id = semantic.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                findings.append(
                    _finding(
                        None,
                        "BINDING_DEFECT",
                        "semantic_case_id_invalid",
                    )
                )
                continue
            if case_id in seen:
                findings.append(
                    _finding(
                        case_id,
                        "BINDING_DEFECT",
                        "duplicate_semantic_case_id",
                    )
                )
                continue
            seen.add(case_id)
            if expected_ids is not None and case_id not in expected_ids:
                findings.append(
                    _finding(
                        case_id,
                        "BINDING_DEFECT",
                        "semantic_case_outside_frozen_packet",
                    )
                )
                continue
            mechanical = mechanical_by_id.get(case_id)
            if mechanical is None:
                findings.append(
                    _finding(
                        case_id,
                        "BINDING_DEFECT",
                        "semantic_case_missing_from_mechanical_evidence",
                    )
                )
                continue
            result = reconcile_case_v6(
                mechanical,
                semantic,
                witness_resolution=witness_resolutions.get(case_id),
            )
            if result["classification"] is not None:
                findings.append(result)

        if expected_ids is not None:
            for case_id in sorted(expected_ids - seen):
                findings.append(
                    _finding(
                        case_id,
                        "BINDING_DEFECT",
                        "frozen_packet_case_missing_semantic_review",
                    )
                )
    else:
        findings.append(
            _finding(
                None,
                "AUDIT_METHOD_DEFECT",
                f"unexpected_semantic_status:{semantic_status}",
            )
        )

    counts = Counter(
        item["classification"]
        for item in findings
        if item.get("classification")
    )
    blocking = {
        key: value
        for key, value in sorted(counts.items())
        if key in BLOCKING_CLASSES and value
    }
    nonblocking = {
        key: value
        for key, value in sorted(counts.items())
        if key in NONBLOCKING_CLASSES and value
    }
    return {
        "schema": "RETENTION_AUDIT_ARCH_V6_RECONCILIATION_V1",
        "status": (
            "RETENTION_ADMITTED_ARCH_V6"
            if not blocking
            else "RETENTION_HOLD_ARCH_V6"
        ),
        "blocking_defect_counts": blocking,
        "nonblocking_finding_counts": nonblocking,
        "defect_counts": dict(sorted(counts.items())),
        "findings": findings,
        "reviewer_false_positives_are_nonblocking": True,
    }
