from __future__ import annotations

from successor.experiments.retention_audit_arch_v4_reconcile import (
    reconcile_audit_v4,
    reconcile_case_v4,
)


def reconcile_case_v5(
    mechanical: dict,
    semantic: dict,
    *,
    witness_resolution: str | None = None,
) -> dict:
    return reconcile_case_v4(
        mechanical,
        semantic,
        witness_resolution=witness_resolution,
    )


def reconcile_audit_v5(
    mechanical_receipt: dict,
    semantic_receipt: dict,
    *,
    method_defects: list[str] | None = None,
    witness_resolutions: dict[str, str] | None = None,
    expected_semantic_case_ids: set[str] | None = None,
) -> dict:
    result = reconcile_audit_v4(
        mechanical_receipt,
        semantic_receipt,
        method_defects=method_defects,
        witness_resolutions=witness_resolutions,
        expected_semantic_case_ids=expected_semantic_case_ids,
    )
    result = dict(result)
    result["schema"] = "RETENTION_AUDIT_ARCH_V5_RECONCILIATION_V1"
    result["status"] = (
        "RETENTION_ADMITTED_ARCH_V5"
        if result.get("status") == "RETENTION_ADMITTED_ARCH_V4"
        else "RETENTION_HOLD_ARCH_V5"
    )
    return result
