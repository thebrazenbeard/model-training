from __future__ import annotations

from copy import deepcopy

from successor.experiments.verify_retention_admission_arch_v4 import (
    verify_admission_v4,
)


def verify_admission_v5(
    evidence: dict,
    *,
    expected_bindings: dict,
) -> dict:
    adapted = deepcopy(evidence)
    reconciliation = adapted.get("reconciliation")
    if isinstance(reconciliation, dict):
        status = reconciliation.get("status")
        if status == "RETENTION_ADMITTED_ARCH_V5":
            reconciliation["status"] = "RETENTION_ADMITTED_ARCH_V4"
        elif status == "RETENTION_HOLD_ARCH_V5":
            reconciliation["status"] = "RETENTION_HOLD_ARCH_V4"

    result = verify_admission_v4(
        adapted,
        expected_bindings=expected_bindings,
    )
    result = dict(result)
    result["schema"] = "RETENTION_ADMISSION_ARCH_V5_RESULT_V1"
    result["status"] = (
        "RETENTION_ADMITTED_ARCH_V5"
        if result.get("status") == "RETENTION_ADMITTED_ARCH_V4"
        else "RETENTION_HOLD_ARCH_V5"
    )
    return result
