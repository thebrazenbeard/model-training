from __future__ import annotations

from copy import deepcopy

from successor.experiments.verify_retention_admission_arch_v6 import (
    verify_admission_v6,
)


def verify_admission_v7(
    evidence: dict,
    *,
    expected_bindings: dict,
) -> dict:
    reasons: list[str] = []
    defect_classes: set[str] = set()

    candidate_verify = evidence.get("candidate_verify", {})
    candidate_ok = (
        candidate_verify.get("status")
        == "CANDIDATE_V3_MUTATION_AND_FRESHNESS_ADEQUATE"
        and candidate_verify.get("case_count") == 1500
        and candidate_verify.get("coding_rows") == 250
        and candidate_verify.get("replacement_count") == 12
        and candidate_verify.get("reference_failures") in ([], None)
        and candidate_verify.get("mutant_survivors") in ([], None)
        and candidate_verify.get("fresh_deficits") in ({}, None)
        and candidate_verify.get("added_all_fresh") is True
        and candidate_verify.get("removed_all_consumed") is True
        and candidate_verify.get("retained_nonchunk_changed") in ([], None)
    )
    if not candidate_ok:
        reasons.append(
            "candidate_v3_freshness_or_mutation_adequacy_failed"
        )
        defect_classes.add("BANK_DEFECT")

    packet = evidence.get("packet_manifest", {})
    if packet.get("excluded_case_count") != 520:
        reasons.append(
            f"packet_excluded_case_count:{packet.get('excluded_case_count')}"
        )
        defect_classes.add("BINDING_DEFECT")

    adapted = deepcopy(evidence)
    adapted["candidate_verify"] = {
        "status": "CANDIDATE_V2_MUTATION_ADEQUATE",
        "coding_rows": 250,
        "reference_failures": [],
        "mutant_survivors": [],
        "noncoding_changed": 0,
    }

    reconciliation = adapted.get("reconciliation")
    if isinstance(reconciliation, dict):
        status = reconciliation.get("status")
        if status == "RETENTION_ADMITTED_ARCH_V7":
            reconciliation["status"] = "RETENTION_ADMITTED_ARCH_V6"
        elif status == "RETENTION_HOLD_ARCH_V7":
            reconciliation["status"] = "RETENTION_HOLD_ARCH_V6"

    base = verify_admission_v6(
        adapted,
        expected_bindings=expected_bindings,
    )
    for reason in base.get("reasons", []):
        if reason not in reasons:
            reasons.append(reason)
    defect_classes.update(base.get("defect_classes", []))

    return {
        "schema": "RETENTION_ADMISSION_ARCH_V7_RESULT_V1",
        "status": (
            "RETENTION_ADMITTED_ARCH_V7"
            if not reasons
            else "RETENTION_HOLD_ARCH_V7"
        ),
        "reasons": reasons,
        "defect_classes": sorted(defect_classes),
        "nonblocking_findings": dict(
            sorted(base.get("nonblocking_findings", {}).items())
        ),
    }
