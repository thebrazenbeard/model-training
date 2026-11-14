from __future__ import annotations


def _add(
    reasons: list[str],
    defect_classes: set[str],
    reason: str,
    defect_class: str | None = None,
) -> None:
    if reason not in reasons:
        reasons.append(reason)
    if defect_class is not None:
        defect_classes.add(defect_class)


def verify_admission_v6(
    evidence: dict,
    *,
    expected_bindings: dict,
) -> dict:
    reasons: list[str] = []
    defect_classes: set[str] = set()

    observed_bindings = evidence.get("bindings")
    if not isinstance(observed_bindings, dict):
        _add(reasons, defect_classes, "bindings_missing", "BINDING_DEFECT")
        observed_bindings = {}
    for key, expected in sorted(expected_bindings.items()):
        if observed_bindings.get(key) != expected:
            _add(
                reasons,
                defect_classes,
                f"binding_mismatch:{key}",
                "BINDING_DEFECT",
            )

    candidate_verify = evidence.get("candidate_verify", {})
    if (
        candidate_verify.get("status") != "CANDIDATE_V2_MUTATION_ADEQUATE"
        or candidate_verify.get("coding_rows") != 250
        or candidate_verify.get("reference_failures") not in ([], None)
        or candidate_verify.get("mutant_survivors") not in ([], None)
        or candidate_verify.get("noncoding_changed") != 0
    ):
        _add(
            reasons,
            defect_classes,
            "candidate_mutation_adequacy_failed",
            "BANK_DEFECT",
        )

    mechanical = evidence.get("mechanical", {})
    if mechanical.get("status") != "MECHANICAL_VALIDATED":
        _add(
            reasons,
            defect_classes,
            "mechanical_not_validated",
            "AUDIT_METHOD_DEFECT",
        )
    if mechanical.get("reviewed") != 1500:
        _add(
            reasons,
            defect_classes,
            f"mechanical_row_count:{mechanical.get('reviewed')}",
            "AUDIT_METHOD_DEFECT",
        )
    if mechanical.get("row_status_counts") != {"MECHANICAL_VALID": 1500}:
        _add(
            reasons,
            defect_classes,
            "mechanical_row_status_mismatch",
            "BANK_DEFECT",
        )
    if (
        mechanical.get("qualification", {}).get("status")
        != "VALIDATORS_QUALIFIED"
    ):
        _add(
            reasons,
            defect_classes,
            "validators_not_qualified",
            "AUDIT_METHOD_DEFECT",
        )

    semantic_screen = evidence.get("semantic_screen", {})
    if semantic_screen.get("status") != "PASS":
        _add(
            reasons,
            defect_classes,
            "semantic_screen_not_pass",
            "AUDIT_METHOD_DEFECT",
        )
    if semantic_screen.get("failure_count") not in (0, None):
        _add(
            reasons,
            defect_classes,
            "semantic_screen_failures_present",
            "AUDIT_METHOD_DEFECT",
        )
    if (
        semantic_screen.get("target_sha256")
        != expected_bindings.get("candidate_sha256")
    ):
        _add(
            reasons,
            defect_classes,
            "semantic_screen_target_mismatch",
            "BINDING_DEFECT",
        )

    reviewer = evidence.get("reviewer_qualification", {})
    if reviewer.get("status") != "REVIEWER_QUALIFIED":
        _add(
            reasons,
            defect_classes,
            "reviewer_not_qualified",
            "REVIEWER_DEFECT",
        )
    contradictions = reviewer.get("structured_contradictions")
    if contradictions not in ([], None):
        _add(
            reasons,
            defect_classes,
            "reviewer_structured_contradictions",
            "REVIEWER_DEFECT",
        )

    packet = evidence.get("packet_manifest", {})
    if packet.get("sample_rows") != 130:
        _add(
            reasons,
            defect_classes,
            f"packet_row_count:{packet.get('sample_rows')}",
            "BINDING_DEFECT",
        )
    if packet.get("family_count") != 26:
        _add(
            reasons,
            defect_classes,
            f"packet_family_count:{packet.get('family_count')}",
            "BINDING_DEFECT",
        )
    if packet.get("per_family") != 5:
        _add(
            reasons,
            defect_classes,
            f"packet_per_family:{packet.get('per_family')}",
            "BINDING_DEFECT",
        )
    if packet.get("disjoint_from_all_predecessors") is not True:
        _add(
            reasons,
            defect_classes,
            "packet_not_disjoint",
            "BINDING_DEFECT",
        )
    if packet.get("packet_sha256") != expected_bindings.get("packet_sha256"):
        _add(
            reasons,
            defect_classes,
            "packet_manifest_hash_mismatch",
            "BINDING_DEFECT",
        )

    packet_ids = packet.get("sample_case_ids")
    packet_ids_valid = (
        isinstance(packet_ids, list)
        and len(packet_ids) == 130
        and len(set(packet_ids)) == 130
        and all(isinstance(item, str) and item for item in packet_ids)
    )
    if not packet_ids_valid:
        _add(
            reasons,
            defect_classes,
            "packet_case_ids_invalid",
            "BINDING_DEFECT",
        )

    semantic = evidence.get("semantic", {})
    if semantic.get("status") != "SEMANTIC_AUDIT_COMPLETE":
        _add(
            reasons,
            defect_classes,
            "semantic_audit_not_complete",
            "AUDIT_METHOD_DEFECT",
        )
    if semantic.get("reviewed") != 130:
        _add(
            reasons,
            defect_classes,
            f"semantic_row_count:{semantic.get('reviewed')}",
            "AUDIT_METHOD_DEFECT",
        )
    semantic_rows = semantic.get("rows")
    semantic_ids = (
        [row.get("case_id") for row in semantic_rows]
        if isinstance(semantic_rows, list)
        and all(isinstance(row, dict) for row in semantic_rows)
        else None
    )
    if (
        not packet_ids_valid
        or semantic_ids is None
        or semantic_ids != packet_ids
    ):
        _add(
            reasons,
            defect_classes,
            "semantic_packet_case_ids_mismatch",
            "BINDING_DEFECT",
        )

    reconciliation = evidence.get("reconciliation", {})
    if reconciliation.get("status") != "RETENTION_ADMITTED_ARCH_V6":
        _add(reasons, defect_classes, "reconciliation_hold")
    blocking = reconciliation.get("blocking_defect_counts")
    if blocking not in ({}, None):
        if isinstance(blocking, dict):
            defect_classes.update(
                key
                for key, value in blocking.items()
                if value and isinstance(key, str)
            )
        _add(reasons, defect_classes, "reconciliation_blocking_defects_present")

    nonblocking = reconciliation.get("nonblocking_finding_counts")
    if not isinstance(nonblocking, dict):
        nonblocking = {}

    return {
        "schema": "RETENTION_ADMISSION_ARCH_V6_RESULT_V1",
        "status": (
            "RETENTION_ADMITTED_ARCH_V6"
            if not reasons
            else "RETENTION_HOLD_ARCH_V6"
        ),
        "reasons": reasons,
        "defect_classes": sorted(defect_classes),
        "nonblocking_findings": dict(sorted(nonblocking.items())),
    }
