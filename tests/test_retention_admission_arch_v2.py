from __future__ import annotations

from copy import deepcopy

from successor.experiments.verify_retention_admission_arch_v2 import (
    verify_admission,
)


EXPECTED = {
    "candidate_sha256": "a" * 64,
    "candidate_manifest_sha256": "b" * 64,
    "exclusion_sha256": "c" * 64,
    "semantic_screen_sha256": "d" * 64,
    "packet_sha256": "e" * 64,
    "protocol_sha256": "f" * 64,
    "reviewer_identity_sha256": "1" * 64,
}


def _evidence() -> dict:
    sample_ids = [f"case-{index:03d}" for index in range(130)]
    return {
        "mechanical": {
            "status": "MECHANICAL_VALIDATED",
            "reviewed": 1500,
            "row_status_counts": {"MECHANICAL_VALID": 1500},
            "qualification": {"status": "VALIDATORS_QUALIFIED"},
        },
        "reviewer_qualification": {
            "status": "REVIEWER_QUALIFIED",
        },
        "packet_manifest": {
            "sample_rows": 130,
            "family_count": 26,
            "per_family": 5,
            "disjoint_from_all_predecessors": True,
            "packet_sha256": EXPECTED["packet_sha256"],
            "sample_case_ids": sample_ids,
        },
        "semantic": {
            "status": "SEMANTIC_AUDIT_COMPLETE",
            "reviewed": 130,
            "rows": [
                {"case_id": case_id}
                for case_id in sample_ids
            ],
        },
        "reconciliation": {
            "status": "RETENTION_ADMITTED_ARCH_V2",
            "defect_counts": {},
        },
        "bindings": dict(EXPECTED),
    }


def test_exact_complete_evidence_admits_retention() -> None:
    result = verify_admission(_evidence(), expected_bindings=EXPECTED)

    assert result["status"] == "RETENTION_ADMITTED_ARCH_V2"
    assert result["reasons"] == []


def test_hash_mismatch_is_binding_defect_and_hold() -> None:
    evidence = _evidence()
    evidence["bindings"]["packet_sha256"] = "9" * 64

    result = verify_admission(evidence, expected_bindings=EXPECTED)

    assert result["status"] == "RETENTION_HOLD_ARCH_V2"
    assert "binding_mismatch:packet_sha256" in result["reasons"]
    assert result["defect_classes"] == ["BINDING_DEFECT"]


def test_incomplete_mechanical_or_semantic_evidence_holds() -> None:
    mechanical = _evidence()
    mechanical["mechanical"]["reviewed"] = 1499
    result = verify_admission(mechanical, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V2"
    assert "mechanical_row_count:1499" in result["reasons"]

    semantic = _evidence()
    semantic["semantic"]["reviewed"] = 129
    result = verify_admission(semantic, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V2"
    assert "semantic_row_count:129" in result["reasons"]


def test_reviewer_or_reconciliation_defect_holds() -> None:
    reviewer = _evidence()
    reviewer["reviewer_qualification"]["status"] = "REVIEWER_UNQUALIFIED"
    result = verify_admission(reviewer, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V2"

    reconciliation = _evidence()
    reconciliation["reconciliation"] = {
        "status": "RETENTION_HOLD_ARCH_V2",
        "defect_counts": {"UNRESOLVED": 1},
    }
    result = verify_admission(reconciliation, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V2"
    assert "reconciliation_hold" in result["reasons"]


def test_semantic_case_ids_must_exactly_match_packet() -> None:
    evidence = _evidence()
    evidence["semantic"]["rows"][0]["case_id"] = "outside-packet"

    result = verify_admission(evidence, expected_bindings=EXPECTED)

    assert result["status"] == "RETENTION_HOLD_ARCH_V2"
    assert "semantic_packet_case_ids_mismatch" in result["reasons"]
    assert "BINDING_DEFECT" in result["defect_classes"]
