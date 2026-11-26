from __future__ import annotations

from successor.experiments.verify_retention_admission_arch_v5 import (
    verify_admission_v5,
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
            "structured_contradictions": [],
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
            "rows": [{"case_id": case_id} for case_id in sample_ids],
        },
        "reconciliation": {
            "status": "RETENTION_ADMITTED_ARCH_V5",
            "defect_counts": {},
        },
        "bindings": dict(EXPECTED),
    }


def test_v5_exact_complete_evidence_admits() -> None:
    result = verify_admission_v5(
        _evidence(),
        expected_bindings=EXPECTED,
    )

    assert result["status"] == "RETENTION_ADMITTED_ARCH_V5"
    assert result["schema"] == "RETENTION_ADMISSION_ARCH_V5_RESULT_V1"
    assert result["reasons"] == []


def test_v5_reviewer_defect_holds_without_bank_defect() -> None:
    evidence = _evidence()
    evidence["reconciliation"] = {
        "status": "RETENTION_HOLD_ARCH_V5",
        "defect_counts": {"REVIEWER_DEFECT": 1},
    }

    result = verify_admission_v5(
        evidence,
        expected_bindings=EXPECTED,
    )

    assert result["status"] == "RETENTION_HOLD_ARCH_V5"
    assert "REVIEWER_DEFECT" in result["defect_classes"]
    assert "BANK_DEFECT" not in result["defect_classes"]
