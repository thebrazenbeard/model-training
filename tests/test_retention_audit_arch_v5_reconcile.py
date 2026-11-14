from __future__ import annotations

from successor.experiments.retention_audit_arch_v5_reconcile import (
    reconcile_audit_v5,
    reconcile_case_v5,
)


def _mechanical() -> dict:
    return {
        "case_id": "case-1",
        "status": "MECHANICAL_VALID",
        "defects": [],
        "mechanical_facts": {"derived_answer": "AA"},
    }


def _semantic(
    *,
    observation: str,
    witness_contract_valid: bool = True,
) -> dict:
    return {
        "case_id": "case-1",
        "observation": observation,
        "derived_answer": None,
        "witness": None,
        "witness_text": "null",
        "witness_contract_valid": witness_contract_valid,
        "reason": "test",
        "confidence": "high",
        "attempts": 1,
        "prior_attempt_failures": [],
    }


def test_v5_invalid_witness_is_reviewer_defect() -> None:
    result = reconcile_case_v5(
        _mechanical(),
        _semantic(
            observation="AMBIGUITY_WITNESS",
            witness_contract_valid=False,
        ),
    )

    assert result["classification"] == "REVIEWER_DEFECT"
    assert result["reason"] == "semantic_witness_contract_violation"


def test_v5_reconciliation_uses_v5_status_labels() -> None:
    result = reconcile_audit_v5(
        {
            "status": "MECHANICAL_VALIDATED",
            "rows": [_mechanical()],
        },
        {
            "status": "SEMANTIC_AUDIT_COMPLETE",
            "reviewed": 1,
            "reasons": [],
            "rows": [
                _semantic(
                    observation="NO_SEMANTIC_DEFECT_FOUND",
                    witness_contract_valid=True,
                )
            ],
        },
        expected_semantic_case_ids={"case-1"},
    )

    assert result["status"] == "RETENTION_ADMITTED_ARCH_V5"
    assert result["schema"] == "RETENTION_AUDIT_ARCH_V5_RECONCILIATION_V1"
