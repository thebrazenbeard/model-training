from __future__ import annotations

from successor.experiments.retention_audit_arch_v4_reconcile import (
    reconcile_audit_v4,
    reconcile_case_v4,
)


def _mechanical(case_id: str = "case-1", answer: str = "AA") -> dict:
    return {
        "case_id": case_id,
        "status": "MECHANICAL_VALID",
        "defects": [],
        "mechanical_facts": {"derived_answer": answer},
    }


def _semantic(
    *,
    observation: str,
    witness_contract_valid: bool = True,
    derived_answer: str | None = None,
) -> dict:
    return {
        "case_id": "case-1",
        "observation": observation,
        "derived_answer": derived_answer,
        "witness": None,
        "witness_text": "null",
        "witness_contract_valid": witness_contract_valid,
        "reason": "test",
        "confidence": "high",
        "attempts": 1,
        "prior_attempt_failures": [],
    }


def test_v4_invalid_required_witness_is_reviewer_defect() -> None:
    result = reconcile_case_v4(
        _mechanical(),
        _semantic(
            observation="AMBIGUITY_WITNESS",
            witness_contract_valid=False,
        ),
    )

    assert result["classification"] == "REVIEWER_DEFECT"
    assert result["reason"] == "semantic_witness_contract_violation"


def test_v4_valid_nonmechanical_witness_remains_unresolved() -> None:
    result = reconcile_case_v4(
        _mechanical(),
        _semantic(
            observation="AMBIGUITY_WITNESS",
            witness_contract_valid=True,
        ),
    )

    assert result["classification"] == "UNRESOLVED"


def test_v4_reconciliation_uses_v4_status_labels() -> None:
    mechanical = {
        "status": "MECHANICAL_VALIDATED",
        "rows": [_mechanical()],
    }
    semantic = {
        "status": "SEMANTIC_AUDIT_COMPLETE",
        "reviewed": 1,
        "reasons": [],
        "rows": [
            _semantic(
                observation="NO_SEMANTIC_DEFECT_FOUND",
                witness_contract_valid=True,
            )
        ],
    }

    result = reconcile_audit_v4(
        mechanical,
        semantic,
        expected_semantic_case_ids={"case-1"},
    )

    assert result["status"] == "RETENTION_ADMITTED_ARCH_V4"
    assert result["defect_counts"] == {}
