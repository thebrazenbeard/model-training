from __future__ import annotations

from successor.experiments.retention_audit_arch_v2_reconcile import (
    reconcile_audit,
    reconcile_case,
)


def _mechanical(status: str = "MECHANICAL_VALID", answer: str = "AA") -> dict:
    return {
        "case_id": "case-1",
        "status": status,
        "defects": [] if status == "MECHANICAL_VALID" else ["seeded"],
        "mechanical_facts": {"derived_answer": answer},
    }


def _semantic(
    observation: str = "NO_SEMANTIC_DEFECT_FOUND",
    *,
    derived_answer: str | None = "AA",
    witness: str | None = None,
) -> dict:
    return {
        "case_id": "case-1",
        "observation": observation,
        "derived_answer": derived_answer,
        "witness": witness,
        "reason": "test",
        "confidence": "high",
        "attempts": 1,
    }


def test_mechanically_proven_wrong_row_is_bank_defect() -> None:
    result = reconcile_case(
        _mechanical("BANK_DEFECT"),
        _semantic(),
    )
    assert result["classification"] == "BANK_DEFECT"


def test_unsupported_semantic_accusation_is_not_bank_defect() -> None:
    result = reconcile_case(
        _mechanical(),
        _semantic(
            "AMBIGUITY_WITNESS",
            witness="There might be another interpretation.",
        ),
        witness_resolution="CONTRADICTED_BY_MECHANICAL",
    )
    assert result["classification"] == "REVIEWER_DEFECT"


def test_unresolved_semantic_witness_stays_unresolved() -> None:
    result = reconcile_case(
        _mechanical(),
        _semantic(
            "AMBIGUITY_WITNESS",
            witness="Two grammatical parses remain possible.",
        ),
    )
    assert result["classification"] == "UNRESOLVED"


def test_derived_answer_conflicting_with_mechanical_fact_is_reviewer_defect() -> None:
    result = reconcile_case(
        _mechanical(answer="AA"),
        _semantic("DERIVED_ANSWER", derived_answer="ZZ"),
    )
    assert result["classification"] == "REVIEWER_DEFECT"


def test_binding_failure_has_priority() -> None:
    result = reconcile_case(
        _mechanical("BINDING_DEFECT"),
        _semantic(),
    )
    assert result["classification"] == "BINDING_DEFECT"


def test_audit_method_and_transport_failures_remain_distinct() -> None:
    mechanical_receipt = {
        "status": "MECHANICAL_VALIDATED",
        "rows": [_mechanical()],
    }
    semantic_receipt = {
        "status": "SEMANTIC_AUDIT_HOLD",
        "reviewed": 0,
        "reasons": ["transport_failure:case-1"],
        "rows": [],
    }

    transport = reconcile_audit(
        mechanical_receipt,
        semantic_receipt,
    )
    assert transport["status"] == "RETENTION_HOLD_ARCH_V2"
    assert transport["defect_counts"] == {"TRANSPORT_DEFECT": 1}

    method = reconcile_audit(
        mechanical_receipt,
        {
            "status": "SEMANTIC_AUDIT_COMPLETE",
            "reviewed": 1,
            "reasons": [],
            "rows": [_semantic()],
        },
        method_defects=["qualification-control-schema-ambiguous"],
    )
    assert method["defect_counts"] == {"AUDIT_METHOD_DEFECT": 1}


def test_reconciliation_is_scoped_to_frozen_semantic_packet_ids() -> None:
    other = {
        "case_id": "case-2",
        "status": "MECHANICAL_VALID",
        "defects": [],
        "mechanical_facts": {"derived_answer": "BB"},
    }
    result = reconcile_audit(
        {
            "status": "MECHANICAL_VALIDATED",
            "rows": [_mechanical(), other],
        },
        {
            "status": "SEMANTIC_AUDIT_COMPLETE",
            "reviewed": 1,
            "reasons": [],
            "rows": [_semantic()],
        },
        expected_semantic_case_ids={"case-1"},
    )

    assert result["status"] == "RETENTION_ADMITTED_ARCH_V2"
    assert result["defect_counts"] == {}


def test_semantic_case_outside_frozen_packet_is_binding_defect() -> None:
    result = reconcile_audit(
        {
            "status": "MECHANICAL_VALIDATED",
            "rows": [_mechanical()],
        },
        {
            "status": "SEMANTIC_AUDIT_COMPLETE",
            "reviewed": 1,
            "reasons": [],
            "rows": [_semantic()],
        },
        expected_semantic_case_ids={"different-case"},
    )

    assert result["status"] == "RETENTION_HOLD_ARCH_V2"
    assert result["defect_counts"] == {"BINDING_DEFECT": 2}
