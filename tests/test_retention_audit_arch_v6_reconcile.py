from __future__ import annotations

from successor.experiments.retention_audit_arch_v6_reconcile import reconcile_audit_v6, reconcile_case_v6


def mech(answer="AA"):
    return {"case_id":"case-1","status":"MECHANICAL_VALID","defects":[],"mechanical_facts":{"derived_answer":answer}}


def sem(observation, derived_answer=None, witness="example", valid=True):
    return {
        "case_id":"case-1",
        "observation":observation,
        "derived_answer":derived_answer,
        "witness":witness,
        "witness_text":str(witness),
        "witness_contract_valid":valid,
        "reason":"test",
        "confidence":"high",
        "attempts":1,
        "prior_attempt_failures":[],
    }


def test_v6_reviewer_false_positive_is_recorded_but_nonblocking():
    row = reconcile_case_v6(mech("AA"), sem("DERIVED_ANSWER", derived_answer="ZZ", witness=None))
    assert row["classification"] == "REVIEWER_DEFECT"
    result = reconcile_audit_v6(
        {"status":"MECHANICAL_VALIDATED","rows":[mech("AA")]},
        {"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":1,"reasons":[],"rows":[sem("DERIVED_ANSWER", derived_answer="ZZ", witness=None)]},
        expected_semantic_case_ids={"case-1"},
    )
    assert result["status"] == "RETENTION_ADMITTED_ARCH_V6"
    assert result["blocking_defect_counts"] == {}
    assert result["nonblocking_finding_counts"] == {"REVIEWER_DEFECT": 1}


def test_v6_unresolved_witness_blocks():
    result = reconcile_audit_v6(
        {"status":"MECHANICAL_VALIDATED","rows":[mech()]},
        {"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":1,"reasons":[],"rows":[sem("AMBIGUITY_WITNESS")]},
        expected_semantic_case_ids={"case-1"},
    )
    assert result["status"] == "RETENTION_HOLD_ARCH_V6"
    assert result["blocking_defect_counts"] == {"UNRESOLVED": 1}


def test_v6_contradicted_witness_becomes_nonblocking_reviewer_defect():
    result = reconcile_audit_v6(
        {"status":"MECHANICAL_VALIDATED","rows":[mech()]},
        {"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":1,"reasons":[],"rows":[sem("CONTRACT_COUNTEREXAMPLE")]},
        witness_resolutions={
            "case-1": {
                "resolution":"CONTRADICTED_BY_DETERMINISTIC_EVIDENCE",
                "evidence":"Independent deterministic contract disproves witness.",
            }
        },
        expected_semantic_case_ids={"case-1"},
    )
    assert result["status"] == "RETENTION_ADMITTED_ARCH_V6"
    assert result["nonblocking_finding_counts"] == {"REVIEWER_DEFECT": 1}


def test_v6_confirmed_bank_defect_blocks():
    result = reconcile_audit_v6(
        {"status":"MECHANICAL_VALIDATED","rows":[mech()]},
        {"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":1,"reasons":[],"rows":[sem("AMBIGUITY_WITNESS")]},
        witness_resolutions={
            "case-1": {
                "resolution":"BANK_DEFECT_CONFIRMED",
                "evidence":"Prompt contract is genuinely ambiguous.",
            }
        },
        expected_semantic_case_ids={"case-1"},
    )
    assert result["status"] == "RETENTION_HOLD_ARCH_V6"
    assert result["blocking_defect_counts"] == {"BANK_DEFECT": 1}
