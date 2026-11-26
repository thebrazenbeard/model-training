from __future__ import annotations
from successor.experiments.retention_audit_arch_v7_reconcile import reconcile_audit_v7, reconcile_case_v7

def mech(answer="AA"):
    return {"case_id":"case-1","status":"MECHANICAL_VALID","defects":[],"mechanical_facts":{"derived_answer":answer}}

def sem(obs,answer=None,witness="example",valid=True):
    return {"case_id":"case-1","observation":obs,"derived_answer":answer,"witness":witness,"witness_text":str(witness),"witness_contract_valid":valid,"reason":"test","confidence":"high","attempts":1,"prior_attempt_failures":[]}

def test_v7_reviewer_false_positive_nonblocking():
    row=reconcile_case_v7(mech("AA"),sem("DERIVED_ANSWER","ZZ",None))
    assert row["classification"]=="REVIEWER_DEFECT"
    result=reconcile_audit_v7({"status":"MECHANICAL_VALIDATED","rows":[mech("AA")]},{"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":1,"reasons":[],"rows":[sem("DERIVED_ANSWER","ZZ",None)]},expected_semantic_case_ids={"case-1"})
    assert result["status"]=="RETENTION_ADMITTED_ARCH_V7"
    assert result["blocking_defect_counts"]=={}
    assert result["nonblocking_finding_counts"]=={"REVIEWER_DEFECT":1}

def test_v7_unresolved_still_blocks():
    result=reconcile_audit_v7({"status":"MECHANICAL_VALIDATED","rows":[mech()]},{"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":1,"reasons":[],"rows":[sem("AMBIGUITY_WITNESS")]},expected_semantic_case_ids={"case-1"})
    assert result["status"]=="RETENTION_HOLD_ARCH_V7"
    assert result["blocking_defect_counts"]=={"UNRESOLVED":1}
