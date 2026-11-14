from __future__ import annotations
import json

from successor.experiments.retention_audit_arch_v5_reviewer import build_reviewer_qualification_controls_v5
from successor.experiments.retention_audit_arch_v6_reviewer import (
    build_reviewer_qualification_controls_v6,
    run_batched_reviewer_qualification_v6,
)

def test_v6_controls_are_fresh_balanced_and_disjoint_from_v5():
    controls=build_reviewer_qualification_controls_v6()
    assert len(controls)==48
    assert not ({x["case_id"] for x in controls} & {x["case_id"] for x in build_reviewer_qualification_controls_v5()})
    classes={x["defect_class"] for x in controls}
    assert len(classes)==6
    for cls in classes:
        rows=[x for x in controls if x["defect_class"]==cls]
        assert sum(x["expected_defect"] for x in rows)==4
        assert sum(not x["expected_defect"] for x in rows)==4

def test_v6_batched_qualification_passes_perfect_structured_reviewer():
    controls=build_reviewer_qualification_controls_v6()
    calls=[]
    def call_structured(prompt,schema):
        start=len(calls)*4
        batch=controls[start:start+4]
        calls.append((prompt,schema))
        return json.dumps({"reviews":[{
            "case_id":c["case_id"],
            "observed_defect":c["expected_defect"],
            "defect_class":c["defect_class"],
            "witness":{"example":"concrete"} if c["expected_defect"] else "NONE",
            "reason":"qualified",
        } for c in batch]})
    result=run_batched_reviewer_qualification_v6(call_structured=call_structured,batch_size=4,max_attempts=3)
    assert result["status"]=="REVIEWER_QUALIFIED"
    assert result["sensitivity"]==[24,24]
    assert result["specificity"]==[24,24]
    assert result["structured_contradictions"]==[]
    assert len(calls)==12
    assert all(x["attempts"]==1 for x in result["batches"])
