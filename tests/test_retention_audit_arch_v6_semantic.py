from __future__ import annotations
import json

from successor.experiments.run_retention_audit_arch_v6_semantic import review_prompt_v6, run_semantic_audit_v6

def case(cid):
    return {
        "case_id":cid,"category":"test","family_id":"location:code_from_name",
        "prompt":"Return only the code.","source_id":"test","source_revision":"TEST",
        "source_terms":"TEST","source_hash":"0"*64,"generation_method":"test",
        "grader_contract":{"kind":"deterministic","grader_id":"exact_text_v1","grader_version":"1"},
        "source_evidence":{"code":"AA"},
    }

def test_v6_prompt_requires_bare_derived_answer():
    prompt=review_prompt_v6([case("case-1")])
    assert "derived_answer must contain only the bare answer requested by the case prompt" in prompt
    assert "no explanation or surrounding prose" in prompt

def test_v6_semantic_runner_uses_v6_schema_and_bare_answer_contract():
    calls=[]
    def call_structured(prompt,schema):
        calls.append((prompt,schema))
        return json.dumps({"reviews":[{
            "case_id":"case-1","observation":"DERIVED_ANSWER","derived_answer":"AA",
            "witness":None,"reason":"clear","confidence":"high"
        }]})
    result=run_semantic_audit_v6([case("case-1")],call_structured=call_structured,expected_count=1,batch_size=1,max_attempts=3)
    assert result["status"]=="SEMANTIC_AUDIT_COMPLETE"
    assert result["schema"]=="RETENTION_AUDIT_ARCH_V6_SEMANTIC_RECEIPT_V1"
    assert result["rows"][0]["derived_answer"]=="AA"
    assert len(calls)==1
