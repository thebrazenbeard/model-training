from __future__ import annotations
import hashlib,json
from pathlib import Path
import pytest

from successor.experiments.build_retention_audit_arch_v2_packet import packet_bytes
from successor.experiments.retention_audit_arch_v6_reviewer import build_reviewer_qualification_controls_v6
from successor.experiments.run_retention_audit_arch_v4_live import exclusive_execution_lock
from successor.experiments.run_retention_audit_arch_v6_live import run_live_review_v6

IDENTITY={
    "provider":"OLLAMA_LOCAL","model":"ministral-3:14b",
    "model_blob_sha256":"bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e",
    "runtime":"ollama version is 0.34.2","temperature":0,"seed":20261001,
    "url":"http://127.0.0.1:11434/api/generate","format":"DYNAMIC_JSON_SCHEMA",
    "stream":False,"structured_output":"JSON_SCHEMA",
}

def packet():
    return [{
        "case_id":f"case-{i}","category":"test","family_id":"location:code_from_name",
        "prompt":f"Prompt {i}","source_id":f"source:{i}","source_revision":"TEST",
        "source_terms":"TEST","source_hash":"0"*64,"generation_method":"test",
        "grader_contract":{"kind":"deterministic","grader_id":"test","grader_version":"1"},
        "source_evidence":{"fact":i},
    } for i in range(5)]

def manifest(rows):
    return {"sample_rows":len(rows),"family_count":1,"per_family":5,
        "sample_case_ids":[r["case_id"] for r in rows],
        "packet_sha256":hashlib.sha256(packet_bytes(rows)).hexdigest(),
        "disjoint_from_all_predecessors":True}

def test_v6_live_qualifies_then_reviews(tmp_path: Path):
    controls=build_reviewer_qualification_controls_v6()
    rows=packet(); calls=[]
    def call_structured(prompt,schema):
        idx=len(calls); calls.append((prompt,schema))
        if idx<12:
            batch=controls[idx*4:(idx+1)*4]
            return json.dumps({"reviews":[{
                "case_id":c["case_id"],"observed_defect":c["expected_defect"],
                "defect_class":c["defect_class"],"witness":{"example":"x"} if c["expected_defect"] else "NONE",
                "reason":"qualified"} for c in batch]})
        return json.dumps({"reviews":[{
            "case_id":r["case_id"],"observation":"NO_SEMANTIC_DEFECT_FOUND",
            "derived_answer":None,"witness":None,"reason":"clear","confidence":"high"} for r in rows]})
    result=run_live_review_v6(rows,manifest(rows),call_structured=call_structured,identity_fn=lambda:dict(IDENTITY),qualification_batch_size=4,semantic_batch_size=5,max_attempts=3,expected_reviewer_identity=dict(IDENTITY),lock_path=tmp_path/"v6.lock")
    assert result["reviewer_qualification"]["status"]=="REVIEWER_QUALIFIED"
    assert result["semantic"]["status"]=="SEMANTIC_AUDIT_COMPLETE"
    assert result["semantic"]["reviewed"]==5
    assert len(calls)==13
    assert not (tmp_path/"v6.lock").exists()

def test_v6_duplicate_fails_before_reviewer_call(tmp_path: Path):
    rows=packet(); calls=[]
    def call_structured(prompt,schema):
        calls.append(1); raise AssertionError
    lock=tmp_path/"v6.lock"
    with exclusive_execution_lock(lock):
        with pytest.raises(RuntimeError,match="execution lock exists"):
            run_live_review_v6(rows,manifest(rows),call_structured=call_structured,identity_fn=lambda:dict(IDENTITY),expected_reviewer_identity=dict(IDENTITY),lock_path=lock)
    assert calls==[]
