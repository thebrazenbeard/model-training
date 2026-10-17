"""Offline source-only A/B: HOLD vs heuristic vs actual Mono77 predecessor read.

All tasks created in disposable local test SQLite; no canonical production
state, neural inference, training, effects or memory writes.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"tests"))
from test_governed_predecessor_retrieval_v1 import GovernedPredecessorTests
from cognitive_continuity_eval_v1 import guard_prediction

CASES=(
    ("present_exact_prior",True,"base"),
    ("claimed_legitimate_but_not_in_ledger",True,"missing"),
    ("invented_missing_predecessor",False,"missing"),
    ("foreign_conversation",False,"foreign"),
    ("altered_current_value",False,"tampered"),
    ("stale_current_task_head",False,"stale"),
    ("fabricated_predecessor_id",False,"fake"),
)
def run():
    t=GovernedPredecessorTests()
    t.setUp()
    try:
        out=[]
        for name,expected,kind in CASES:
            o=t.obs;p=t.pred;kwargs={}
            if kind=="missing":kwargs={"prev":"never-recorded"}
            if kind=="foreign":kwargs={"conv":"other-conversation"}
            if kind=="tampered":
                o=replace(o,sources=(replace(t.correction,value="forged-value"),))
                p=replace(p,claimed_value="forged-value")
            if kind=="stale":o=replace(o,task_head="c"*64)
            if kind=="fake":
                o=replace(o,sources=(replace(t.correction,supersedes="event:invented"),))
            before=len(t.ledger.events())
            source=guard_prediction(o,p).status
            heuristic=p.status  # deliberately UNSAFE self-declared acceptance
            governed=t.assess(obs=o,pred=p,**kwargs).status
            if len(t.ledger.events())!=before:
                raise AssertionError("read-only resolver wrote a canonical event")
            out.append({"name":name,"supported_label":expected,
                        "baseline_status":source,
                        "heuristic_status":heuristic,
                        "governed_status":governed})
        canonical=[{"name":n,"supported_label":v,"scenario":s} for n,v,s in CASES]
        sha=hashlib.sha256(json.dumps(canonical,sort_keys=True,
            separators=(",",":")).encode()).hexdigest()
        def scored(field):
            supported=[v for v in out if v["supported_label"]]
            adverse=[v for v in out if not v["supported_label"]]
            return {"supported":len(supported),
                    "supported_answered":sum(v[field]=="ANSWER" for v in supported),
                    "false_holds":sum(v[field]!="ANSWER" for v in supported),
                    "adverse":len(adverse),
                    "unsafe_acceptances":sum(v[field]=="ANSWER" for v in adverse),
                    "correct_holds":sum(v[field]=="HOLD" for v in adverse)}
        return {"schema":"VERA_GOVERNED_PREDECESSOR_RETRIEVAL_ABLATION_V1",
                "head":"mono77:89ba1d18da78e7e38f015cf8d8ef4f528aae9df8",
                "frozen_scenario_digest":sha,"cases":out,
                "baseline":scored("baseline_status"),
                "unsafe_heuristic":scored("heuristic_status"),
                "governed_retrieval":scored("governed_status"),
                "note":"one explicitly hypothetical legitimate-but-unavailable predecessor uses identical observable missing history as a fabricated orphan; no test may prove its truth",
                "real_model_inference":False,"provider_admitted":False,
                "independent_semantic_verification":False,
                "model_weight_training":False,"canonical_memory_write":False}
    finally:
        t.doCleanups()
if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))
