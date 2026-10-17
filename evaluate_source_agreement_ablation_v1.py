"""Frozen synthetic A/B: former unique-source guard vs current agreement guard.

No model execution. Labels describe whether an exact scoped source restatement
should be preserved; they are not evidence of independent real-world truth.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import json
import time
import tracemalloc

from cognitive_continuity_eval_v1 import (
    CognitiveObservation,CognitivePrediction,CognitiveSource,guard_prediction,
)

def frozen_cases():
    source=CognitiveSource("ev:original","machine","veraos","Lappy")
    base=CognitiveObservation(
        case_id="ablation:base",category="multi_turn",task_id="task:one",
        request_id="request:one",utterance="Which host did I specify?",
        sources=(source,),referent="machine",scope="veraos",
        task_head="1"*64,accepted_task_head="1"*64,
        request_source_digest="2"*64,accepted_source_digest="2"*64)
    pred=CognitivePrediction(
        status="ANSWER",task_id=base.task_id,request_id=base.request_id,
        referent="machine",scope="veraos",claimed_value="Lappy",
        source_event_id="ev:original")
    agree=CognitiveSource("ev:agree","machine","veraos","Lappy")
    conflict=CognitiveSource("ev:conflict","machine","veraos","VN1")
    return (
        ("single_source",base,pred,True),
        ("agreeing_first",replace(base,sources=(source,agree)),pred,True),
        ("agreeing_second",replace(base,sources=(source,agree)),
         replace(pred,source_event_id="ev:agree"),True),
        ("conflict",replace(base,sources=(source,conflict)),pred,False),
        ("stale_head",replace(base,task_head="3"*64),pred,False),
        ("fabricated_source",base,replace(pred,source_event_id="ev:invented"),False),
    )

def previous_unique_source_guard(obs,pred):
    """Recreate only the frozen pre-change unique-active-source constraint."""
    result=guard_prediction(obs,pred)
    if result.status != "ANSWER":
        return result
    candidates=[s for s in obs.sources
                if s.referent==obs.referent and s.scope==obs.scope]
    superseded={s.supersedes for s in candidates if s.supersedes is not None}
    active=[s for s in candidates if s.source_event_id not in superseded]
    if len(active)!=1:
        return CognitivePrediction(status="HOLD",task_id=pred.task_id,
                                   request_id=pred.request_id)
    return result

def score(cases,guard):
    # The expectation label is held only by the scorer.
    rows=[]
    for name,obs,pred,expected_support in cases:
        result=guard(obs,pred)
        rows.append({"case":name,"supported":expected_support,
                     "status":result.status})
    preserved=sum(x["supported"] and x["status"]=="ANSWER" for x in rows)
    false_holds=sum(x["supported"] and x["status"]!="ANSWER" for x in rows)
    unsafe=sum(not x["supported"] and x["status"]=="ANSWER" for x in rows)
    correct_holds=sum(not x["supported"] and x["status"]=="HOLD" for x in rows)
    return {"supported":sum(x["supported"] for x in rows),
            "preserved":preserved,"false_holds":false_holds,
            "unsupported":sum(not x["supported"] for x in rows),
            "correct_holds":correct_holds,"unsafe_acceptances":unsafe,
            "cases":rows}

def benchmark(guard,cases,iterations=1000):
    tracemalloc.start()
    start=time.perf_counter_ns()
    for _ in range(iterations):
        for _,obs,pred,_ in cases:
            guard(obs,pred)
    elapsed=time.perf_counter_ns()-start
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"iterations":iterations,"calls":iterations*len(cases),
            "nanoseconds_per_call":round(elapsed/(iterations*len(cases))),
            "python_tracemalloc_peak_bytes":peak,
            "scope":"local CPython microbenchmark, not model latency/VRAM"}

def evaluate():
    cases=frozen_cases()
    identifiers=[(name,obs.case_id,obs.task_id,obs.request_id,
                  [(s.source_event_id,s.value,s.kind,s.supersedes)
                   for s in obs.sources],pred.source_event_id,supported)
                 for name,obs,pred,supported in cases]
    digest=sha256(json.dumps(identifiers,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"schema":"VERA_TRAINING_SOURCE_AGREEMENT_ABLATION_V1",
            "fixture_sha256":digest,
            "previous_baseline":score(cases,previous_unique_source_guard),
            "candidate":score(cases,guard_prediction),
            "cost_baseline":benchmark(previous_unique_source_guard,cases),
            "cost_candidate":benchmark(guard_prediction,cases),
            "model_inference":False,"new_model_weight_training":False,
            "independent_heldout":False,"independent_truth_assurance":False,
            "real_provider_admission":False}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,sort_keys=True))
