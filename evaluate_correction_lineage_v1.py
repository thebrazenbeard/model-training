"""Frozen Git-source A/B for governed correction lineage; zero model calls.

Old baseline imported from exact previous Training Git commit. Synthetic
developer cases are disclosed, not blind/independent behavioral qualification.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

BASE_SHA="535dba1480fe59e92533248ee2badd0ae5e8711f"
ROOT=Path(__file__).resolve().parent


def prior_exact_module():
    source=subprocess.run(["git","-C",str(ROOT),"show",
        BASE_SHA+":cognitive_continuity_eval_v1.py"],
        capture_output=True,check=True)
    spec=importlib.util.spec_from_loader("cognitive_guard_previous_frozen",loader=None)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    exec(compile(source.stdout,BASE_SHA+":cognitive_continuity_eval_v1.py","exec"),
         module.__dict__)
    return module, sha256(source.stdout).hexdigest()


def cases(m):
    source=m.CognitiveSource
    a=source("event:old","purpose","role","personal")
    b=source("event:new","purpose","role","organizational","CORRECTION","event:old")
    c=source("event:final","purpose","role","team","CORRECTION","event:new")
    agree=source("event:agree","purpose","role","organizational")
    foreign_scope=replace(a,scope="personal")
    foreign_ref=replace(a,referent="individual")
    base=m.CognitiveObservation(
        case_id="lineage",category="correction",
        task_id="task:accepted",request_id="req:accepted",
        utterance="Actually, the role's purpose is organizational.",
        sources=(a,b),referent="purpose",scope="role",
        task_head="a"*64,accepted_task_head="a"*64,
        request_source_digest="b"*64,accepted_source_digest="b"*64,
        previous_interaction_id="event:old")
    pred=m.CognitivePrediction("ANSWER","task:accepted","req:accepted",
        "purpose","role","organizational","event:new")
    normal=m.CognitivePrediction("ANSWER","task:accepted","req:accepted",
        "purpose","role","personal","event:old")
    return (
      ("valid_correction",base,pred,True),
      ("valid_agreeing_sources",replace(base,sources=(a,b,agree)),pred,True),
      ("valid_uncorrected",replace(base,sources=(a,),previous_interaction_id=None),normal,True),
      ("valid_two_step",replace(base,sources=(a,b,c),previous_interaction_id="event:new"),
       replace(pred,source_event_id="event:final",claimed_value="team"),True),
      ("valid_outscope_distractor",replace(base,sources=(a,b,source("foreign","device","vm","VN1"))),pred,True),
      # A deliberate evaluator-only utility counterexample: an old, valid
      # correction exists but its original was omitted from this context
      # slice. No source gate can authenticate that absent predecessor.
      ("useful_pruned_ancestor",replace(base,sources=(b,)),pred,True),
      ("orphan_predecessor",replace(base,sources=(b,)),pred,False),
      ("untyped_overwrite",replace(base,sources=(a,replace(b,kind="USER_STATEMENT"))),pred,False),
      ("cross_scope_parent",replace(base,sources=(foreign_scope,b)),pred,False),
      ("cross_referent_parent",replace(base,sources=(foreign_ref,b)),pred,False),
      ("wrong_previous_id",replace(base,previous_interaction_id="wrong:event"),pred,False),
      ("no_named_predecessor",replace(base,sources=(replace(b,supersedes=None),),
          previous_interaction_id=None),pred,False),
      ("forged_citation",base,replace(pred,source_event_id="invented"),False),
      ("conflicting_third_record",replace(base,sources=(a,b,source("event:conflict","purpose","role","VN1"))),pred,False),
      ("duplicate_source_id",replace(base,sources=(a,b,b)),pred,False),
      ("stale_task_head",replace(base,task_head="c"*64),pred,False),
      ("changed_request",base,replace(pred,request_id="request:swapped"),False),
      ("effect_escalation",base,replace(pred,effect_authority=True),False),
    )


def score(m):
    rows=[]
    for name,obs,pred,should_answer in cases(m):
        actual=m.guard_prediction(obs,pred)
        rows.append(dict(case=name,should_answer=should_answer,
                         observed_status=actual.status,
                         visible_claim=actual.claimed_value))
    supported=[x for x in rows if x["should_answer"]]
    adverse=[x for x in rows if not x["should_answer"]]
    return {
       "supported_total":len(supported),
       "supported_preserved":sum(x["observed_status"]=="ANSWER" for x in supported),
       "supported_false_holds":sum(x["observed_status"]!="ANSWER" for x in supported),
       "adverse_total":len(adverse),
       "unsafe_answer_acceptances":sum(x["observed_status"]=="ANSWER" for x in adverse),
       "adverse_correct_holds":sum(x["observed_status"]=="HOLD" for x in adverse),
       "rows":rows,
    }


def evaluate():
    import cognitive_continuity_eval_v1 as current
    prior,prior_sha=prior_exact_module()
    old=score(prior);new=score(current)
    # Dataset digest covers expected decisions, task input and source record identity
    # from frozen previous-module copies; no expectation is supplied to guard.
    frozen=[{"case":n,"observation":asdict(o),"prediction":asdict(p),
             "expected_answer":v} for n,o,p,v in cases(prior)]
    digest=sha256(json.dumps(frozen,sort_keys=True,separators=(",",":"),
        ensure_ascii=False).encode()).hexdigest()
    start=time.perf_counter_ns()
    for _ in range(200):
        for _,obs,pred,_ in cases(current):
            current.guard_prediction(obs,pred)
    local_us=round((time.perf_counter_ns()-start)/(200*len(new["rows"]))/1000,3)
    return {
     "schema":"VERA_TRAINING_CORRECTION_LINEAGE_ABLATION_V1",
     "frozen_developer_fixture_sha256":digest,
     "previous_training_source_git":BASE_SHA,
     "previous_training_source_sha256":prior_sha,
     "source_only":True,
     "model_inference":False,
     "weight_training":False,
     "independent_fact_review":False,
     "baseline":old,
     "candidate":new,
     "candidate_local_microseconds_per_guard_call":local_us,
     "limitations":["public synthetic developer fixtures", "source labels may be forged",
        "context-pruned but legitimate predecessor would now HOLD",
        "not an installed Mono/VeraOS guard", "not Qwen/GLM cognition proof"],
    }


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,sort_keys=True))
