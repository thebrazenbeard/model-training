"""TRAINING SOURCE EXPERIMENT ONLY. NEVER call on production task state.

Existing Mono task ledger + existing action_lock serialize global scan/checkpoint
for receiver-scoped synthetic nonce claims. A TASK_CHECKPOINT is NOT an approved
provider reservation, issuer attestation, signed result or effect permission.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from vera_core.local_conversation import open_message
from vera_core.task_execution import TaskExecutionLedger, TaskExecutionError

SCHEMA = "TRAINING_RECEIVER_SCOPED_NONCE_SOURCE_ABLATION_V1"
TAG = "NONCE_RESEARCH_ONLY_NO_PROVIDER"
CASES = (
    ("same_scope_first", True),
    ("cold_same_scope_replay", False),
    ("cross_task_same_scope_replay", False),
    ("simultaneous_distinct_task_replay", False),
    ("valid_different_nonce", True),
    ("valid_different_session", True),
    ("stale_task_event_head", False),
)
def stable(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def digest(obj):
    return sha256(stable(obj)).hexdigest()

def identity(args):
    return digest({"receiver_id": args["receiver_id"],
                  "host_generation": args["host_generation"],
                  "provider_session": args["provider_session"],
                  "nonce": args["nonce"]})

def reserve(args):
    ledger = TaskExecutionLedger(args["path"])
    claim = identity(args)
    started = time.perf_counter_ns()
    # Existing Mono PortableFileLock serializes WHOLE scan+write, globally
    # for all tasks attached to this exact task ledger; separate SQLite lock
    # file is Mono's native locking primitive, not a trust database.
    with ledger.action_lock():
        # Source is resolved from receiver-owned Mono ledger, not caller prose.
        state = ledger.read(args["task_id"])
        if (state.closed or state.opened_lifecycle_evidence_digest
                != args["source_digest"]):
            return {"accepted": False, "reason": "CANONICAL_SOURCE_MISMATCH",
                    "elapsed_ns": time.perf_counter_ns()-started}
        events = ledger.events()
        if any(e.event_type == "TASK_CHECKPOINT"
               and e.payload.get("next_frontier") == TAG
               and e.payload.get("lifecycle_evidence_digest") == claim
               for e in events):
            return {"accepted": False, "reason": "DUPLICATE_SCOPE_NONCE",
                    "elapsed_ns": time.perf_counter_ns()-started}
        try:
            state = ledger.checkpoint(
                args["task_id"], "receiver-nonce-"+claim,
                completed_evidence=("source-only-receiver-claim",),
                blockers=("UNQUALIFIED_ISSUER_AND_MODEL",),
                protected_effects_still_gated=("PROVIDER_EFFECT",),
                next_frontier=TAG,
                lifecycle_evidence_digest=claim,
                expected_task_head=args["task_event_head"])
        except TaskExecutionError as exc:
            return {"accepted": False, "reason": "MONO_TASK_CAS_REJECT",
                    "error_type": type(exc).__name__,
                    "elapsed_ns": time.perf_counter_ns()-started}
    return {"accepted": True, "reason": "SOURCE_CHECKPOINT_ONLY",
            "task_closed": state.closed,
            "elapsed_ns": time.perf_counter_ns()-started}

def child(args):
    p=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker"],
                     input=json.dumps(args), text=True, capture_output=True,
                     timeout=25, env=os.environ.copy())
    if p.returncode:
        raise RuntimeError("worker exited "+str(p.returncode)+" "+p.stderr[-500:])
    return json.loads(p.stdout)

def concurrent(args1,args2):
    # start both children and write inputs before reading either response
    cmd=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker"]
    processes=[subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,text=True,env=os.environ.copy())
               for _ in range(2)]
    for process,arg in zip(processes,(args1,args2)):
        process.stdin.write(json.dumps(arg))
        process.stdin.close()
    outputs=[]
    for process in processes:
        stdout=process.stdout.read()
        stderr=process.stderr.read()
        process.wait(timeout=25)
        if process.returncode:
            raise RuntimeError("worker exited "+str(process.returncode)+" "+stderr[-500:])
        outputs.append(json.loads(stdout))
    return outputs

def new_task(ledger,message):
    x=open_message(ledger,"canonical-fixture-conversation",message,"exact  literal "+message)
    return {"task_id":x["task_id"],"task_event_head":x["opened_head"],
            "source_digest":x["source_digest"]}

def args(path, task, nonce, *, session="session-A"):
    return {"path":str(path),"task_id":task["task_id"],
            "task_event_head":task["task_event_head"],
            "source_digest":task["source_digest"],
            "receiver_id":"receiver-fixture","host_generation":"gen-1",
            "provider_session":session,"nonce":nonce}

def evaluate():
    with tempfile.TemporaryDirectory(prefix="training-global-nonce-") as temp:
        path=Path(temp)/"mono.sqlite3"
        ledger=TaskExecutionLedger(path)
        first_task=new_task(ledger,"first")
        second_task=new_task(ledger,"second")
        third_task=new_task(ledger,"third")
        first=child(args(path,first_task,"nonce-Q"))
        cold=child(args(path,first_task,"nonce-Q"))
        cross=child(args(path,second_task,"nonce-Q"))
        race_one=new_task(ledger,"race-one")
        race_two=new_task(ledger,"race-two")
        race=concurrent(args(path,race_one,"nonce-R"),args(path,race_two,"nonce-R"))
        different=child(args(path,third_task,"nonce-Z"))
        fresh_session=child(args(path,new_task(ledger,"other-session"),
                               "nonce-Q",session="session-B"))
        # An unauthenticated caller can CLAIM a different generation.
        forged_generation=child({**args(path,new_task(ledger,"spoof-generation"),
                                  "nonce-Q"), "host_generation":"gen-SPOOF"})
        stale=child(args(path,first_task,"nonce-other"))
        source_task=new_task(ledger,"source-mismatch")
        forged_source=child({**args(path,source_task,"nonce-forged-source"),
                             "source_digest":"0"*64})
        states=ledger.tasks()
        return {
            "schema":SCHEMA,
            "fixture_sha256":digest(CASES),
            "scope":"receiver_id|host_generation|provider_session|nonce",
            "first":first,"cold_replay":cold,"different_task_replay":cross,
            "race":race,"different_nonce":different,
            "different_session":fresh_session,"stale_head":stale,
            "forged_generation":forged_generation,
            "forged_source":forged_source,
            "race_accepts":sum(x["accepted"] for x in race),
            "all_tasks_open":all(not x.closed for x in states),
            "events_only_task_open_and_checkpoint":
                all(e.event_type in ("TASK_OPENED","TASK_CHECKPOINT")
                    for e in ledger.events()),
            "real_issuer_verified":False, "real_loaded_model":False,
            "real_provider_transport":False, "memory_admitted":False,
            "effect_authorized":False,
            "admission_status":"HOLD_RESEARCH_TASK_CHECKPOINT_IS_NOT_NONCE_API"
        }

if __name__=="__main__":
    if "--worker" in sys.argv:
        print(json.dumps(reserve(json.loads(sys.stdin.read())),sort_keys=True))
    else:
        print(json.dumps(evaluate(),indent=2,sort_keys=True))
