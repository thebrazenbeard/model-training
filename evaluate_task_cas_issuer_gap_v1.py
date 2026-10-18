"""Actual Mono task-CAS source ceiling, NOT receiver/model/provider admission.

All writes are disposable canonical task CHECKPOINTs. No real trust store,
receiver, loaded model, provider result, memory or protected effects exist.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from vera_core.local_conversation import open_message
from vera_core.task_execution import TaskExecutionError, TaskExecutionLedger

CASES = (
    "baseline_untrusted_forged_receiver_accepted_as_task_checkpoint",
    "own_task_stale_event_CAS_rejects",
    "unrelated_task_B_global_tip_does_not_stale_A",
    "same_task_old_head_replay_rejects",
    "same_nonce_different_tasks_accepted_without_receiver_authority",
    "same_nonce_different_roots_accepted_without_shared_writer",
    "cold_process_crash_before_task_checkpoint",
    "cold_process_crash_after_checkpoint_leaves_UNKNOWN_without_result",
)
EVIDENCE_SCOPE = "SOURCE_CAS_ONLY_NOT_PROVIDER_CLAIM"

def _sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _open(ledger, name):
    return open_message(ledger, "research-conversation", name, "literal-"+name)

def checkpoint(ledger, task, marker, issuer="caller-claims-qualified"):
    return ledger.checkpoint(
        task["task_id"], marker,
        completed_evidence=("UNTRUSTED_ISSUER:"+issuer,
                            "UNTRUSTED_MODEL_SHA:"+"9"*64,
                            "UNTRUSTED_ROUTE:route-forged"),
        blockers=("REAL_RECEIVER_UNVERIFIED",),
        protected_effects_still_gated=("PROVIDER", "MEMORY", "EFFECT"),
        next_frontier="TASK_CAS_NOT_PROVIDER_PERMIT",
        lifecycle_evidence_digest="f"*64,
        expected_task_head=task["opened_head"],
    )

def _attempt(ledger, task, marker):
    try:
        state=checkpoint(ledger,task,marker)
        return {"task_checkpoint_accepted":True, "task_still_open":not state.closed}
    except TaskExecutionError as exc:
        return {"task_checkpoint_accepted":False,"error_type":type(exc).__name__}

def _child_exit(payload):
    if payload["when"] == "after":
        ledger=TaskExecutionLedger(Path(payload["path"]))
        checkpoint(ledger,payload["task"],"synthetic-checkpoint-before-crash")
    os._exit(23)

def _crash(path,task,when):
    p=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--crash"],
      input=json.dumps({"path":str(path),"task":task,"when":when}),
      text=True,capture_output=True,timeout=15,env=os.environ.copy())
    if p.returncode!=23:
        raise RuntimeError(f"unexpected child return {p.returncode}: {p.stderr[-500:]}")
    ledger=TaskExecutionLedger(path)
    return {"child_exit":p.returncode,
            "task_event_count":len(ledger.events(task["task_id"])),
            "task_still_open":not ledger.read(task["task_id"]).closed}

def evaluate():
    with TemporaryDirectory(prefix="training-cas-not-issuer-") as td:
        root=Path(td)
        ledger=TaskExecutionLedger(root/"one.sqlite3")
        forged=_open(ledger,"forged-receiver")
        forged_result=_attempt(ledger,forged,"research-untrusted-attestation")
        forged_event=ledger.events(forged["task_id"])[-1]
        forged_payload=forged_event.payload
        # A checkpoint after an initial read MUST invalidate a stale A opener.
        a=_open(ledger,"A")
        snapshot=ledger.events(a["task_id"])[-1].event_digest
        _attempt(ledger,a,"actual-A-writer")
        stale=_attempt(ledger,a,"stale-synthetic-claim")
        b=_open(ledger,"B")
        b_checkpoint=_attempt(ledger,b,"unrelated-B-writer")
        # B advances GLOBAL but leaves task C's event digest unchanged.
        c=_open(ledger,"C")
        c_head=c["opened_head"]
        _attempt(ledger,b,"second-B-stale") # deliberately rejected
        d=_open(ledger,"D")
        _attempt(ledger,d,"D-writer")
        # Latest GLOBAL head is now D; C still has its original event head.
        global_not_c=ledger.verify_chain()!=c_head
        c_good=_attempt(ledger,c,"C-legitimate-task-only")
        first=_open(ledger,"nonce-first")
        second=_open(ledger,"nonce-second")
        # Same PUBLIC marker name / nonce is scoped to each canonical task.
        n1=_attempt(ledger,first,"same-nonce")
        n2=_attempt(ledger,second,"same-nonce")
        another=TaskExecutionLedger(root/"other.sqlite3")
        other=_open(another,"other-root")
        other_n=_attempt(another,other,"same-nonce")
        pre_task=_open(ledger,"crash-pre")
        post_task=_open(ledger,"crash-post")
        pre=_crash(root/"one.sqlite3",pre_task,"before")
        post=_crash(root/"one.sqlite3",post_task,"after")
        return {
          "schema":"TRAINING_NATIVE_TASK_CAS_NOT_RECEIVER_ISSUER_V1",
          "cases_sha256":_sha(CASES),
          "cases_count":len(CASES),
          "forged_receiver_task_checkpoint_accepted":forged_result["task_checkpoint_accepted"],
          "forged_payload_contains_untrusted_claim":any("UNTRUSTED_ISSUER:" in v for v in forged_payload["completed_evidence"]),
          "stale_A_task_CAS_rejected":not stale["task_checkpoint_accepted"],
          "unchanged_C_after_other_tasks_accepted":global_not_c and c_good["task_checkpoint_accepted"],
          "old_head_replay_rejected":not _attempt(ledger,first,"another-nonce")["task_checkpoint_accepted"],
          "same_nonce_distinct_tasks_both_accepted":n1["task_checkpoint_accepted"] and n2["task_checkpoint_accepted"],
          "same_nonce_other_root_accepted":other_n["task_checkpoint_accepted"],
          "precommit_crash":pre,"postcommit_crash":post,
          "postcommit_claim_has_no_verified_result":post["task_event_count"]==2 and post["task_still_open"],
          "all_tasks_unclosed":all(not t.closed for t in ledger.tasks()) and not another.read(other["task_id"]).closed,
          "real_receiver_authenticated":False,
          "real_loaded_artifact_verified":False,
          "global_nonce_consume_once_proven":False,
          "selected_provider_route_verified":False,
          "provider_result_admitted":False,
          "memory_promoted":False,
          "effect_performed":False,
          "classification":"SOURCE_TASK_CAS_VALID_AUTHENTICATED_PROVIDER_CLAIM_HOLD",
        }

if __name__=="__main__":
    if "--crash" in sys.argv:
        _child_exit(json.loads(sys.stdin.read()))
    else:
        print(json.dumps(evaluate(),indent=2,sort_keys=True))
