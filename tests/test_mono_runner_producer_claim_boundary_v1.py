"""Training-owned actual-source producer substitution negatives; no provider authority.

These extend test_mono_runner_untrusted_producer_v1.py pinned to exact
Mono#77 and PORTAL#62 source roots. A locally accepted Runner WorkUnit does
NOT attest its claimed Mono task, issuer, model, process, session or nonce.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_mono_runner_untrusted_producer_v1 import (
    api, source, work, seed, claim, UNAPPROVED,
)


def test_forged_source_digest_and_task_head_are_only_caller_payload(api,tmp_path):
    runtime,opened,delegation=source(api,tmp_path)
    item=work(api,opened,delegation,
              claimed_source_digest="0"*64,
              claimed_event_head="f"*64,
              claimed_task_id="fabricated-foreign-task",
              claimed_nonce="reused-untrusted-nonce")
    assert item.payload["claimed_source_digest"] != opened["source_digest"]
    assert item.payload["claimed_event_head"] != delegation.binding_event_digest
    store,budget=seed(api,tmp_path/"forged-source.db",item)
    try:
        assert claim(api,store,budget,item).lease.holder == "untrusted-holder"
        assert len(store.unresolved_attempts()) == 1
        assert not runtime.tasks.read(opened["task_id"]).closed
    finally: store.close()


def test_known_unapproved_marker_removal_changes_to_local_claim_not_verified_producer(api,tmp_path):
    _,opened,delegation=source(api,tmp_path)
    marked=work(api,opened,delegation,**{UNAPPROVED:{"producer_authenticated":False}})
    plain=work(api,opened,delegation)
    assert UNAPPROVED in marked.payload and UNAPPROVED not in plain.payload
    assert api["work_unit_fingerprint"](plain) != api["work_unit_fingerprint"](marked)
    marked_store,marked_budget=seed(api,tmp_path/"marked.db",marked)
    try:
        with pytest.raises(ValueError,match="unapproved Mono producer"):
            claim(api,marked_store,marked_budget,marked)
        assert marked_store.unresolved_attempts() == ()
    finally: marked_store.close()
    plain_store,plain_budget=seed(api,tmp_path/"unmarked.db",plain)
    try:
        assert claim(api,plain_store,plain_budget,plain).lease.holder=="untrusted-holder"
        assert len(plain_store.unresolved_attempts())==1
    finally: plain_store.close()


def test_actual_cold_task_writer_advances_A_between_runner_seed_and_local_admit(api,tmp_path):
    runtime,opened,delegation=source(api,tmp_path)
    item=work(api,opened,delegation)
    store,budget=seed(api,tmp_path/"runner-race.db",item)
    script = """
import json,os,sys
from pathlib import Path
from vera_core import QualifiedVeraRuntime, VeraStateDirectory
root, task_id, old_head, source_digest = sys.argv[1:]
rt=QualifiedVeraRuntime.from_state_directory(
    VeraStateDirectory(Path(root),project_id="TRAINING_SOURCE_PROBE",identity_id="TEST_ONLY"))
rt.tasks.checkpoint(task_id,"cold-actual-A-before-Runner-claim",
    completed_evidence=(),blockers=("HOLD",),
    protected_effects_still_gated=("PROVIDER",),
    next_frontier="SOURCE_TEST_ONLY",lifecycle_evidence_digest=source_digest,
    expected_task_head=old_head)
print(json.dumps({"child_pid":os.getpid(),
                  "task_head":rt.tasks.events(task_id)[-1].event_digest}))
"""
    try:
        child=subprocess.run(
            [sys.executable,"-B","-c",script,str(tmp_path/"mono"),
             opened["task_id"],delegation.binding_event_digest,opened["source_digest"]],
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1",
                 "PYTHONPATH":os.pathsep.join(sys.path)},
            text=True,capture_output=True,timeout=20,check=False)
        assert child.returncode==0,child.stderr
        result=json.loads(child.stdout)
        assert result["child_pid"] != os.getpid()
        assert result["task_head"] != delegation.binding_event_digest
        assert runtime.tasks.events(opened["task_id"])[-1].event_digest == result["task_head"]
        # Unmarked Runner local admit does not read the advanced Mono task head.
        assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
        assert not runtime.tasks.read(opened["task_id"]).closed
        assert len(store.unresolved_attempts())==1
    finally: store.close()
