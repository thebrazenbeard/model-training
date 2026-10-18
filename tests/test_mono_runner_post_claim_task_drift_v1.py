"""Cross-store cold task drift after a valid LOCAL Runner claim.

Pins real Mono #77 and PORTAL #62 code via the #118 source fixture.
This proves nothing about a trusted producer, model or external effect.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_mono_runner_untrusted_producer_v1 import api, source, work, seed, claim

COLD_ADVANCE = r"""
import json,os,sys
from pathlib import Path
from vera_core import QualifiedVeraRuntime,VeraStateDirectory
root,task_id,expected,source_digest=sys.argv[1:]
runtime=QualifiedVeraRuntime.from_state_directory(
    VeraStateDirectory(Path(root),project_id="TRAINING_SOURCE_PROBE",identity_id="TEST_ONLY"))
state=runtime.tasks.checkpoint(task_id,"cold-A-after-local-Runner-claim",
    completed_evidence=(),blockers=("NO_PROVIDER_RESULT",),
    protected_effects_still_gated=("PROVIDER_EFFECT",),
    next_frontier="UNKNOWN_NO_REDISPATCH",
    lifecycle_evidence_digest=source_digest,expected_task_head=expected)
print(json.dumps({"pid":os.getpid(),
    "task_event_head":runtime.tasks.events(task_id)[-1].event_digest,
    "closed":state.closed}))
"""


def cold_checkpoint_A(tmp_path, opened, delegation):
    child = subprocess.run([
        sys.executable,"-B","-c",COLD_ADVANCE,str(tmp_path/"mono"),
        opened["task_id"],delegation.binding_event_digest,
        opened["source_digest"]],
        env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1",
             "PYTHONPATH":os.pathsep.join(sys.path)},
        text=True,capture_output=True,timeout=20,check=False)
    assert child.returncode==0,child.stderr
    response=json.loads(child.stdout)
    assert response["pid"]!=os.getpid()
    assert response["closed"] is False
    assert response["task_event_head"]!=delegation.binding_event_digest
    return response


def test_runner_admission_then_cold_mono_checkpoint_requires_unknown_not_done(api,tmp_path):
    runtime,opened,delegation=source(api,tmp_path)
    item=work(api,opened,delegation)
    db=tmp_path/"runner-existing-root.sqlite3"
    store,budget=seed(api,db,item)
    try:
        lease=claim(api,store,budget,item)
        assert lease.lease.holder=="untrusted-holder"
        assert len(store.unresolved_attempts())==1
    finally:
        store.close()
    advanced=cold_checkpoint_A(tmp_path,opened,delegation)
    cold_store=api["SqliteDispatchAdmissionStore"](db)
    try:
        assert len(cold_store.unresolved_attempts())==1
        with pytest.raises(ValueError):
            claim(api,cold_store,budget,item)
        assert len(cold_store.unresolved_attempts())==1
        assert runtime.tasks.events(opened["task_id"])[-1].event_digest==advanced["task_event_head"]
        assert not runtime.tasks.read(opened["task_id"]).closed
        assert runtime.pc_execution_transport is None
        assert not runtime.provider_execution_transports
    finally:
        cold_store.close()


def test_independent_runner_root_still_local_claim_not_shared_nonce(api,tmp_path):
    runtime,opened,delegation=source(api,tmp_path)
    item=work(api,opened,delegation,claimed_nonce="SAME_UNTRUSTED_NONCE")
    first,budget=seed(api,tmp_path/"runner-1.sqlite3",item)
    try:
        assert claim(api,first,budget,item).lease.holder=="untrusted-holder"
    finally:
        first.close()
    cold_checkpoint_A(tmp_path,opened,delegation)
    # This independent SQLite root knows neither prior Runner attempt nor
    # current Mono task head, so local admission is allowed. Not a provider!
    second,budget2=seed(api,tmp_path/"runner-2.sqlite3",item)
    try:
        assert claim(api,second,budget2,item).lease.holder=="untrusted-holder"
        assert len(second.unresolved_attempts())==1
        assert not runtime.tasks.read(opened["task_id"]).closed
    finally:
        second.close()
