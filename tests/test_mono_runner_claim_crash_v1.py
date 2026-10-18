"""Cold-process crash boundary of LOCAL Runner claim, not a Mono producer grant.

Requires the same pinned exact Mono and PORTAL checkout as the accompanying
test_mono_runner_untrusted_producer_v1.py. No backend or protected effect runs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_mono_runner_untrusted_producer_v1 import api, source, work, seed, claim

CHILD = r"""
import os,sys
from runner.durable_dispatch import SqliteDispatchAdmissionStore
db,fp,phase = sys.argv[1:]
if phase == "after":
    store=SqliteDispatchAdmissionStore(db)
    store.admit(lineage_id="fixture",work_fingerprint_value=fp,budget_scope_id="root",
        expected_budget_generation=1,expected_work_generation=1,
        holder="untrusted-holder",now=100.0,ttl=20.0)
    store.close()
os._exit(73)
"""

def cold(db, fingerprint, phase):
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1",
         "PYTHONPATH":os.pathsep.join(sys.path)}
    proc=subprocess.run([sys.executable,"-B","-c",CHILD,str(db),fingerprint,phase],
         text=True,capture_output=True,timeout=20,env=env,check=False)
    assert proc.returncode == 73, proc.stderr
    return proc.returncode


def test_crash_before_local_claim_does_not_consume_runner_attempt(api,tmp_path):
    runtime,a,ref=source(api,tmp_path)
    item=work(api,a,ref)
    db=tmp_path/"runner-before.sqlite3"
    store,budget=seed(api,db,item)
    try:
        assert cold(db,api["work_unit_fingerprint"](item),"before")==73
        assert store.unresolved_attempts() == ()
        assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
        assert not runtime.tasks.read(a["task_id"]).closed
    finally: store.close()


def test_crash_after_local_claim_preserves_unresolved_hold(api,tmp_path):
    runtime,a,ref=source(api,tmp_path)
    item=work(api,a,ref)
    db=tmp_path/"runner-after.sqlite3"
    store,budget=seed(api,db,item)
    try:
        assert cold(db,api["work_unit_fingerprint"](item),"after")==73
        assert len(store.unresolved_attempts()) == 1
        with pytest.raises(ValueError):
            claim(api,store,budget,item)
        assert len(store.unresolved_attempts()) == 1
        assert not runtime.tasks.read(a["task_id"]).closed
        # This is still NO observed selected Mono producer, process-loaded
        # model, authenticated issuer, or verified provider RESULT.
    finally: store.close()
