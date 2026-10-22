"""Training source-only producer substitution tests; no provider effect or trust grant.

Requires detached source roots TRAINING_MONO_ROOT at Mono#77 8fd64c97... and
TRAINING_PORTAL_ROOT at PORTAL#62 6ef585f3...; SKIP unless both exact heads.
All ledgers/Runner databases are disposable pytest tmp_path fixtures.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import pytest

MONO = "8fd64c97998a8d24943439a96409f2114646905e"
PORTAL = "6ef585f36194a538a01b751d0802a9e876815f0a"
UNAPPROVED = "UNAPPROVED_MONO_PRODUCER_ASSOCIATION_V1"

@pytest.fixture(scope="module")
def api():
    for env, expected in (("TRAINING_MONO_ROOT", MONO), ("TRAINING_PORTAL_ROOT", PORTAL)):
        root = os.environ.get(env)
        if not root or not Path(root).is_dir():
            pytest.skip("EXACT_SOURCE_ROOT_UNAVAILABLE:" + env)
        try:
            head = subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True,timeout=8).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("SOURCE_HEAD_UNVERIFIED:" + env)
        if head != expected:
            pytest.skip("SOURCE_HEAD_MISMATCH:" + env + ":" + head)
    for src in sorted((Path(os.environ["TRAINING_MONO_ROOT"]) / "packages").glob("*/src")):
        sys.path.insert(0,str(src))
    sys.path.insert(0,os.environ["TRAINING_PORTAL_ROOT"])
    from vera_core import QualifiedVeraRuntime, VeraStateDirectory
    from vera_core.local_conversation import open_message
    from runner.models import ExactSubject
    from runner.work_units import WorkUnit, WorkUnitStatus, work_unit_fingerprint
    from runner.budgets import BudgetEnvelope
    from runner.durable_dispatch import SqliteDispatchAdmissionStore
    return locals()

def source(api,tmp):
    runtime = api["QualifiedVeraRuntime"].from_state_directory(
        api["VeraStateDirectory"](tmp/"mono",project_id="TRAINING_SOURCE_PROBE",identity_id="TEST_ONLY"))
    opened = api["open_message"](runtime.tasks,"untrusted","A","literal A")
    delegated = runtime.delegate_task_work(opened["task_id"],"read",repository="thebrazenbeard/project-runner",
        ref="main",subject="source-only",assignee_ref="TEST_ACTOR",allowed_effects=(),
        prohibited_effects=("protected",),return_shape=("receipt",),evidence_refs=("fixture",))
    return runtime,opened,delegated.delegation_ref("read")

def work(api, opened, delegation, **changes):
    payload={"execution_authority":False,"protected_effects_authorized":False,
        "claimed_task_id":opened["task_id"],"claimed_event_head":delegation.binding_event_digest,
        "claimed_source_digest":opened["source_digest"],"claimed_issuer":"caller-text",
        "claimed_model_sha256":"9"*64,"claimed_host_generation":"caller-generation",
        "claimed_session":"caller-session","claimed_nonce":"ONE_NONCE","claimed_route":"caller-route"}
    payload.update(changes)
    return api["WorkUnit"](id="untrusted-work",root_frontier_id="test-only",parent_work_id=None,
        inputs=(api["ExactSubject"](repository=delegation.repository,ref=delegation.ref,commit="a"*40),),
        operation="PORTFOLIO_BOUND_CLAIM",required_capabilities=("read",),
        collision_keys=("source:one",),recursion_depth=0,
        budget_allocation={"children":0,"active":1,"retries":0,"backend_jobs":1},
        expected_outputs=("receipt",),completion_criteria=("NO_EFFECT",),
        status=api["WorkUnitStatus"].PENDING,payload=payload)

def seed(api,db,item):
    store=api["SqliteDispatchAdmissionStore"](db)
    budget=api["BudgetEnvelope"](lineage_id="fixture",max_depth=0,depth=0,
        remaining_children=0,remaining_active=1,remaining_retries=0,remaining_backend_jobs=1)
    assert store.initialize_root(budget=budget,work=item,effective_capabilities=("read",))==(1,1)
    return store,budget

def claim(api,store,budget,item):
    return store.admit(lineage_id=budget.lineage_id,
        work_fingerprint_value=api["work_unit_fingerprint"](item),budget_scope_id=budget.scope_id,
        expected_budget_generation=1,expected_work_generation=1,
        holder="untrusted-holder",now=100.0,ttl=20.0)

def test_unmarked_forged_claim_is_local_not_authenticated_mono(api,tmp_path):
    rt,a,ref=source(api,tmp_path)
    item=work(api,a,ref,claimed_issuer="forged",claimed_model_sha256="0"*64)
    assert UNAPPROVED not in item.payload
    store,budget=seed(api,tmp_path/"runner.db",item)
    try:
        assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
        assert len(store.unresolved_attempts())==1
        assert not rt.tasks.read(a["task_id"]).closed
        assert rt.pc_execution_transport is None and not rt.provider_execution_transports
    finally: store.close()

def test_stale_original_mono_A_not_checked_by_unmarked_local_claim(api,tmp_path):
    rt,a,ref=source(api,tmp_path)
    item=work(api,a,ref)
    rt.tasks.checkpoint(a["task_id"],"advance-A",completed_evidence=(),blockers=("HOLD",),
        protected_effects_still_gated=("PROVIDER",),next_frontier="HOLD",
        lifecycle_evidence_digest=a["source_digest"],expected_task_head=ref.binding_event_digest)
    assert rt.tasks.events(a["task_id"])[-1].event_digest != ref.binding_event_digest
    store,budget=seed(api,tmp_path/"runner.db",item)
    try:
        assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
        assert not rt.tasks.read(a["task_id"]).closed
    finally: store.close()

def test_unrelated_B_global_head_change_preserves_A_event_head(api,tmp_path):
    rt,a,ref=source(api,tmp_path)
    global_before=rt.tasks.verify_chain()
    api["open_message"](rt.tasks,"untrusted","B","independent B")
    assert rt.tasks.verify_chain()!=global_before
    assert rt.tasks.events(a["task_id"])[-1].event_digest==ref.binding_event_digest
    item=work(api,a,ref)
    store,budget=seed(api,tmp_path/"runner.db",item)
    try: assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
    finally: store.close()

def test_explicitly_unapproved_proposal_rejected_before_write(api,tmp_path):
    _,a,ref=source(api,tmp_path)
    item=work(api,a,ref,**{UNAPPROVED:{"producer_authenticated":False}})
    store,budget=seed(api,tmp_path/"runner.db",item)
    try:
        def snapshot():
            return {n:store.connection.execute('SELECT * FROM "'+n+'" ORDER BY rowid').fetchall()
                for n, in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")}
        before=snapshot()
        with pytest.raises(ValueError,match="unapproved Mono producer"):
            claim(api,store,budget,item)
        assert snapshot()==before
    finally: store.close()

def test_separate_runner_roots_accept_same_unverified_nonce(api,tmp_path):
    _,a,ref=source(api,tmp_path)
    item=work(api,a,ref)
    for suffix in ("one","two"):
        store,budget=seed(api,tmp_path/(suffix+".db"),item)
        try:
            assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
        finally: store.close()

def test_caller_identity_changes_fingerprint_without_authentication(api,tmp_path):
    _,a,ref=source(api,tmp_path)
    old=work(api,a,ref)
    changed=work(api,a,ref,claimed_issuer="another",claimed_session="rotated",
                 claimed_host_generation="forged",claimed_route="another")
    assert api["work_unit_fingerprint"](old)!=api["work_unit_fingerprint"](changed)
    assert UNAPPROVED not in changed.payload
    for index,item in enumerate((old,changed)):
        store,budget=seed(api,tmp_path/(str(index)+".db"),item)
        try: assert claim(api,store,budget,item).lease.holder=="untrusted-holder"
        finally: store.close()
