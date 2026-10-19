"""Adverse self-consistent task/WorkUnit association is not issuer authority.

Pinned by the inherited api fixture to exact Mono#77 and PORTAL#62.
The unmarked WorkUnit is intentionally a LOCAL positive-control claim.
"""
from __future__ import annotations
import pytest
from test_mono_runner_untrusted_producer_v1 import (
    api, source, work, seed, claim, UNAPPROVED,
)

def test_consistent_task_work_fingerprint_marker_strip_and_stale_replay(api,tmp_path):
    runtime, opened, delegation = source(api,tmp_path)
    task_id = opened["task_id"]
    original_head = delegation.binding_event_digest
    assert runtime.tasks.events(task_id)[-1].event_digest == original_head
    # A legitimate unrelated task B changes the global head, not A's CAS.
    global_before = runtime.tasks.verify_chain()
    api["open_message"](runtime.tasks,"untrusted","B","independent B")
    assert runtime.tasks.verify_chain() != global_before
    assert runtime.tasks.events(task_id)[-1].event_digest == original_head

    # This is deliberately internally consistent and recomputed, but is
    # caller-made and no independent receiver has issued its authority.
    candidate = work(api,opened,delegation,
        claimed_issuer="selected-receiver",claimed_nonce="supposed-consumed",
        claimed_route="selected-route",claimed_session="selected-session",
        claimed_host_generation="selected-host",
        claimed_model_sha256="a"*64,
        claimed_work_fingerprint="caller-computed")
    marked = work(api,opened,delegation,**{
        UNAPPROVED:{"producer_authenticated":False}
    })
    assert UNAPPROVED in marked.payload
    assert UNAPPROVED not in candidate.payload
    assert api["work_unit_fingerprint"](marked) != api["work_unit_fingerprint"](candidate)
    blocked,budget=seed(api,tmp_path/"marked.db",marked)
    try:
        before=blocked.connection.total_changes
        with pytest.raises(ValueError,match="unapproved Mono producer"):
            claim(api,blocked,budget,marked)
        assert blocked.connection.total_changes==before
        assert not blocked.unresolved_attempts()
    finally: blocked.close()

    # Marker stripping plus a new correct immutable fingerprint can be
    # admitted by the LOCAL Runner, but does not authenticate Mono producer.
    store,budget=seed(api,tmp_path/"local.db",candidate)
    try:
        accepted=claim(api,store,budget,candidate)
        assert accepted.lease.holder=="untrusted-holder"
        assert len(store.unresolved_attempts())==1
        assert runtime.tasks.events(task_id)[-1].event_digest==original_head
        assert runtime.pc_execution_transport is None
        assert not runtime.provider_execution_transports
        assert not runtime.tasks.read(task_id).closed
    finally: store.close()

    # Independent same-task writer advances canonical A after LOCAL claim;
    # no optimistic acceptance of its previous head or redispatch.
    runtime.tasks.checkpoint(task_id,"A-advanced",
        completed_evidence=(),blockers=("NO_PROVIDER_RESULT",),
        protected_effects_still_gated=("PROVIDER_EFFECT",),
        next_frontier="UNKNOWN_NO_REDISPATCH",
        lifecycle_evidence_digest=opened["source_digest"],
        expected_task_head=original_head)
    assert runtime.tasks.events(task_id)[-1].event_digest!=original_head
    cold=api["SqliteDispatchAdmissionStore"](tmp_path/"local.db")
    try:
        assert len(cold.unresolved_attempts())==1
        with pytest.raises(ValueError):
            claim(api,cold,budget,candidate)
        assert len(cold.unresolved_attempts())==1
        assert not runtime.tasks.read(task_id).closed
    finally: cold.close()

def test_portal63_inspected_local_claim_not_authenticated(api,tmp_path):
    """Exercise the actual PORTAL #63 selected-or-HOLD candidate."""
    import importlib.util
    import os
    from pathlib import Path
    portal = Path(os.environ["TRAINING_PORTAL_ROOT"])
    spec=importlib.util.spec_from_file_location(
        "portal63_mono_proposal",
        portal/"tests"/"proposals"/"mono_runner_producer_contract.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime,opened,delegation=source(api,tmp_path)
    native=work(api,opened,delegation,claimed_issuer="fake-selected",
        claimed_nonce="fake-consumed",claimed_model_sha256="f"*64)
    proposal=module.propose(runtime=runtime,delegation=delegation,
        actor_ref="TEST_ACTOR",
        expected_task_head=delegation.binding_event_digest,work=native)
    receipt=module.inspect_observed_local_work(
        runtime=runtime,delegation=delegation,actor_ref="TEST_ACTOR",
        expected_task_head=delegation.binding_event_digest,
        proposal=proposal,observed_work=proposal["work"])
    assert receipt["source_proposal_revalidated"] is True
    assert receipt["observed_work_matches_proposal"] is True
    assert receipt["state"]=="HOLD"
    assert receipt["producer_authenticated"] is False
    assert receipt["dispatch_authorized"] is False
    assert receipt["memory_admitted"] is False
    assert receipt["effect"]=="NO_EFFECT"
    stripped=work(api,opened,delegation,claimed_issuer="fake-selected",
        claimed_nonce="fake-consumed",claimed_model_sha256="f"*64)
    untrusted=module.inspect_observed_local_work(
        runtime=runtime,delegation=delegation,actor_ref="TEST_ACTOR",
        expected_task_head=delegation.binding_event_digest,
        proposal=proposal,observed_work=stripped)
    assert untrusted["observed_work_matches_proposal"] is False
    assert untrusted["producer_authenticated"] is False
    store,budget=seed(api,tmp_path/"unmarked-local.db",stripped)
    try:
        assert claim(api,store,budget,stripped).lease.holder=="untrusted-holder"
    finally:store.close()
    runtime.tasks.checkpoint(opened["task_id"],"stale-A",completed_evidence=(),
        blockers=("NO_PROVIDER_RESULT",),protected_effects_still_gated=("PROVIDER_EFFECT",),
        next_frontier="UNKNOWN_NO_REDISPATCH",
        lifecycle_evidence_digest=opened["source_digest"],
        expected_task_head=delegation.binding_event_digest)
    stale=module.inspect_observed_local_work(
        runtime=runtime,delegation=delegation,actor_ref="TEST_ACTOR",
        expected_task_head=delegation.binding_event_digest,
        proposal=proposal,observed_work=proposal["work"])
    assert stale["source_proposal_revalidated"] is False
    assert stale["state"]=="HOLD"
    assert stale["producer_authenticated"] is False
