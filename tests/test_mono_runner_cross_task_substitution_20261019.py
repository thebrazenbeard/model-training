"""Adverse exact-source cross-task delegation and claimed issuer substitution.

LOCAL Runner admission is deliberately not interpreted as Mono task authority.
Requires pinned Mono #77 / PORTAL #62 via inherited api fixture.
"""
from __future__ import annotations

from test_mono_runner_untrusted_producer_v1 import api, source, work, seed, claim


def test_cross_task_delegation_substitution_remains_local_only(api, tmp_path):
    runtime, opened_a, delegation_a = source(api, tmp_path)
    opened_b = api["open_message"](runtime.tasks, "untrusted", "B", "literal B")
    delegated_b = runtime.delegate_task_work(
        opened_b["task_id"], "read",
        repository="thebrazenbeard/project-runner", ref="main",
        subject="source-only-independent-B", assignee_ref="TEST_ACTOR",
        allowed_effects=(), prohibited_effects=("protected",),
        return_shape=("receipt",), evidence_refs=("fixture",),
    )
    delegation_b = delegated_b.delegation_ref("read")
    assert opened_a["task_id"] != opened_b["task_id"]
    assert delegation_a.binding_event_digest != delegation_b.binding_event_digest

    # A WorkUnit is constructed with A's claimed task/source but B's
    # authentic task-delegation reference. The Runner does not receive an
    # authenticated A->WorkUnit association.
    substituted = work(
        api, opened_a, delegation_b,
        claimed_issuer="forged-receiver",
        claimed_nonce="COUNTERFEIT_SHARED_NONCE",
        claimed_model_sha256="f" * 64,
    )
    assert substituted.payload["claimed_task_id"] == opened_a["task_id"]
    assert substituted.payload["claimed_event_head"] == delegation_b.binding_event_digest
    store, budget = seed(api, tmp_path / "substituted-runner.db", substituted)
    try:
        local = claim(api, store, budget, substituted)
        assert local.lease.holder == "untrusted-holder"
        assert len(store.unresolved_attempts()) == 1
        assert not runtime.tasks.read(opened_a["task_id"]).closed
        assert not runtime.tasks.read(opened_b["task_id"]).closed
        assert runtime.pc_execution_transport is None
        assert not runtime.provider_execution_transports
    finally:
        store.close()


def test_counterfeit_latest_head_and_receiver_claim_remain_local(api, tmp_path):
    runtime, opened, delegation = source(api, tmp_path)
    counterfeit = work(
        api, opened, delegation,
        claimed_event_head="0" * 64,
        claimed_issuer="receiver-owned-not-verified",
        claimed_host_generation="spoofed-generation",
        claimed_session="spoofed-process-session",
        claimed_nonce="SPOOFED_NONCE",
    )
    assert counterfeit.payload["claimed_event_head"] != delegation.binding_event_digest
    store, budget = seed(api, tmp_path / "counterfeit-runner.db", counterfeit)
    try:
        local = claim(api, store, budget, counterfeit)
        assert local.lease.holder == "untrusted-holder"
        assert len(store.unresolved_attempts()) == 1
        assert runtime.tasks.events(opened["task_id"])[-1].event_digest == delegation.binding_event_digest
        assert not runtime.tasks.read(opened["task_id"]).closed
        assert runtime.pc_execution_transport is None
        assert not runtime.provider_execution_transports
    finally:
        store.close()


def test_consistent_proposal_stale_cold_A_then_stripped_rehash_is_local_only(api, tmp_path):
    """Real source association can HOLD, go stale, then be stripped into LOCAL claim."""
    import importlib.util
    import os
    from dataclasses import replace
    from pathlib import Path
    import pytest
    from test_mono_runner_post_claim_task_drift_v1 import cold_checkpoint_A

    runtime, opened, delegation = source(api, tmp_path)
    item = work(
        api, opened, delegation,
        claimed_task_id=opened["task_id"],
        claimed_event_head=delegation.binding_event_digest,
        claimed_source_digest=opened["source_digest"],
        claimed_issuer="receiver-selected-CLAIM-NOT-AUTHENTICATED",
        claimed_nonce="scoped-on-paper-only",
        claimed_host_generation="host-generation-claimed",
        claimed_session="loaded-process-claimed",
        claimed_model_sha256="c" * 64,
        claimed_route="selected-model-claimed",
    )
    path = Path(os.environ["TRAINING_PORTAL_ROOT"]) / "tests" / "proposals" / "mono_runner_producer_contract.py"
    spec = importlib.util.spec_from_file_location("training_portal_proposal", path)
    assert spec is not None and spec.loader is not None
    proposal_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proposal_api)
    kwargs = dict(runtime=runtime, delegation=delegation,
                  actor_ref="TEST_ACTOR", expected_task_head=delegation.binding_event_digest)
    proposed = proposal_api.propose(**kwargs, work=item)
    assert proposed["association"]["task_id"] == opened["task_id"]
    assert proposed["association"]["expected_task_head"] == delegation.binding_event_digest
    assert proposed["association"]["opened_lifecycle_evidence_digest"] == opened["source_digest"]
    assert proposed["association"]["producer_authenticated"] is False
    assert proposed["fingerprint"] == api["work_unit_fingerprint"](proposed["work"])
    first = proposal_api.revalidate(**kwargs, proposal=proposed)
    assert first["state"] == "HOLD" and first["producer_authenticated"] is False
    assert first["dispatch_authorized"] is False

    global_before = runtime.tasks.verify_chain()
    api["open_message"](runtime.tasks, "untrusted", "B", "independent B")
    assert runtime.tasks.verify_chain() != global_before
    assert runtime.tasks.events(opened["task_id"])[-1].event_digest == delegation.binding_event_digest
    assert proposal_api.revalidate(**kwargs, proposal=proposed)["state"] == "HOLD"
    advanced = cold_checkpoint_A(tmp_path, opened, delegation)
    print(f"TRAINING_COLD_PROBE parent_pid={os.getpid()} child_pid={advanced['pid']} "
          f"old_task_head={delegation.binding_event_digest} new_task_head={advanced['task_event_head']}")
    assert advanced["task_event_head"] != delegation.binding_event_digest
    # Canonical task-local CAS protects writes; the copied delegation actor label
    # still validates, so delegation validation ALONE is not a freshness fence.
    runtime.validate_task_delegation(delegation, actor_ref="TEST_ACTOR")
    with pytest.raises(ValueError, match="stale task-scoped producer snapshot"):
        proposal_api.revalidate(**kwargs, proposal=proposed)

    marked = proposed["work"]
    marked_store, marked_budget = seed(api, tmp_path / "marked.db", marked)
    try:
        with pytest.raises(ValueError, match="unapproved Mono producer"):
            claim(api, marked_store, marked_budget, marked)
        assert len(marked_store.unresolved_attempts()) == 0
    finally:
        marked_store.close()
    stripped = replace(marked, payload={k: v for k, v in marked.payload.items()
                                        if k != proposal_api.KEY})
    assert api["work_unit_fingerprint"](stripped) != proposed["fingerprint"]
    local_store, local_budget = seed(api, tmp_path / "stripped.db", stripped)
    try:
        local = claim(api, local_store, local_budget, stripped)
        assert local.lease.holder == "untrusted-holder"
        assert len(local_store.unresolved_attempts()) == 1
        with pytest.raises(ValueError):
            claim(api, local_store, local_budget, stripped)
        assert len(local_store.unresolved_attempts()) == 1
        assert not runtime.tasks.read(opened["task_id"]).closed
        from vera_core.local_conversation import recover_conversation
        history = recover_conversation(runtime.tasks, None, "untrusted")
        assert len(history) == 2
        assert all(not entry["memory_admitted"] and not entry["accepted_as_task_result"]
                   and not entry["display_authorized"] for entry in history)
        assert runtime.pc_execution_transport is None
        assert not runtime.provider_execution_transports
    finally:
        local_store.close()
