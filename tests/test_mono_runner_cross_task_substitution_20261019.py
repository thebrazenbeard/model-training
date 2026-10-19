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
        subject="source-only", assignee_ref="TEST_ACTOR",
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
