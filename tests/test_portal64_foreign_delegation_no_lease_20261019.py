"""Hostile #64 current B delegation with caller-claimed task A.

Current canonical B/source and matching immutable fingerprint do not select
claimant A, target commit, receiver issuer, nonce or actual Runner lease.
Physical effects forbidden: disposable source ledgers only.
"""
from __future__ import annotations
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest
from test_mono_runner_untrusted_producer_v1 import api, source, work, seed, claim

PORTAL64 = "a12f67251e34e810de90680ecb809e96ad356a1e"


@pytest.fixture(scope="module")
def portal64(api):
    root = Path(os.environ["TRAINING_PORTAL64_ROOT"])
    assert subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip() == PORTAL64
    spec = importlib.util.spec_from_file_location(
        "training_portal64_foreign",
        root / "tests" / "proposals" / "mono_runner_producer_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_foreign_B_current_proposal_claims_A_and_matches_without_lease(api, portal64, tmp_path):
    runtime, opened_a, delegation_a = source(api, tmp_path)
    opened_b = api["open_message"](runtime.tasks, "untrusted", "B", "literal-B")
    state_b = runtime.delegate_task_work(
        opened_b["task_id"], "read",
        repository=delegation_a.repository, ref=delegation_a.ref,
        subject="foreign-B-independent-scope", assignee_ref="TEST_ACTOR",
        allowed_effects=(), prohibited_effects=("protected",),
        return_shape=("receipt",), evidence_refs=("fixture",),
    )
    delegation_b = state_b.delegation_ref("read")
    assert delegation_a.task_id != delegation_b.task_id
    assert runtime.tasks.events(opened_a["task_id"])[-1].event_digest == delegation_a.binding_event_digest
    assert runtime.tasks.events(opened_b["task_id"])[-1].event_digest == delegation_b.binding_event_digest
    # Public payload claims valid A source, while supplied live ref belongs to B.
    item = work(
        api, opened_a, delegation_b,
        claimed_event_head=delegation_a.binding_event_digest,
        claimed_issuer="claimed-selected-receiver",
        claimed_nonce="claimed-consumed-nonce",
        claimed_host_generation="claimed-active-host",
        claimed_session="claimed-model-process",
        claimed_route="claimed-selected-route",
        claimed_model_sha256="f" * 64,
        claimed_target_commit_authorized=True,
    )
    assert item.payload["claimed_task_id"] == opened_a["task_id"]
    assert item.payload["claimed_event_head"] == delegation_a.binding_event_digest
    assert item.inputs[0].commit == "a" * 40
    params = dict(
        runtime=runtime, delegation=delegation_b, actor_ref="TEST_ACTOR",
        expected_task_head=delegation_b.binding_event_digest,
    )
    proposal = portal64.propose(**params, work=item)
    assert proposal["association"]["task_id"] == opened_b["task_id"]
    assert proposal["association"]["task_id"] != item.payload["claimed_task_id"]
    assert proposal["fingerprint"] == api["work_unit_fingerprint"](proposal["work"])
    absent = tmp_path / "NO_RUNNER_LEASE_WAS_CREATED.sqlite3"
    assert not absent.exists()
    inspected = portal64.inspect_observed_local_work(
        **params, proposal=proposal, observed_work=proposal["work"]
    )
    assert not absent.exists()
    # STRONGER CONTRACT RED FIRST: a future *selected* producer would refuse
    # this mismatch. This source-only helper can currently revalidate B.
    assert inspected["source_proposal_revalidated"] is False
    assert inspected["observed_work_matches_proposal"] is True
    assert inspected["target_commit_authorized"] is False
    assert all(inspected[key] is False for key in (
        "producer_authenticated", "dispatch_authorized", "memory_admitted",
    ))
    assert inspected["state"] == "HOLD" and inspected["effect"] == "NO_EFFECT"

    # Changing A does not change the genuinely bound B; no derived authority.
    runtime.tasks.checkpoint(opened_a["task_id"], "A cold-evidence-less",
        completed_evidence=(), blockers=("NO_RESULT",),
        protected_effects_still_gated=("PROVIDER_EFFECT",),
        next_frontier="UNKNOWN_NO_REDISPATCH",
        lifecycle_evidence_digest=opened_a["source_digest"],
        expected_task_head=delegation_a.binding_event_digest)
    assert portal64.inspect_observed_local_work(
        **params, proposal=proposal, observed_work=proposal["work"]
    )["source_proposal_revalidated"] is True

    # Stripping the producer marker and rehashing produces LOCAL claims
    # independently at each SQLite root, not a shared receiver nonce.
    stripped = replace(proposal["work"], payload={
        key: value for key, value in proposal["work"].payload.items()
        if key != portal64.KEY
    })
    assert api["work_unit_fingerprint"](stripped) != proposal["fingerprint"]
    for index in (1, 2):
        store, budget = seed(api, tmp_path / f"independent-runner-{index}.sqlite3", stripped)
        try:
            assert claim(api, store, budget, stripped).lease.holder == "untrusted-holder"
            assert len(store.unresolved_attempts()) == 1
            with pytest.raises(ValueError):
                claim(api, store, budget, stripped)
            assert len(store.unresolved_attempts()) == 1
        finally:
            store.close()
    assert not runtime.tasks.read(opened_a["task_id"]).closed
    assert not runtime.tasks.read(opened_b["task_id"]).closed
    assert runtime.pc_execution_transport is None
    assert not runtime.provider_execution_transports

    runtime.tasks.checkpoint(opened_b["task_id"], "B advanced",
        completed_evidence=(), blockers=("NO_RESULT",),
        protected_effects_still_gated=("PROVIDER_EFFECT",),
        next_frontier="UNKNOWN_NO_REDISPATCH",
        lifecycle_evidence_digest=opened_b["source_digest"],
        expected_task_head=delegation_b.binding_event_digest)
    stale = portal64.inspect_observed_local_work(
        **params, proposal=proposal, observed_work=proposal["work"]
    )
    assert stale["source_proposal_revalidated"] is False
    assert stale["target_commit_authorized"] is False
    assert stale["state"] == "HOLD"
