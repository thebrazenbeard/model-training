"""Adversarial exact-source PORTAL #63 selected-or-HOLD classifier review.

Reuses pinned Training #119/#120 Mono #77 and native PORTAL #62 fixtures.
#63 introduces only a test/proposal inspector, not a selected trusted receiver.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest
from test_mono_runner_untrusted_producer_v1 import api, source, work, seed, claim
from test_mono_runner_post_claim_task_drift_v1 import cold_checkpoint_A

PORTAL_CANDIDATE = "782e01052a05c94c6c5c7729d3b51c1db91975d2"


@pytest.fixture(scope="module")
def candidate(api):
    root = os.environ.get("TRAINING_PORTAL_CANDIDATE_ROOT")
    if not root or not Path(root).is_dir():
        pytest.skip("PORTAL_63_EXACT_CANDIDATE_ROOT_NOT_SET")
    head = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if head != PORTAL_CANDIDATE:
        pytest.skip("PORTAL_63_HEAD_MISMATCH:" + head)
    path = Path(root) / "tests" / "proposals" / "mono_runner_producer_contract.py"
    spec = importlib.util.spec_from_file_location("training_portal63_proposal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portal63_observed_local_claim_never_promotes_consistent_canonical_source(
    api, candidate, tmp_path
):
    runtime, opened, delegation = source(api, tmp_path)
    native = work(
        api, opened, delegation,
        claimed_task_id=opened["task_id"],
        claimed_source_digest=opened["source_digest"],
        claimed_event_head=delegation.binding_event_digest,
        claimed_issuer="receiver-claimed",
        claimed_nonce="nonce-claimed",
        claimed_host_generation="host-claimed",
        claimed_session="session-claimed",
        claimed_model_sha256="d" * 64,
        claimed_route="loaded-model-claimed",
    )
    arguments = dict(
        runtime=runtime, delegation=delegation, actor_ref="TEST_ACTOR",
        expected_task_head=delegation.binding_event_digest,
    )
    proposal = candidate.propose(**arguments, work=native)
    expected = candidate.inspect_observed_local_work(
        **arguments, proposal=proposal, observed_work=proposal["work"],
    )
    # Even exact work fingerprint and canonical current task do not authenticate.
    assert expected["source_proposal_revalidated"] is True
    assert expected["observed_work_matches_proposal"] is True
    assert expected["state"] == "HOLD"
    assert expected["producer_authenticated"] is False
    assert expected["dispatch_authorized"] is False
    assert expected["memory_admitted"] is False

    # Strip prohibited marker, rehash, and attach positive-sounding caller claims.
    unmarked_payload = {key: value for key, value in proposal["work"].payload.items()
                        if key != candidate.KEY}
    unmarked_payload.update({
        "producer_authenticated": True,
        "issuer": "FORGED_SELECTED_ISSUER",
        "receiver_nonce": "FORGED_RECEIVER_NONCE",
        "loaded_model": "SELF_ATTESTED_LOADED_MODEL",
    })
    observed = replace(proposal["work"], payload=unmarked_payload)
    assert api["work_unit_fingerprint"](observed) != proposal["fingerprint"]
    store, budget = seed(api, tmp_path / "observed-local.sqlite3", observed)
    try:
        local = claim(api, store, budget, observed)
        assert local.lease.holder == "untrusted-holder"
        attempts_before = store.unresolved_attempts()
        assert len(attempts_before) == 1

        inspect = lambda: candidate.inspect_observed_local_work(
            **arguments, proposal=proposal, observed_work=observed,
        )
        checked = inspect()
        assert checked["state"] == "HOLD" and checked["effect"] == "NO_EFFECT"
        assert checked["source_proposal_revalidated"] is True
        assert checked["observed_work_matches_proposal"] is False
        assert all(checked[flag] is False for flag in (
            "producer_authenticated", "dispatch_authorized", "memory_admitted"))
        assert "OBSERVED_WORK_NOT_SEALED_PROPOSAL" in checked["reasons"]
        assert store.unresolved_attempts() == attempts_before

        original_global = runtime.tasks.verify_chain()
        api["open_message"](runtime.tasks, "untrusted", "B", "unrelated task B")
        assert runtime.tasks.verify_chain() != original_global
        assert runtime.tasks.events(opened["task_id"])[-1].event_digest == delegation.binding_event_digest
        assert inspect()["source_proposal_revalidated"] is True

        cold = cold_checkpoint_A(tmp_path, opened, delegation)
        assert cold["pid"] != os.getpid()
        assert runtime.validate_task_delegation(delegation, actor_ref="TEST_ACTOR") == delegation
        stale = inspect()
        assert stale["state"] == "HOLD"
        assert stale["source_proposal_revalidated"] is False
        assert stale["observed_work_matches_proposal"] is False
        assert "TASK_SOURCE_UNVERIFIED" in stale["reasons"]
        assert all(stale[flag] is False for flag in (
            "producer_authenticated", "dispatch_authorized", "memory_admitted"))
        assert store.unresolved_attempts() == attempts_before
        with pytest.raises(ValueError):
            claim(api, store, budget, observed)
        assert store.unresolved_attempts() == attempts_before
        from vera_core.local_conversation import recover_conversation
        history = recover_conversation(runtime.tasks, None, "untrusted")
        assert len(history) == 2
        assert all(not entry["accepted_as_task_result"] and
                   not entry["memory_admitted"] and
                   not entry["display_authorized"] for entry in history)
        assert not runtime.tasks.read(opened["task_id"]).closed
        assert runtime.pc_execution_transport is None
        assert not runtime.provider_execution_transports
        print(f"PORTAL63_ADVERSARIAL_COLD parent_pid={os.getpid()} child_pid={cold['pid']} "
              f"local_attempt_count={len(attempts_before)} state={stale['state']}")
    finally:
        store.close()
