"""Source-only conformance against real Mono task and prior frozen Training probes.

Neither this evidence compiler nor any synthetic witness is a provider, trust,
nonce or effect authority. All SQLite roots are temporary/disposable.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter_ns
import json
import secrets

from vera_core.task_execution import TaskExecutionLedger
from evaluate_action_lock_fence_v1 import evaluate as action_lock_fence
from evaluate_checkpoint_poison_v1 import evaluate as checkpoint_poison
from evaluate_receiver_nonce_lock_v1 import evaluate as receiver_scope_probe
from evaluate_receiver_bound_preflight_v1 import (
    evaluate as synthetic_preflight_probe, original_task, make_receipt,
    witness_of, preflight, mac,
)

SCHEMA = "TRAINING_RECEIVER_ATOMICITY_ADVERSE_MATRIX_V1"
FROZEN_CASES = (
    "ordinary_checkpoint_can_poison_public_nonce_marker",
    "own_task_checkpoint_after_snapshot_stale_accept",
    "unrelated_B_only_changes_global_head",
    "same_receiver_nonce_replay_across_tasks_and_cold_process",
    "independent_SQLite_roots_share_no_nonce_domain",
    "caller_forges_generation_session_and_model_witness",
    "precommit_and_postcommit_crash_cut",
    "action_lock_does_not_fence_native_task_checkpoint",
)
def _sha(x):
    return sha256(json.dumps(x, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def actual_preflight_race():
    with TemporaryDirectory(prefix="training-authority-task-race-") as td:
        ledger = TaskExecutionLedger(Path(td) / "mono.sqlite3")
        a = original_task(ledger)
        key = secrets.token_bytes(32)
        receipt = make_receipt(a)
        witness = witness_of(receipt)
        ready, done = Event(), Event()
        errors = []
        class Snapshot:
            def read(self, task_id):
                return ledger.read(task_id)
            def events(self, task_id):
                values = ledger.events(task_id)
                ready.set()
                if not done.wait(5):
                    raise AssertionError("actual checkpoint writer timeout")
                return values
        def writer():
            try:
                if not ready.wait(5):
                    raise AssertionError("source snapshot not reached")
                ledger.checkpoint(
                    a["task_id"], "actual-source-advance",
                    completed_evidence=("test-only",),
                    blockers=("NO_PROVIDER",),
                    protected_effects_still_gated=("PROVIDER",),
                    next_frontier="source-only task advance",
                    lifecycle_evidence_digest="3" * 64,
                    expected_task_head=a["opened_head"])
            except Exception as exc:
                errors.append(str(exc))
            finally:
                done.set()
        thread = Thread(target=writer, name="real-sqlite-writer")
        thread.start()
        try:
            result = preflight(receipt, mac(key, receipt), key, witness,
                               Snapshot(), a, set())
        finally:
            thread.join(5)
        assert not thread.is_alive() and not errors, errors
        return {"unsafe_stale_preflight_accepted": result,
                "actual_A_events": len(ledger.events(a["task_id"])),
                "canonical_task_closed": ledger.read(a["task_id"]).closed}


def forged_synthetic_witness():
    with TemporaryDirectory(prefix="training-forged-witness-") as td:
        ledger = TaskExecutionLedger(Path(td) / "mono.sqlite3")
        original = original_task(ledger)
        attacker_key = secrets.token_bytes(32)
        forged = make_receipt(original)
        forged.update({
            "provider_id": "attacker-claims-provider",
            "model_id": "attacker-model",
            "host_id": "attacker-host",
            "host_generation": "attacker-generation",
            "session_id": "attacker-session",
            "loaded_artifact_sha256": "9" * 64,
            "route_id": "attacker-route",
            "selected_routes": ["attacker-route"],
            "unique_durable_route_selection": "attacker-route",
        })
        # The entire fixture key and synthetic receiver witness is author-owned.
        # Success here is deliberately a RED against "field equality == trust".
        accepted = preflight(
            forged, mac(attacker_key, forged), attacker_key, witness_of(forged),
            ledger, original, set())
        return {"coherent_forged_fixture_passes_preflight": accepted,
                "real_issuer_authenticated": False,
                "real_loaded_model_observed": False,
                "canonical_task_closed": ledger.read(original["task_id"]).closed}


def evaluate():
    started = perf_counter_ns()
    baseline = {}
    for label, runner in (
        ("nonce_scope", receiver_scope_probe),
        ("poison", checkpoint_poison),
        ("lock", action_lock_fence),
        ("preflight", synthetic_preflight_probe),
    ):
        tic = perf_counter_ns()
        baseline[label] = runner()
        baseline[label]["orchestrator_elapsed_ns"] = perf_counter_ns()-tic
    race = actual_preflight_race()
    forged = forged_synthetic_witness()
    p, n, l, r = (baseline[x] for x in ("poison", "nonce_scope", "lock", "preflight"))
    records = [
        ("counterfeit_checkpoint", "REJECT_CHECKPOINT_AUTHORITY",
         p["counterfeit_marker_blocks_legitimate_first"]),
        ("stale_A_during_preflight", "UNSAFE_STALE_ACCEPT",
         race["unsafe_stale_preflight_accepted"] and race["actual_A_events"] == 2),
        ("unrelated_B_current_A", "SOURCE_TASK_SCOPE_CORRECT",
         l["task_head_unchanged_by_unrelated_B"] and l["global_head_changed_by_unrelated_B"]),
        ("cold_and_cross_task_same_nonce", "REPLAY_GAP",
         p["foreign_root_same_nonce_accepted"] and
         (not n["cold_replay"]["accepted"]) and
         (not n["different_task_replay"]["accepted"])),
        ("different_SQLite_roots", "NO_SHARED_COLLISION_OWNER",
         p["foreign_root_same_nonce_accepted"]),
        ("forged_generation_and_loaded_model", "UNAUTHENTICATED_FIXTURE",
         n["forged_generation"]["accepted"] and
         forged["coherent_forged_fixture_passes_preflight"]),
        ("crash_cut_pre_post", "POSTCOMMIT_UNKNOWN_HOLD",
         p["after_precommit_crash_accepted"] and
         p["after_postcommit_crash_rejected"]),
        ("action_lock_nonfence", "NO_ALL_WRITER_LOCK",
         l["cold_writer_accepted_while_action_lock_held"] and
         l["writer_checkpoint_visible_before_unlock"]),
    ]
    return {
        "schema": SCHEMA,
        "case_sha256": _sha(FROZEN_CASES),
        "training_parent": "cd7de17a9e97d6d752da90d200a7555a295205c2",
        "mono_ledger_blob": "b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de",
        "case_total": len(records),
        "case_results": [{"name":name, "classification":kind, "observed":bool(value)}
                         for name,kind,value in records],
        "all_adverse_mechanisms_observed": all(value for _,_,value in records),
        "old_synthetic_preflight_adverse_accepted": r["candidate_unsafe"],
        "old_synthetic_preflight_positives_retained": r["candidate_positive"],
        "old_synthetic_preflight_false_holds": r["candidate_false_hold"],
        "coherent_fixture_forgery": forged,
        "race": race,
        "source_probe_elapsed_ns": {name:run["orchestrator_elapsed_ns"]
                                    for name, run in baseline.items()},
        "real_issuer_verified": False,
        "real_process_loaded_model_verified": False,
        "selected_receiver_and_collision_domain_verified": False,
        "durable_host_cross_root_nonce_authority_verified": False,
        "provider_result_admitted": False,
        "memory_admitted": False,
        "protected_effect": False,
        "production_disposition": "HOLD_NO_TRUSTED_RECEIVER_ATOMIC_AUTHORITY",
        "elapsed_ns": perf_counter_ns()-started,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), sort_keys=True, indent=2))
