"""Disposable Mono task-journal poisoning falsifier. NOT provider/nonce code.

An ordinary caller allowed to append a task CHECKPOINT can create the same
publicly defined lifecycle digest and next_frontier used by Training #112's
research nonce scanner. The scanner cannot authenticate who created the marker.
Run only on temporary SQLite roots; never touch production task/memory state.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns

from vera_core.task_execution import TaskExecutionLedger
from evaluate_receiver_nonce_lock_v1 import TAG, args, identity, new_task, reserve

SCHEMA = "TRAINING_UNAUTHENTICATED_CHECKPOINT_POISON_V1"
CASES = (
    "clean_first_claim",
    "forged_marker_denies_first_claim",
    "wrong_nonce_does_not_deny",
    "wrong_receiver_scope_does_not_deny",
    "foreign_root_does_not_share_claim",
    "crash_before_marker_allows_first",
    "crash_after_marker_denies_first",
    "reused_nonce_with_forged_generation_accepted",
)


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def counterfeit_marker(path: Path, attacker: dict, claimed_digest: str) -> None:
    """Research-only attacker uses the PUBLIC existing Mono task checkpoint API."""
    ledger = TaskExecutionLedger(path)
    ledger.checkpoint(
        attacker["task_id"], "counterfeit-nonce-marker-" + claimed_digest,
        completed_evidence=("ordinary-task-not-provider",),
        blockers=("PROVIDER_NOT_AUTHENTICATED",),
        protected_effects_still_gated=("PROVIDER_EFFECT",),
        next_frontier=TAG,
        lifecycle_evidence_digest=claimed_digest,
        expected_task_head=attacker["task_event_head"],
    )


def crash_worker(payload: dict) -> None:
    # A real cold process exits without Python teardown or output. The phase
    # is test-only and the ledger file lives inside TemporaryDirectory.
    if payload["phase"] == "after":
        counterfeit_marker(Path(payload["path"]), payload["attacker"],
                           payload["claim_digest"])
    os._exit(17)


def crash(path: Path, attacker: dict, claim_digest: str, phase: str) -> int:
    child = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--die"],
        input=json.dumps({"path": str(path), "attacker": attacker,
                          "claim_digest": claim_digest, "phase": phase}),
        text=True, capture_output=True, timeout=12, check=False,
        env=os.environ.copy(),
    )
    if child.returncode != 17:
        raise RuntimeError(f"unexpected crash fixture status {child.returncode}: "
                           + child.stderr[-500:])
    return child.returncode


def setup(path: Path, suffix: str, nonce: str):
    ledger = TaskExecutionLedger(path)
    victim = new_task(ledger, "victim-" + suffix)
    attacker = new_task(ledger, "attacker-" + suffix)
    claim = args(path, victim, nonce)
    return ledger, victim, attacker, claim


def evaluate() -> dict:
    started = perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="training-checkpoint-poison-") as root:
        root = Path(root)
        a_path = root / "scope-one.sqlite3"
        a, victim, attacker, claim = setup(a_path, "poison", "nonce-P")
        counterfeit_marker(a_path, attacker, identity(claim))
        blocked = reserve(claim)
        source_after = a.read(victim["task_id"])
        marker = a.events(attacker["task_id"])[-1]
        # The attacker marker is itself a valid *task* checkpoint but contains
        # no authentic process-loaded model, issuer or admission event.
        marker_is_valid_task_history = a.verify_chain() == marker.event_digest
        marker_is_foreign_task = marker.task_id != victim["task_id"]

        clean_path = root / "clean.sqlite3"
        clean, c_v, c_a, clean_claim = setup(clean_path, "clean", "nonce-F")
        fresh = reserve(clean_claim)

        nonce_path = root / "different-nonce.sqlite3"
        _, _, n_attacker, n_claim = setup(nonce_path, "nonce", "nonce-A")
        counterfeit_marker(nonce_path, n_attacker, identity({**n_claim, "nonce": "other"}))
        wrong_nonce = reserve(n_claim)

        scope_path = root / "different-scope.sqlite3"
        _, _, s_attacker, s_claim = setup(scope_path, "scope", "nonce-B")
        counterfeit_marker(scope_path, s_attacker,
                           identity({**s_claim, "receiver_id": "foreign-receiver"}))
        wrong_scope = reserve(s_claim)

        # No common claim data across two independent valid task databases.
        other_path = root / "another-root.sqlite3"
        _, _, other_attacker, other_claim = setup(other_path, "other", "nonce-P")
        cross_root = reserve(other_claim)

        pre_path = root / "crash-before.sqlite3"
        _, _, pre_attacker, pre_claim = setup(pre_path, "before", "nonce-C")
        pre_exit = crash(pre_path, pre_attacker, identity(pre_claim), "before")
        after_pre_crash = reserve(pre_claim)

        post_path = root / "crash-after.sqlite3"
        _, _, post_attacker, post_claim = setup(post_path, "after", "nonce-D")
        post_exit = crash(post_path, post_attacker, identity(post_claim), "after")
        after_post_crash = reserve(post_claim)

        gen_path = root / "forged-generation.sqlite3"
        _, _, gen_attacker, gen_claim = setup(gen_path, "generation", "nonce-G")
        legit_generation = reserve(gen_claim)
        # Attacker claims same nonce in a *different* unauthenticated generation.
        gen_victim = new_task(TaskExecutionLedger(gen_path), "spoofed-generation")
        spoofed_generation = reserve({**args(gen_path, gen_victim, "nonce-G"),
                                      "host_generation": "self-declared-gen-SPOOF"})

        all_ledger_events_are_task_only = all(
            e.event_type in ("TASK_OPENED", "TASK_CHECKPOINT")
            for path in root.glob("*.sqlite3")
            for e in TaskExecutionLedger(path).events()
        )
        return {
            "schema": SCHEMA, "frozen_case_sha256": digest(CASES),
            "source_parent_training": "51bd9749c2bf77b9b2b3f9d94f55a56e3ef29896",
            "source_mono_task_ledger_blob": "b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de",
            "clean_first_accepted": fresh["accepted"],
            "counterfeit_marker_blocks_legitimate_first": (
                not blocked["accepted"] and blocked["reason"] == "DUPLICATE_SCOPE_NONCE"),
            "poisoned_victim_stays_open": not source_after.closed,
            "forged_marker_is_valid_task_history": marker_is_valid_task_history,
            "forged_marker_is_foreign_task": marker_is_foreign_task,
            "different_nonce_accepted": wrong_nonce["accepted"],
            "different_scope_accepted": wrong_scope["accepted"],
            "foreign_root_same_nonce_accepted": cross_root["accepted"],
            "precommit_crash_exit": pre_exit,
            "postcommit_crash_exit": post_exit,
            "after_precommit_crash_accepted": after_pre_crash["accepted"],
            "after_postcommit_crash_rejected": not after_post_crash["accepted"],
            "postcommit_rejection_reason": after_post_crash["reason"],
            "first_generation_accepted": legit_generation["accepted"],
            "spoofed_generation_reuse_accepted": spoofed_generation["accepted"],
            "all_events_open_or_checkpoint": all_ledger_events_are_task_only,
            "real_issuer_authenticated": False,
            "receiver_generation_attested": False,
            "process_loaded_artifact_observed": False,
            "canonical_task_completed": False,
            "protected_effect_performed": False,
            "memory_promoted": False,
            "production_protocol_decision": "REJECT_CHECKPOINT_AS_NONCE_AUTHORITY",
            "elapsed_ns": perf_counter_ns()-started,
        }


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--die":
        crash_worker(json.loads(sys.stdin.read()))
    else:
        print(json.dumps(evaluate(), sort_keys=True, indent=2))
