"""Does Mono action_lock actually fence independent canonical task checkpoints?

Source-only experiment against the EXISTING Mono TaskExecutionLedger and
PortableFileLock. No real provider, nonce permit, memory or external effect.
The separate SQLite action-lock database is NOT a transaction on task events.
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

from vera_core.local_conversation import open_message
from vera_core.task_execution import TaskExecutionLedger, TaskExecutionError

SCHEMA = "TRAINING_MONO_ACTION_LOCK_NOT_TASK_FENCE_V1"
CASES = (
    "unlocked_cold_checkpoint_succeeds",
    "foreign_process_checkpoint_commits_while_action_lock_held",
    "same_task_stale_cas_still_rejects",
    "unrelated_task_global_head_cannot_stale_task_event",
    "all_records_unclosed_no_provider",
)


def source_case_digest() -> str:
    return sha256(json.dumps(CASES, separators=(",", ":")).encode()).hexdigest()


def new_task(ledger: TaskExecutionLedger, message: str) -> dict:
    opened = open_message(ledger, "source-only-training-conversation",
                          message, "literal-" + message)
    return {"task_id": opened["task_id"], "task_head": opened["opened_head"],
            "source_digest": opened["source_digest"]}


def checkpoint(args: dict) -> dict:
    ledger = TaskExecutionLedger(args["path"])
    began = perf_counter_ns()
    try:
        state = ledger.checkpoint(
            args["task_id"], args["checkpoint_id"],
            completed_evidence=("test-only-no-provider",),
            blockers=("PROVIDER_NOT_ADMITTED",),
            protected_effects_still_gated=("PROTECTED_EFFECT",),
            next_frontier="source-experiment-task-only",
            lifecycle_evidence_digest="f" * 64,
            expected_task_head=args["expected_task_head"],
        )
        return {"accepted": True, "closed": state.closed,
                "elapsed_ns": perf_counter_ns() - began}
    except (TaskExecutionError, ValueError) as exc:
        return {"accepted": False, "failure": type(exc).__name__,
                "elapsed_ns": perf_counter_ns() - began}


def cold_writer(args: dict) -> dict:
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--checkpoint"]
    runner = subprocess.run(command, input=json.dumps(args), text=True,
                            capture_output=True, env=os.environ.copy(),
                            timeout=12, check=False)
    if runner.returncode:
        raise RuntimeError(f"cold task writer exited {runner.returncode}: "
                           + runner.stderr[-700:])
    return json.loads(runner.stdout)


def task_args(path: Path, task: dict, ident: str):
    return {"path": str(path), "task_id": task["task_id"],
            "checkpoint_id": ident, "expected_task_head": task["task_head"]}


def evaluate() -> dict:
    began = perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="training-action-lock-fence-") as tmp:
        path = Path(tmp) / "mono.sqlite3"
        ledger = TaskExecutionLedger(path)

        control = new_task(ledger, "control")
        control_result = cold_writer(task_args(path, control, "control-one"))

        guarded = new_task(ledger, "guarded-A")
        original_a_head = guarded["task_head"]
        with ledger.action_lock():
            # The child process does not acquire the action_lock or its own
            # action_lock handle. It uses ordinary existing TaskExecutionLedger
            # checkpoint API on the canonical task events SQLite path.
            external = cold_writer(task_args(path, guarded, "foreign-writer"))
            events_under_lock = ledger.events(guarded["task_id"])
            global_head_under_lock = ledger.head
            # Explicit signal: checkpoint committed before our context exits.
            committed_while_action_lock_held = (
                external["accepted"]
                and len(events_under_lock) == 2
                and events_under_lock[-1].event_type == "TASK_CHECKPOINT"
            )

        old_a_again = cold_writer(task_args(path, guarded, "stale-duplicate"))
        latest_a_event_head = ledger.events(guarded["task_id"])[-1].event_digest

        other = new_task(ledger, "unrelated-B")
        a_event_head_after_b = ledger.events(guarded["task_id"])[-1].event_digest
        global_head_after_b = ledger.head

        all_events = ledger.events()
        return {
            "schema": SCHEMA,
            "frozen_case_sha256": source_case_digest(),
            "training_base_head": "7ad7088993249c00f718896c371eaa68c9013eb2",
            "mono_task_execution_blob": "b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de",
            "cold_unlocked_checkpoint_accepted": control_result["accepted"],
            "cold_writer_accepted_while_action_lock_held": external["accepted"],
            "writer_checkpoint_visible_before_unlock": committed_while_action_lock_held,
            "action_lock_still_held_at_readback": True,
            "stale_task_cas_rejected": not old_a_again["accepted"],
            "stale_rejection_type": old_a_again.get("failure"),
            "original_task_head_changed_by_writer": latest_a_event_head != original_a_head,
            "task_head_unchanged_by_unrelated_B": a_event_head_after_b == latest_a_event_head,
            "global_head_changed_by_unrelated_B": global_head_after_b != global_head_under_lock,
            "global_chain_verified": ledger.verify_chain() == global_head_after_b,
            "all_tasks_open": all(not state.closed for state in ledger.tasks()),
            "events_only_open_or_checkpoint": all(
                e.event_type in ("TASK_OPENED", "TASK_CHECKPOINT")
                for e in all_events),
            "real_provider_authenticated": False,
            "loaded_model_observed": False,
            "canonical_result_accepted": False,
            "memory_admitted": False,
            "protected_effect_performed": False,
            "finding": "ACTION_LOCK_NOT_IMPLICIT_LOCK_FOR_ORDINARY_TASK_CHECKPOINT",
            "elapsed_ns": perf_counter_ns() - began,
        }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--checkpoint":
        print(json.dumps(checkpoint(json.loads(sys.stdin.read())), sort_keys=True))
    else:
        print(json.dumps(evaluate(), sort_keys=True, indent=2))
