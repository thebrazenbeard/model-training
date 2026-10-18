"""Disposable Mono task-ledger CAS nonce-bound *research*, not provider authorization.

Tests real SQLite BEGIN IMMEDIATE across independent CPython processes; uses
existing TaskExecutionLedger.checkpoint, never a new trust database.
The checkpoint is NOT a production nonce or provider-admission protocol.
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

SCHEMA = "TRAINING_CANONICAL_TASK_NONCE_SOURCE_PROBE_V1"
CASE_DESCRIPTIONS = (
    ("same_task_cold_replay", False),
    ("same_task_concurrent_replay", False),
    ("same_task_stale_head", False),
    ("different_task_same_nonce", False),
    ("valid_first_reservation", True),
    ("valid_new_nonce_after_current_head", True),
)


def digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worker(args):
    ledger = TaskExecutionLedger(args["ledger"])
    receipt_key = digest({"task_id": args["task_id"],
                          "source_digest": args["source_digest"],
                          "nonce": args["nonce"]})
    start = perf_counter_ns()
    try:
        current = ledger.checkpoint(
            args["task_id"], "nonce-precheck-" + receipt_key,
            completed_evidence=("source-precheck-only",),
            blockers=("NO_PROVIDER_ADMITTED",),
            protected_effects_still_gated=("PROVIDER_EFFECT",),
            next_frontier="Do not execute or complete; fixture nonce reservation",
            lifecycle_evidence_digest=receipt_key,
            expected_task_head=args["expected_task_head"],
        )
        return {"accepted": True, "journal_head": current.journal_head_digest,
                "task_closed": current.closed,
                "elapsed_ns": perf_counter_ns() - start}
    except (TaskExecutionError, ValueError) as exc:
        return {"accepted": False, "failure": type(exc).__name__, "detail": str(exc),
                "elapsed_ns": perf_counter_ns() - start}


def subprocess_worker(args):
    proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                           "--worker"], input=json.dumps(args), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, timeout=20, env=os.environ.copy())
    if proc.returncode:
        raise RuntimeError("child failure: " + proc.stderr[-700:])
    return json.loads(proc.stdout)


def two_concurrent(args):
    runners = [subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
               "--worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
               stderr=subprocess.PIPE, text=True, env=os.environ.copy())
               for _ in range(2)]
    # Give BOTH children their input before awaiting any result.
    # This is overlapping process execution, not a deterministic barrier.
    payload = json.dumps(args)
    for proc in runners:
        proc.stdin.write(payload)
        proc.stdin.close()
    out = []
    for proc in runners:
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        proc.wait(timeout=20)
        if proc.returncode:
            raise RuntimeError("child failure: " + stderr[-700:])
        out.append(json.loads(stdout))
    return out


def task(ledger, message_id, literal):
    opened = open_message(ledger, "conversation-safe", message_id, literal)
    return {"task_id": opened["task_id"], "source_digest": opened["source_digest"],
            "expected_task_head": opened["opened_head"]}


def args_for(ledger, t, nonce):
    return {"ledger": str(ledger.path), **t, "nonce": nonce}


def evaluate():
    fixture_digest = digest(CASE_DESCRIPTIONS)
    with tempfile.TemporaryDirectory(prefix="training-task-cas-nonce-") as tmp:
        path = Path(tmp) / "mono_tasks.sqlite3"
        ledger = TaskExecutionLedger(path)
        a = task(ledger, "message-a", "literal source a")
        b = task(ledger, "message-b", "literal source b")
        # A: same task, cold process replay from first expected head.
        base = args_for(ledger, a, "nonce-A")
        first = subprocess_worker(base)
        same_cold = subprocess_worker(base)
        # B: two real child processes racing against same task/head.
        race_task = task(ledger, "message-race", "literal race")
        race = two_concurrent(args_for(ledger, race_task, "nonce-race"))
        # C: stale head must reject independently of nonce.
        stale = subprocess_worker(args_for(ledger, a, "different-nonce-stale"))
        # D: same nonce on two other distinct tasks can BOTH be accepted.
        # This is a documented RED for *global provider nonce* uniqueness.
        cross_a = task(ledger, "message-cross-a", "cross a")
        cross_b = task(ledger, "message-cross-b", "cross b")
        x = subprocess_worker(args_for(ledger, cross_a, "shared-nonce"))
        y = subprocess_worker(args_for(ledger, cross_b, "shared-nonce"))
        # E: legitimate new nonce when CAS expected task head is refreshed.
        latest = ledger.events(a["task_id"])[-1].event_digest
        new = subprocess_worker(args_for(ledger, {**a, "expected_task_head": latest},
                                             "nonce-B"))
        # Source integrity all task events are actual persisted checkpoints,
        # not provider events, and all task results remain unclosed.
        task_ids = [a["task_id"], b["task_id"], race_task["task_id"],
                    cross_a["task_id"], cross_b["task_id"]]
        states = {tid: ledger.read(tid) for tid in task_ids}
        return {
            "schema": SCHEMA, "fixture_sha256": fixture_digest,
            "positive_first": bool(first["accepted"]),
            "positive_next": bool(new["accepted"]), "next_case": new,
            "cold_replay_rejected": not same_cold["accepted"],
            "stale_task_head_rejected": not stale["accepted"],
            "concurrent_accept_count": sum(r["accepted"] for r in race),
            "cross_task_same_nonce_accept_count": int(x["accepted"])+int(y["accepted"]),
            "global_nonce_uniqueness": "RED_TWO_DISTINCT_TASKS_ACCEPT_SAME_NONCE",
            "task_states_unclosed": all(not s.closed for s in states.values()),
            "event_types_only_open_and_checkpoint": all(
               set(e.event_type for e in ledger.events(tid)) <=
               {"TASK_OPENED", "TASK_CHECKPOINT"} for tid in task_ids),
            "cold_replay_ns": same_cold["elapsed_ns"],
            "first_reservation_ns": first["elapsed_ns"],
            "new_reservation_ns": new["elapsed_ns"],
            "real_provider_authenticated": False,
            "real_model_loaded": False,
            "memory_written": False,
            "provider_effect_performed": False,
            "real_guest_boot": False,
            "decision": "TASK_LOCAL_CAS_SOURCE_UTILITY_ONLY_GLOBAL_NONCE_HOLD"
        }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(worker(json.loads(sys.stdin.read())), sort_keys=True))
    else:
        print(json.dumps(evaluate(), indent=2, sort_keys=True))
