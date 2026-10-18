"""Test-only receiver-bound admission preflight; NEVER verifies a real provider.

Receiver measurements and signing keys are SYNTHETIC fixture inputs. The
preflight does not authorize effects, mark TASK_COMPLETED, or admit memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from pathlib import Path
import secrets
import tempfile
from time import perf_counter_ns
import tracemalloc

from vera_core.local_conversation import open_message
from vera_core.task_execution import TaskExecutionLedger

CONTRACT = Path(__file__).parent / "docs/contracts/TRAINING_COGNITIVE_CAPABILITY_INTERFACE_V1.json"
REQUIRED = tuple(json.loads(CONTRACT.read_text(encoding="utf-8"))["provider_receipt_required"])
CASES = (
    ("valid_alpha", True, "valid"),
    ("valid_beta", True, "valid_second"),
    ("forged_origin", False, "foreign_source"),
    ("wrong_loaded_model", False, "wrong_artifact"),
    ("wrong_provider_identity", False, "wrong_provider_id"),
    ("wrong_model_identity", False, "wrong_model_id"),
    ("wrong_host_identity", False, "wrong_host_id"),
    ("wrong_manifest", False, "wrong_manifest"),
    ("replayed_nonce", False, "nonce_replay"),
    ("cold_restart_nonce_replay", False, "nonce_cold_replay"),
    ("stale_task_head", False, "stale_head"),
    ("foreign_conversation", False, "foreign_conversation"),
    ("two_selected_routes", False, "multiple_routes"),
    ("synthetic_done", False, "synthetic_done"),
    ("tampered_response", False, "response_mismatch"),
    ("revoked_issuer", False, "revoked"),
    ("unobserved_session", False, "session_mismatch"),
    ("missing_receiver_witness", False, "missing_witness"),
    ("missing_durable_run", False, "missing_run"),
)
SCHEMA = "TRAINING_RECEIVER_BOUND_PREFLIGHT_EVAL_V1"


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(obj):
    return sha256(encode(obj)).hexdigest()


def mac(key, receipt):
    return hmac.digest(key, encode(receipt), "sha256").hex()


@dataclass(frozen=True)
class ReceiverWitness:
    # All values originate in a separate, test-owned fixture, NOT a provider.
    provider_id: str
    model_id: str
    host_id: str
    receiver_owned_verifier_id: str
    model_manifest_sha256: str
    endpoint: str
    loaded_artifact_sha256: str
    host_generation: str
    process_id: str
    session_id: str
    route_id: str
    transport_response_digest: str
    response_sha256: str
    preactive_done_event_id: str
    preactive_completed_run_id: str
    task_result_readback_digest: str


def baseline(receipt):
    """25-field structural presence is NOT source/provider authentication."""
    return all(name in receipt and receipt[name] is not None for name in REQUIRED)


def preflight(receipt, signature, key, witness, ledger, original, used_nonces,
              *, revoked=False):
    """Only PRECHECK_ELIGIBLE, never a task closeout or provider/effect grant."""
    if not baseline(receipt) or witness is None or key is None or revoked:
        return False
    if not isinstance(receipt.get("conversation_id"), str):
        return False
    if receipt["conversation_id"] != original["conversation_id"]:
        return False
    if receipt["request_id"] != original["message_id"]:
        return False
    if receipt["task_id"] != original["task_id"]:
        return False
    if receipt["source_digest"] != original["source_digest"]:
        return False
    if receipt["accepted_task_journal_head"] != original["opened_head"]:
        return False
    state = ledger.read(original["task_id"])
    if state.closed or state.journal_head_digest != original["opened_head"]:
        return False
    events = ledger.events(original["task_id"])
    if len(events) != 1 or events[0].event_type != "TASK_OPENED":
        return False
    if receipt.get("synthetic") is not False:
        return False
    if receipt.get("current_trust_and_revocation") != "CURRENT":
        return False
    if not isinstance(receipt.get("selected_routes"), list):
        return False
    if receipt["selected_routes"] != [receipt["route_id"]]:
        return False
    if receipt["unique_durable_route_selection"] != receipt["route_id"]:
        return False
    if receipt.get("nonce_consumed_once") is not True:
        return False
    nonce = receipt.get("one_use_nonce")
    if not isinstance(nonce, str) or nonce in used_nonces:
        return False
    if not isinstance(signature, str) or not hmac.compare_digest(
        mac(key, receipt), signature
    ):
        return False
    # Signature authenticates a fixture message to a fixture key only.
    # Independently supplied receiver observations must also agree.
    for field in ReceiverWitness.__dataclass_fields__:
        if receipt.get(field) != getattr(witness, field):
            return False
    used_nonces.add(nonce)
    return True


def original_task(ledger):
    opened = open_message(ledger, "conversation-good", "message-good",
                          "exact  literal")
    return {"conversation_id": "conversation-good", "message_id": "message-good",
            "task_id": opened["task_id"], "opened_head": opened["opened_head"],
            "source_digest": opened["source_digest"]}


def make_receipt(original, suffix="alpha"):
    r = {k: f"fixture-{k}" for k in REQUIRED}
    r.update({"conversation_id": original["conversation_id"],
        "request_id": original["message_id"], "task_id": original["task_id"],
        "accepted_task_journal_head": original["opened_head"],
        "source_digest": original["source_digest"],
        "provider_id": "fixture-provider", "model_id": "fixture-model",
        "loaded_artifact_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "host_id": "host-fixture", "host_generation": "generation-A",
        "process_id": "PID-1234", "session_id": "session-A",
        "endpoint": "inert://fixture",
        "one_use_nonce": f"nonce-{suffix}",
        "nonce_consumed_once": True,
        "receiver_owned_verifier_id": "fixture-receiver",
        "current_trust_and_revocation": "CURRENT",
        "route_id": "route-A",
        "unique_durable_route_selection": "route-A",
        "selected_routes": ["route-A"],
        "transport_request_digest": "c" * 64,
        "transport_response_digest": "d" * 64,
        "response_sha256": "e" * 64,
        "preactive_done_event_id": "fixture-event-done",
        "preactive_completed_run_id": "fixture-run-completed",
        "task_result_readback_digest": "f" * 64,
        "synthetic": False})
    return r


def witness_of(r):
    return ReceiverWitness(**{k: r[k] for k in ReceiverWitness.__dataclass_fields__})


def case_input(name, mode, original, key):
    r = make_receipt(original, suffix=name)
    witness = witness_of(r)
    signature = mac(key, r)
    revocation = False
    existing_nonce = False
    if mode == "foreign_source":
        r["source_digest"] = "0" * 64
        signature = mac(secrets.token_bytes(32), r)  # malicious issuer key
    elif mode == "wrong_artifact":
        r["loaded_artifact_sha256"] = "9" * 64
        signature = mac(key, r)  # even signed wrong loaded identity rejected
    elif mode == "wrong_provider_id":
        r["provider_id"] = "provider-substituted"
        signature = mac(key, r)
    elif mode == "wrong_model_id":
        r["model_id"] = "model-substituted"
        signature = mac(key, r)
    elif mode == "wrong_host_id":
        r["host_id"] = "unobserved-host"
        signature = mac(key, r)
    elif mode == "wrong_manifest":
        r["model_manifest_sha256"] = "6" * 64
        signature = mac(key, r)
    elif mode == "nonce_replay":
        existing_nonce = True
    elif mode == "nonce_cold_replay":
        # Reconstructed process has forgotten nonce usage: KNOWN RED.
        existing_nonce = False
    elif mode == "stale_head":
        r["accepted_task_journal_head"] = "8" * 64
        signature = mac(key, r)
    elif mode == "foreign_conversation":
        r["conversation_id"] = "conversation-foreign"
        signature = mac(key, r)
    elif mode == "multiple_routes":
        r["selected_routes"] = ["route-A", "route-B"]
        signature = mac(key, r)
    elif mode == "synthetic_done":
        r["synthetic"] = True
        signature = mac(key, r)
    elif mode == "response_mismatch":
        r["response_sha256"] = "7" * 64
        signature = mac(key, r)
    elif mode == "revoked":
        revocation = True
    elif mode == "session_mismatch":
        r["session_id"] = "session-other"
        signature = mac(key, r)
    elif mode == "missing_witness":
        witness = None
    elif mode == "missing_run":
        witness = ReceiverWitness(**{**witness.__dict__,
            "preactive_completed_run_id": "different-run"})
    return (r, signature, witness, revocation, existing_nonce)


def evaluate():
    frozen = [{"name": n, "precheck_eligible": gold, "mutation": mode}
              for n, gold, mode in CASES]
    fixture_sha = digest(frozen)
    key = secrets.token_bytes(32)
    with tempfile.TemporaryDirectory(prefix="training-receiver-preflight-") as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        rows = []
        for name, gold, mode in CASES:
            r, signature, witness, revoked, existing = case_input(name, mode, original, key)
            nonces = {r["one_use_nonce"]} if existing else set()
            b = baseline(r)
            c = preflight(r, signature, key, witness, ledger, original, nonces,
                          revoked=revoked)
            rows.append({"id": name, "gold_precheck_eligible": gold,
                         "baseline": b, "candidate": c,
                         "correct": c == gold,
                         "no_canonical_close": not ledger.read(original["task_id"]).closed})
        assert all(r["no_canonical_close"] for r in rows)
        # Same fixture/host/process only; no loaded model is observed.
        mock_key = secrets.token_bytes(32)
        reference, _, witness, _, _ = case_input("timed", "valid", original, mock_key)
        sig = mac(mock_key, reference)
        tracemalloc.start()
        t0 = perf_counter_ns()
        for _ in range(150):
            baseline(reference)
        baseline_ns = (perf_counter_ns() - t0) / 150
        tb = perf_counter_ns()
        for _ in range(150):
            preflight(reference, sig, mock_key, witness, ledger, original, set())
        candidate_ns = (perf_counter_ns() - tb) / 150
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {"schema": SCHEMA, "frozen_sha256": fixture_sha,
        "positive_total": sum(gold for _, gold, _ in CASES),
        "adverse_total": sum(not gold for _, gold, _ in CASES),
        "baseline_positive": sum(r["baseline"] and r["gold_precheck_eligible"] for r in rows),
        "baseline_unsafe": sum(r["baseline"] and not r["gold_precheck_eligible"] for r in rows),
        "candidate_positive": sum(r["candidate"] and r["gold_precheck_eligible"] for r in rows),
        "candidate_unsafe": sum(r["candidate"] and not r["gold_precheck_eligible"] for r in rows),
        "candidate_false_hold": sum(not r["candidate"] and r["gold_precheck_eligible"] for r in rows),
        "rows": rows, "bench_calls_each": 150,
        "baseline_ns_per_call": round(baseline_ns),
        "candidate_ns_per_call": round(candidate_ns),
        "tracemalloc_peak_bytes": peak_bytes,
        "real_issuer_trust": False, "real_loaded_model": False,
        "real_pre_active_event": False, "real_provider_transport": False,
        "canonical_task_completed": False, "memory_admitted": False,
        "fixture_key_exported": False, "disposition": "RESEARCH_ONLY_HOLD_REAL_PROVIDER"}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
