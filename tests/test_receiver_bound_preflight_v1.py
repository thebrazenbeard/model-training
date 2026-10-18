"""Hard-negative first: caller-signed identity drift must not pass a receiver binder."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import secrets
from evaluate_receiver_bound_preflight_v1 import (
    CASES, ReceiverWitness, baseline, case_input, evaluate,
    mac, make_receipt, original_task, preflight, witness_of,
)
from vera_core.task_execution import TaskExecutionLedger


def test_receiver_signed_wrong_provider_must_hold():
    with TemporaryDirectory() as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        key = secrets.token_bytes(32)
        r = make_receipt(original)
        w = witness_of(r)
        r["provider_id"] = "provider-substituted"
        assert baseline(r)
        assert not preflight(r, mac(key, r), key, w, ledger, original, set())


def test_receiver_signed_wrong_model_must_hold():
    with TemporaryDirectory() as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        key = secrets.token_bytes(32)
        r = make_receipt(original)
        w = witness_of(r)
        r["model_id"] = "wrong-model"
        assert not preflight(r, mac(key, r), key, w, ledger, original, set())


def test_fixture_digest_and_explicit_ceiling():
    x = evaluate()
    assert (x["baseline_positive"], x["candidate_positive"]) == (2, 2)
    assert x["candidate_unsafe"] == 1
    assert x["real_issuer_trust"] is False
    assert x["real_loaded_model"] is False
    assert x["canonical_task_completed"] is False
    assert x["fixture_key_exported"] is False


def test_nonce_is_consumed_once_in_one_fixture_session():
    with TemporaryDirectory() as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        key = secrets.token_bytes(32)
        r = make_receipt(original)
        witness = witness_of(r)
        used = set()
        proof = mac(key, r)
        assert preflight(r, proof, key, witness, ledger, original, used)
        assert not preflight(r, proof, key, witness, ledger, original, used)
        assert used == {r["one_use_nonce"]}
        assert not ledger.read(original["task_id"]).closed


def test_actual_mono_checkpoint_advances_head_then_rejects_old_receipt():
    with TemporaryDirectory() as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        key = secrets.token_bytes(32)
        r = make_receipt(original)
        w = witness_of(r)
        ledger.checkpoint(
            original["task_id"], "source-only-fixture-checkpoint",
            completed_evidence=("test-only-open",),
            blockers=("no-admitted-provider",),
            protected_effects_still_gated=("PROVIDER_EFFECT",),
            next_frontier="Preserve original held task",
            lifecycle_evidence_digest="1" * 64,
            expected_task_head=original["opened_head"],
        )
        assert ledger.read(original["task_id"]).journal_head_digest != original["opened_head"]
        assert baseline(r)
        assert not preflight(r, mac(key, r), key, w, ledger, original, set())
        assert not ledger.read(original["task_id"]).closed


def test_wrong_manifest_and_host_are_independently_compared():
    with TemporaryDirectory() as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        key = secrets.token_bytes(32)
        for field in ("host_id", "model_manifest_sha256", "receiver_owned_verifier_id"):
            r = make_receipt(original, suffix=field)
            w = witness_of(r)
            r[field] = "unexpected"
            assert not preflight(r, mac(key, r), key, w, ledger, original, set())


def test_current_frozen_case_results_and_tradeoff():
    x = evaluate()
    assert x["frozen_sha256"] == "5523f12b8b3e761fa63bf38c9cbeb4bf68b1c92c4a524393cc071ffe0de77dd9"
    assert x["positive_total"] == 2 and x["adverse_total"] == 17
    assert (x["baseline_positive"], x["baseline_unsafe"]) == (2, 17)
    assert (x["candidate_positive"], x["candidate_unsafe"], x["candidate_false_hold"]) == (2, 1, 0)
    assert x["candidate_ns_per_call"] > x["baseline_ns_per_call"]


def test_undurable_nonce_store_is_known_unsafe_across_verifier_recreation():
    with TemporaryDirectory() as td:
        ledger = TaskExecutionLedger(Path(td) / "tasks.sqlite3")
        original = original_task(ledger)
        key = secrets.token_bytes(32)
        receipt = make_receipt(original)
        witness = witness_of(receipt)
        proof = mac(key, receipt)
        first_process_nonces = set()
        assert preflight(receipt, proof, key, witness, ledger, original,
                         first_process_nonces)
        assert receipt["one_use_nonce"] in first_process_nonces
        # A re-created verifier loses its nonce state. The duplicate PASSES.
        # This is intentionally a RED outcome and a production admission HOLD.
        restarted_verifier_nonces = set()
        assert preflight(receipt, proof, key, witness, ledger, original,
                         restarted_verifier_nonces)
        assert not ledger.read(original["task_id"]).closed