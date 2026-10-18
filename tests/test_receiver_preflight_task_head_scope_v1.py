"""Training admission preflight: unrelated task activity must not cause false HOLD.

Source subject: Mono #77 a351d362... task_execution.py blob b79504e5...
and Training #112 59b46ce... preflight. These are synthetic receipt
tests, never accepted model/provider evidence.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import secrets

from vera_core.local_conversation import open_message
from vera_core.task_execution import TaskExecutionLedger
from evaluate_receiver_bound_preflight_v1 import (
    baseline, mac, make_receipt, original_task, preflight, witness_of,
)

def checked_context(ledger, original):
    key = secrets.token_bytes(32)
    receipt = make_receipt(original)
    return key, receipt, witness_of(receipt)

def test_unrelated_task_must_not_false_hold_original_receipt():
    with TemporaryDirectory() as temp:
        ledger = TaskExecutionLedger(Path(temp) / "mono.sqlite3")
        original = original_task(ledger)
        key, receipt, witness = checked_context(ledger, original)
        assert preflight(receipt, mac(key,receipt), key, witness, ledger, original, set())
        foreign = open_message(ledger, "conversation-good", "unrelated-B", "separate literal")
        assert ledger.verify_chain() == foreign["opened_head"]
        assert ledger.read(original["task_id"]).journal_head_digest != original["opened_head"]
        assert ledger.events(original["task_id"])[-1].event_digest == original["opened_head"]
        # Hard positive: task B cannot invalidate task A's actual current source.
        assert preflight(receipt, mac(key,receipt), key, witness, ledger, original, set())

def test_same_task_advance_remains_hold():
    with TemporaryDirectory() as temp:
        ledger = TaskExecutionLedger(Path(temp) / "mono.sqlite3")
        original = original_task(ledger)
        key, receipt, witness = checked_context(ledger, original)
        ledger.checkpoint(original["task_id"], "advance-original",
            completed_evidence=("test-only",), blockers=("no-admitted-provider",),
            protected_effects_still_gated=("PROVIDER_EFFECT",),
            next_frontier="keep held", lifecycle_evidence_digest="1"*64,
            expected_task_head=original["opened_head"])
        assert baseline(receipt)
        assert not preflight(receipt, mac(key,receipt), key, witness, ledger, original, set())

def test_interleaving_does_not_override_wrong_source_or_scope():
    with TemporaryDirectory() as temp:
        ledger = TaskExecutionLedger(Path(temp) / "mono.sqlite3")
        original = original_task(ledger)
        open_message(ledger, "conversation-good", "unrelated-B", "separate literal")
        key, receipt, witness = checked_context(ledger, original)
        for field, value in (("source_digest", "0"*64),
                             ("conversation_id", "foreign-conversation"),
                             ("selected_routes", ["route-A","route-B"])):
            wrong = {**receipt,field:value}
            assert not preflight(wrong, mac(key,wrong), key, witness, ledger, original, set())

def test_fixture_not_provider_admission():
    with TemporaryDirectory() as temp:
        ledger = TaskExecutionLedger(Path(temp) / "mono.sqlite3")
        original = original_task(ledger)
        key, receipt, witness = checked_context(ledger, original)
        open_message(ledger, "conversation-good", "unrelated-B", "separate literal")
        assert not ledger.read(original["task_id"]).closed
        assert receipt["synthetic"] is False  # authored fixture value, NOT verified proof
        assert witness.provider_id == "fixture-provider"
