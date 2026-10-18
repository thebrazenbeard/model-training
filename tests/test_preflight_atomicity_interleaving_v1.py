"""Deterministic actual SQLite interleaving: source preflight is not admission.

Separate writer thread commits an A CHECKPOINT *after* preflight snapshots
events(A) and *before* the preflight returns the fixture verdict. Intentional
RED assertion: stale precheck returns True; no model/provider is admitted.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import secrets

from vera_core.task_execution import TaskExecutionLedger
from evaluate_receiver_bound_preflight_v1 import (
    mac, make_receipt, original_task, preflight, witness_of,
)


def fixture(ledger):
    original = original_task(ledger)
    receipt = make_receipt(original)
    key = secrets.token_bytes(32)
    return original, receipt, key, witness_of(receipt)


def advance_a(ledger, original):
    ledger.checkpoint(
        original["task_id"], "concurrent-fixture-task-A",
        completed_evidence=("test-only-task-advance",),
        blockers=("NO_ADMITTED_PROVIDER",),
        protected_effects_still_gated=("PROVIDER_EFFECT",),
        next_frontier="source race must not complete task",
        lifecycle_evidence_digest="3" * 64,
        expected_task_head=original["opened_head"],
    )


class ConcurrentSnapshot:
    """Return a real stale event snapshot only AFTER the actual write commits."""
    def __init__(self, ledger, let_writer_start, writer_done):
        self.ledger = ledger
        self.let_writer_start = let_writer_start
        self.writer_done = writer_done

    def read(self, task_id):
        return self.ledger.read(task_id)

    def events(self, task_id):
        captured = self.ledger.events(task_id)
        self.let_writer_start.set()
        if not self.writer_done.wait(timeout=5):
            raise AssertionError("writer did not finish actual SQLite task checkpoint")
        return captured


def test_valid_static_fixture_can_precheck_without_treating_it_as_provider():
    with TemporaryDirectory(prefix="training-preflight-race-control-") as root:
        ledger = TaskExecutionLedger(Path(root) / "tasks.sqlite3")
        original, receipt, key, witness = fixture(ledger)
        used = set()
        assert preflight(receipt, mac(key, receipt), key,
                         witness, ledger, original, used)
        assert not ledger.read(original["task_id"]).closed


def test_already_advanced_same_task_receipt_rejected():
    with TemporaryDirectory(prefix="training-preflight-race-stale-") as root:
        ledger = TaskExecutionLedger(Path(root) / "tasks.sqlite3")
        original, receipt, key, witness = fixture(ledger)
        advance_a(ledger, original)
        assert not preflight(receipt, mac(key, receipt), key,
                             witness, ledger, original, set())
        assert len(ledger.events(original["task_id"])) == 2


def test_actual_second_thread_can_advance_a_after_preflight_snapshot():
    with TemporaryDirectory(prefix="training-preflight-race-") as root:
        ledger = TaskExecutionLedger(Path(root) / "tasks.sqlite3")
        original, receipt, key, witness = fixture(ledger)
        let_writer_start = Event()
        writer_done = Event()
        errors = []

        def writer():
            try:
                if not let_writer_start.wait(timeout=5):
                    raise AssertionError("reader never reached task event snapshot")
                advance_a(ledger, original)
            except Exception as exc:
                errors.append(str(exc))
            finally:
                writer_done.set()

        thread = Thread(target=writer, name="research-task-A-real-writer")
        thread.start()
        try:
            used = set()
            # The source preflight accepts stale A after an actual SQLite
            # checkpoint committed, showing the missing atomic owner gate.
            stale_accepted = preflight(receipt, mac(key, receipt), key,
                witness, ConcurrentSnapshot(ledger, let_writer_start, writer_done),
                original, used)
        finally:
            thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors, errors
        assert len(ledger.events(original["task_id"])) == 2
        assert ledger.events(original["task_id"])[-1].event_type == "TASK_CHECKPOINT"
        assert not ledger.read(original["task_id"]).closed
        assert stale_accepted is True  # intentional unsafe SOURCE RED
        assert receipt["one_use_nonce"] in used
        assert receipt["synthetic"] is False  # caller field, not model proof
