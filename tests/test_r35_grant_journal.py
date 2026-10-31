"""Synthetic grant-journal invariants; these tests confer no bank authority."""

import concurrent.futures
import unittest

from trainingday.r35_grant_journal import (
    AuditIndeterminate,
    GrantRecord,
    GrantHold,
    GrantStatus,
    JournalSnapshot,
    RollbackSuspected,
    SyntheticGrantJournal,
)


class SyntheticGrantJournalTests(unittest.TestCase):
    def setUp(self):
        self.journal = SyntheticGrantJournal()
        self.original = self.journal.prepare("grant-1", "nonce-1")

    @staticmethod
    def accept_audit(_event):
        return True

    def test_single_consume_then_replay_is_denied(self):
        consumed = self.journal.consume(
            "grant-1", "nonce-1", self.original.revision, self.accept_audit)
        self.assertEqual(consumed.status, GrantStatus.CONSUMED)
        self.assertGreater(consumed.revision, self.original.revision)
        with self.assertRaises(GrantHold):
            self.journal.consume(
                "grant-1", "nonce-1", consumed.revision, self.accept_audit)

    def test_nonce_cannot_be_reprepared_after_consumption(self):
        self.journal.consume(
            "grant-1", "nonce-1", self.original.revision, self.accept_audit)
        with self.assertRaises(GrantHold):
            self.journal.prepare("grant-2", "nonce-1")

    def test_wrong_nonce_and_stale_revision_are_denied(self):
        with self.assertRaises(GrantHold):
            self.journal.consume(
                "grant-1", "wrong-nonce", self.original.revision, self.accept_audit)
        with self.assertRaises(GrantHold):
            self.journal.consume(
                "grant-1", "nonce-1", 0, self.accept_audit)
        self.assertEqual(self.journal.lookup("grant-1").status, GrantStatus.PREPARED)

    def test_custom_nonce_comparison_cannot_impersonate_exact_nonce(self):
        class AnyNonce:
            def __ne__(self, _other):
                return False

        with self.assertRaises(GrantHold):
            self.journal.consume("grant-1", AnyNonce(),
                                 self.original.revision, self.accept_audit)
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.PREPARED)

    def test_custom_grant_id_hash_and_equality_cannot_impersonate_string(self):
        class AnyGrantId:
            def __hash__(self):
                return hash("grant-1")

            def __eq__(self, _other):
                return True

        with self.assertRaises(GrantHold):
            self.journal.consume(AnyGrantId(), "nonce-1",
                                 self.original.revision, self.accept_audit)
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.PREPARED)

    def test_concurrent_claim_has_exactly_one_winner_inside_process(self):
        audit_events = []

        def audit(event):
            audit_events.append(event)
            return True

        def attempt(_):
            try:
                self.journal.consume("grant-1", "nonce-1", self.original.revision, audit)
                return "consumed"
            except GrantHold:
                return "held"

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            outcomes = list(pool.map(attempt, range(30)))
        self.assertEqual(outcomes.count("consumed"), 1)
        self.assertEqual(outcomes.count("held"), 29)
        self.assertEqual(len(audit_events), 1)

    def test_audit_write_failure_is_indeterminate_not_retriable(self):
        def audit(_event):
            raise OSError("write may have happened before disconnect")

        with self.assertRaises(AuditIndeterminate):
            self.journal.consume("grant-1", "nonce-1", self.original.revision, audit)
        record = self.journal.lookup("grant-1")
        self.assertEqual(record.status, GrantStatus.RECONCILIATION_REQUIRED)
        with self.assertRaises(GrantHold):
            self.journal.consume("grant-1", "nonce-1", record.revision, self.accept_audit)

    def test_audit_not_explicit_true_is_indeterminate(self):
        with self.assertRaises(AuditIndeterminate):
            self.journal.consume(
                "grant-1", "nonce-1", self.original.revision, lambda _event: None)
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.RECONCILIATION_REQUIRED)

    def test_crash_observed_inside_audit_cannot_be_overwritten_by_success(self):
        def audit(_event):
            self.journal.mark_crash_suspected()
            return True

        with self.assertRaises(GrantHold):
            self.journal.consume("grant-1", "nonce-1",
                                 self.original.revision, audit)
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.RECONCILIATION_REQUIRED)

    def test_crash_suspected_seals_pending_grants(self):
        self.journal.mark_crash_suspected()
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.RECONCILIATION_REQUIRED)
        with self.assertRaises(GrantHold):
            self.journal.consume("grant-1", "nonce-1", self.original.revision,
                                 self.accept_audit)
        with self.assertRaises(GrantHold):
            self.journal.prepare("grant-2", "nonce-2")

    def test_stale_snapshot_does_not_restore_grant_authority(self):
        stale = self.journal.snapshot()
        self.journal.consume(
            "grant-1", "nonce-1", self.original.revision, self.accept_audit)
        current = self.journal.snapshot()
        self.assertGreater(current.revision, stale.revision)
        with self.assertRaises(RollbackSuspected):
            SyntheticGrantJournal.import_untrusted_snapshot(
                stale, asserted_min_revision=current.revision)
        untrusted = SyntheticGrantJournal.import_untrusted_snapshot(stale)
        with self.assertRaises(GrantHold):
            untrusted.consume("grant-1", "nonce-1", self.original.revision,
                              self.accept_audit)

    def test_malformed_snapshot_records_none_is_controlled_hold(self):
        with self.assertRaises(GrantHold):
            SyntheticGrantJournal.import_untrusted_snapshot(
                JournalSnapshot(revision=1, records=None))

    def test_malformed_snapshot_record_status_and_revision_are_held(self):
        invalid_records = (
            GrantRecord("grant-1", "nonce-1", "PREPARED", 1),
            GrantRecord("grant-1", "nonce-1", GrantStatus.PREPARED, "1"),
            GrantRecord("grant-1", "nonce-1", GrantStatus.PREPARED, -1),
            GrantRecord("grant-1", "nonce-1", GrantStatus.PREPARED, 9),
        )
        for record in invalid_records:
            with self.subTest(record=repr(record)):
                with self.assertRaises(GrantHold):
                    SyntheticGrantJournal.import_untrusted_snapshot(
                        JournalSnapshot(revision=2, records=(record,)))

    def test_prepare_return_is_not_internal_record_alias(self):
        object.__setattr__(self.original, "nonce", "attacker-nonce")
        self.assertEqual(self.journal.lookup("grant-1").nonce, "nonce-1")
        consumed = self.journal.consume("grant-1", "nonce-1", 1,
                                        self.accept_audit)
        self.assertEqual(consumed.status, GrantStatus.CONSUMED)

    def test_lookup_return_mutation_cannot_rewrite_internal_status(self):
        observed = self.journal.lookup("grant-1")
        object.__setattr__(observed, "status", GrantStatus.CONSUMED)
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.PREPARED)

    def test_consume_return_mutation_cannot_reactivate_grant(self):
        consumed = self.journal.consume("grant-1", "nonce-1",
                                        self.original.revision, self.accept_audit)
        object.__setattr__(consumed, "status", GrantStatus.PREPARED)
        with self.assertRaises(GrantHold):
            self.journal.consume("grant-1", "nonce-1", consumed.revision,
                                 self.accept_audit)
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.CONSUMED)

    def test_snapshot_record_mutation_cannot_edit_live_journal(self):
        snapshot = self.journal.snapshot()
        object.__setattr__(snapshot.records[0], "nonce", "attacker-nonce")
        object.__setattr__(snapshot.records[0], "status", GrantStatus.CONSUMED)
        self.assertEqual(self.journal.lookup("grant-1").nonce, "nonce-1")
        self.assertEqual(self.journal.lookup("grant-1").status,
                         GrantStatus.PREPARED)

    def test_audit_callback_cannot_rewrite_appended_event(self):
        leaked_event_refs = []

        def hostile_audit(event):
            leaked_event_refs.append(event)
            object.__setattr__(event, "grant_id", "forged-grant")
            object.__setattr__(event, "nonce", "forged-nonce")
            object.__setattr__(event, "kind", "FORGED_AUTHORIZATION")
            object.__setattr__(event, "revision", -9)
            return True

        self.journal.consume("grant-1", "nonce-1", self.original.revision,
                             hostile_audit)
        audit_entry = self.journal._events[-1]  # Hostile test of internal alias.
        self.assertEqual(audit_entry.grant_id, "grant-1")
        self.assertEqual(audit_entry.nonce, "nonce-1")
        self.assertEqual(audit_entry.kind, "CONSUME_SYNTHETIC")
        self.assertGreater(audit_entry.revision, 0)
        object.__setattr__(leaked_event_refs[0], "grant_id", "changed-later")
        self.assertEqual(self.journal._events[-1].grant_id, "grant-1")

    def test_invalid_inputs_never_prepare(self):
        with self.assertRaises(GrantHold):
            self.journal.prepare("", "nonce-3")
        with self.assertRaises(GrantHold):
            self.journal.prepare("grant-3", "")
        with self.assertRaises(GrantHold):
            self.journal.prepare("grant-1", "nonce-4")


if __name__ == "__main__":
    unittest.main()
