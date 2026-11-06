"""Hostile source-seam tests: no actual model/service/SQLite writes."""
import unittest
from types import SimpleNamespace

from trainingday.neurosync_statebus_gate import (
    CrossRepoEvidence, GovernanceHold, HostPrincipal, PhysicalTrainingHold,
    RepoPins, TrainingDayNeuroSyncGate,
)

A, B, C, D, E = (c * 64 for c in "abcde")
PINS = RepoPins("sha1:" + "a" * 40, "sha1:" + "b" * 40, A, B)
EVIDENCE = CrossRepoEvidence("evt_123", C, PINS, D, "TEST")
PRINCIPAL = HostPrincipal("patrick", "trainingday", "sess1")


def event(event_id="evt_123", content_hash=C, *, offset=4,
          chain=E, target="trainingday", session="sess1"):
    return SimpleNamespace(
        event_offset=offset, key_revision=2, chain_hash=chain,
        delta=SimpleNamespace(event_id=event_id, content_hash=content_hash,
                              target_id=target, session_id=session))


class FakeLedger:
    def __init__(self, items=None):
        self.items = (event(),) if items is None else items
        self.read_calls = 0
        self.commit_calls = 0

    def events(self, target_id, session_id, *, after_offset, limit):
        self.read_calls += 1
        return tuple(item for item in self.items if item.event_offset > after_offset)[:limit]

    def commit(self, *_args, **_kwargs):
        self.commit_calls += 1
        raise AssertionError("commit is not a safe receipt read")


class GovernanceTests(unittest.TestCase):
    def gate(self, ledger, *, authorize=lambda _: True, verify=lambda _: True):
        return TrainingDayNeuroSyncGate(
            pinned=PINS, ledger=ledger,
            authorize=authorize, verify_attestation=verify)

    def recover(self, gate, evidence=EVIDENCE):
        return gate.recover_committed_receipt(principal=PRINCIPAL, evidence=evidence)

    def test_success_authorizes_and_attests_before_and_after(self):
        ledger, policy_calls, verify_calls = FakeLedger(), [], []
        gate = self.gate(ledger,
                         authorize=lambda check: policy_calls.append(check.operation) or True,
                         verify=lambda receipt: verify_calls.append(receipt) or True)
        view = self.recover(gate)
        self.assertEqual(view.event_id, EVIDENCE.event_id)
        self.assertTrue(view.already_committed)
        self.assertEqual(policy_calls, ["statebus.receipt-read.pre", "statebus.receipt-read.post"])
        self.assertEqual(verify_calls, [EVIDENCE, EVIDENCE])
        self.assertEqual(ledger.commit_calls, 0)

    def test_denied_current_policy_never_reads_and_never_commits(self):
        ledger = FakeLedger()
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger, authorize=lambda _: False))
        self.assertEqual((ledger.read_calls, ledger.commit_calls), (0, 0))

    def test_truthy_or_throwing_policy_fails_closed(self):
        ledger = FakeLedger()
        for callback in [lambda _: "yes", lambda _: 1,
                         lambda _: (_ for _ in ()).throw(RuntimeError("secret"))]:
            with self.subTest(callback=callback), self.assertRaises(GovernanceHold):
                self.recover(self.gate(ledger, authorize=callback))
        self.assertEqual(ledger.read_calls, 0)

    def test_mid_read_policy_revocation_suppresses_receipt(self):
        ledger, calls = FakeLedger(), [0]
        def policy(_):
            calls[0] += 1
            return calls[0] == 1
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger, authorize=policy))
        self.assertEqual((ledger.read_calls, ledger.commit_calls), (1, 0))

    def test_mid_read_attestation_revocation_suppresses_receipt(self):
        ledger, calls = FakeLedger(), [0]
        def attest(_):
            calls[0] += 1
            return calls[0] == 1
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger, verify=attest))
        self.assertEqual(calls[0], 2)
        self.assertEqual((ledger.read_calls, ledger.commit_calls), (1, 0))

    def test_unavailable_or_false_verifier_never_reads(self):
        ledger = FakeLedger()
        for verifier in [lambda _: False,
                         lambda _: (_ for _ in ()).throw(OSError("offline"))]:
            with self.subTest(verifier=verifier), self.assertRaises(GovernanceHold):
                self.recover(self.gate(ledger, verify=verifier))
        self.assertEqual(ledger.read_calls, 0)

    def test_cross_repo_pin_drift_denies_before_read(self):
        ledger = FakeLedger()
        wrong = RepoPins(PINS.neurosync_git_oid, PINS.training_git_oid, E, B)
        claim = CrossRepoEvidence(EVIDENCE.event_id, C, wrong, D, "TEST")
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger), claim)
        self.assertEqual(ledger.read_calls, 0)

    def test_mismatch_and_poisoned_events_no_commit(self):
        for poison in [event(content_hash=E), event(target="other"),
                       event(session="other"), event(chain="bad")]:
            ledger = FakeLedger((poison,))
            with self.subTest(poison=poison), self.assertRaises(GovernanceHold):
                self.recover(self.gate(ledger))
            self.assertEqual(ledger.commit_calls, 0)

    def test_mismatched_receipt_after_would_be_write_is_avoided(self):
        ledger = FakeLedger((event(content_hash=E),))
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger))
        self.assertEqual(ledger.commit_calls, 0)

    def test_missing_event_does_not_fall_back_to_commit(self):
        ledger = FakeLedger(items=())
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger))
        self.assertEqual(ledger.commit_calls, 0)

    def test_nonmonotonic_ledger_offsets_are_rejected(self):
        ledger = FakeLedger((event(event_id="other_1", offset=5),
                             event(event_id="other_2", offset=4)))
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger))
        self.assertEqual(ledger.commit_calls, 0)

    def test_scan_has_a_strict_four_page_limit(self):
        ledger = FakeLedger(tuple(event(event_id=f"other_{i}", offset=i)
                                  for i in range(1, 2002)))
        with self.assertRaises(GovernanceHold):
            self.recover(self.gate(ledger))
        self.assertEqual(ledger.read_calls, 4)
        self.assertEqual(ledger.commit_calls, 0)

    def test_recovered_view_is_narrow(self):
        result = self.recover(self.gate(FakeLedger()))
        self.assertEqual(result.event_offset, 4)
        self.assertFalse(hasattr(result, "delta"))

    def test_physical_training_always_holds(self):
        self.recover(self.gate(FakeLedger()))
        with self.assertRaises(PhysicalTrainingHold):
            TrainingDayNeuroSyncGate.require_physical_training_authority("LoRA")


if __name__ == "__main__":
    unittest.main()
