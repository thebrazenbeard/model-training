"""Synthetic protocol checks; no host authority or checkpoint reload."""

from __future__ import annotations

import dataclasses
import unittest

from trainingday import r140_host_custody_service as custody


def request(**changes: object) -> custody.CustodyRequest:
    base = custody.CustodyRequest(
        request_id="request-0001", caller_principal="trainer-001",
        host_id="host-alpha", lease_id="lease-0001", run_id="run-00001",
        run_contract_sha256="a" * 64,
        source_commit_sha256="b" * 64,
        checkpoint_weights_sha256="c" * 64,
        checkpoint_state_sha256="d" * 64,
        step=5, checkpoint_object_id="checkpoint-5", authority_epoch=7,
    )
    return dataclasses.replace(base, **changes)


class SyntheticHostBackend:
    """Deliberately caller-controlled; proves no real host authorization."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.commits = 0
        self.reserves = 0
        self.abandoned: list[str] = []
        self.reserve_override = None
        self.snapshot_override = None
        self.commit_override = None
        self.host_clock = 100
        self.expire_during_snapshot = False

    def host_time(self):
        return self.host_clock

    def reserve(self, req, sha):
        self.reserves += 1
        if sha in self.seen:
            raise ValueError("synthetic nonce replay")
        self.seen.add(sha)
        value = custody.HostReservation(
            request_sha256=sha, reservation_id="reservation-1",
            caller_principal=req.caller_principal, host_id=req.host_id,
            lease_id=req.lease_id, authority_epoch=req.authority_epoch,
            expires_at=101,
        )
        return self.reserve_override or value

    def snapshot(self, req, reservation):
        if self.expire_during_snapshot:
            self.host_clock = 102
        value = custody.HostSnapshot(
            request_sha256=req.digest(),
            reservation_id=reservation.reservation_id,
            source_commit_sha256=req.source_commit_sha256,
            checkpoint_weights_sha256=req.checkpoint_weights_sha256,
            checkpoint_state_sha256=req.checkpoint_state_sha256,
            object_id=req.checkpoint_object_id, step=req.step,
            descriptor_bound=True,
        )
        return self.snapshot_override or value

    def commit(self, req, reservation, snapshot):
        self.commits += 1
        if isinstance(self.commit_override, Exception):
            raise self.commit_override
        value = custody.HostCommit(
            request_sha256=req.digest(),
            reservation_id=reservation.reservation_id,
            snapshot_sha256=snapshot.digest(),
            receipt_id="receipt-001",
            authority_epoch=req.authority_epoch,
            monotonic_sequence=50,
            state="HOST_BACKEND_COMMITTED_UNVERIFIED",
        )
        return self.commit_override or value

    def abandon(self, reservation, reason):
        self.abandoned.append(reason)


class R140HostCustodyTests(unittest.TestCase):
    def setUp(self):
        self.host = SyntheticHostBackend()
        self.service = custody.HostCustodyService(self.host)
        self.req = request()

    def assert_rejected(self, reason, req=None):
        with self.assertRaises(custody.CustodyServiceError) as captured:
            self.service.execute(req if req is not None else self.req)
        self.assertEqual(captured.exception.reason, reason)
        return captured.exception

    def test_correlation_only_never_authorizes_training_resume(self):
        receipt = self.service.execute(self.req)
        self.assertEqual(receipt.status, "SOURCE_ONLY_HOST_AUTHORITY_UNVERIFIED")
        self.assertEqual(receipt.request_sha256, self.req.digest())
        self.assertFalse(receipt.authorizes_resume)
        self.assertFalse(receipt.authorizes_training)
        self.assertFalse(receipt.physical_durability_verified)
        self.assertFalse(receipt.physical_reload_verified)
        self.assertEqual(self.host.commits, 1)

    def test_duplicate_request_fail_closed_without_second_commit(self):
        self.service.execute(self.req)
        self.assert_rejected("RESERVATION_REJECTED")
        self.assertEqual(self.host.commits, 1)

    def test_caller_or_lease_change_makes_new_digest(self):
        a = request()
        self.assertNotEqual(a.digest(), request(caller_principal="trainer-002").digest())
        self.assertNotEqual(a.digest(), request(lease_id="lease-0002").digest())
        self.assertNotEqual(a.digest(), request(host_id="host-bravo").digest())
        self.assertNotEqual(a.digest(), request(authority_epoch=8).digest())
        self.assertNotEqual(a.digest(), request(checkpoint_object_id="checkpoint-6").digest())

    def test_rejects_boolean_step_and_epoch_and_bad_hash(self):
        for modification in ({"step": True}, {"authority_epoch": False},
                             {"authority_epoch": 10**5000},
                             {"authority_epoch": 2**63 - 1},
                             {"authority_epoch": -1},
                             {"source_commit_sha256": "B" * 64},
                             {"checkpoint_weights_sha256": "a" * 63}):
            self.assert_rejected("MALFORMED_REQUEST_DIGEST" if "sha256" in next(iter(modification))
                                 else "MALFORMED_REQUEST_COUNTER", request(**modification))
        self.assertEqual(self.host.reserves, 0)

    def test_rejects_malformed_identifiers_before_backend(self):
        self.assert_rejected("MALFORMED_REQUEST_IDENTIFIER", request(lease_id="x"))
        self.assertEqual(self.host.reserves, 0)

    def test_rejects_wrong_or_expired_reservation(self):
        original = self.host.reserve(self.req, self.req.digest())
        self.host.seen.clear()
        self.host.reserve_override = dataclasses.replace(original, lease_id="lease-other")
        self.assert_rejected("RESERVATION_BINDING_MISMATCH")
        self.assertEqual(self.host.commits, 0)
        self.assertEqual(self.host.abandoned, [])  # never cancel foreign reservation

    def test_expired_reservation_refused(self):
        original = self.host.reserve(self.req, self.req.digest())
        self.host.seen.clear()
        self.host.reserve_override = dataclasses.replace(original, expires_at=100)
        self.assert_rejected("RESERVATION_BINDING_MISMATCH")
        self.assertEqual(self.host.commits, 0)

    def test_slow_snapshot_crossing_expiry_holds_before_commit(self):
        self.host.expire_during_snapshot = True
        self.assert_rejected("SNAPSHOT_REJECTED")
        self.assertEqual(self.host.commits, 0)
        self.assertEqual(self.host.abandoned, ["SNAPSHOT_REJECTED"])

    def test_missing_host_clock_fails_closed_before_reservation(self):
        self.host.host_clock = True
        self.assert_rejected("AUTHORITY_CLOCK_UNAVAILABLE")
        self.assertEqual(self.host.reserves, 0)

    def test_rejects_snapshot_other_object(self):
        original = self.host.snapshot(
            self.req, custody.HostReservation(self.req.digest(), "reservation-1",
            "trainer-001", "host-alpha", "lease-0001", 7, 101),
        )
        self.host.snapshot_override = dataclasses.replace(original, object_id="checkpoint-6")
        self.assert_rejected("SNAPSHOT_REJECTED")
        self.assertEqual(self.host.commits, 0)
        self.assertEqual(self.host.abandoned[-1], "SNAPSHOT_REJECTED")

    def test_rejects_claimed_non_descriptor_snapshot(self):
        original = self.host.snapshot(
            self.req, custody.HostReservation(self.req.digest(), "reservation-1",
            "trainer-001", "host-alpha", "lease-0001", 7, 101),
        )
        self.host.snapshot_override = dataclasses.replace(original, descriptor_bound=False)
        self.assert_rejected("SNAPSHOT_REJECTED")
        self.assertEqual(self.host.commits, 0)

    def test_commit_exception_is_unknown_and_not_retried(self):
        self.host.commit_override = OSError("simulated lost reply after commit")
        failure = self.assert_rejected("COMMIT_OUTCOME_UNKNOWN")
        self.assertTrue(failure.ambiguous_commit)
        self.assertEqual(self.host.commits, 1)
        self.assertEqual(self.host.abandoned[-1], "COMMIT_OUTCOME_UNKNOWN")

    def test_mismatched_commit_claim_is_unknown(self):
        reservation = self.host.reserve(self.req, self.req.digest())
        snapshot = self.host.snapshot(self.req, reservation)
        self.host.seen.clear()
        self.host.commit_override = dataclasses.replace(
            self.host.commit(self.req, reservation, snapshot),
            monotonic_sequence=0,
        )
        self.host.commits = 0
        failure = self.assert_rejected("COMMIT_RECEIPT_MISMATCH")
        self.assertTrue(failure.ambiguous_commit)
        self.assertEqual(self.host.commits, 1)

    def test_direct_contract_guard_rejects_cross_request_rebind(self):
        reservation = self.host.reserve(self.req, self.req.digest())
        snapshot = self.host.snapshot(self.req, reservation)
        commit = self.host.commit(self.req, reservation, snapshot)
        with self.assertRaises(custody.CustodyServiceError) as captured:
            custody.correlate_contract(
                request(caller_principal="trainer-002"),
                reservation, snapshot, commit, now=100,
            )
        self.assertEqual(captured.exception.reason, "RESERVATION_BINDING_MISMATCH")


if __name__ == "__main__":
    unittest.main()
