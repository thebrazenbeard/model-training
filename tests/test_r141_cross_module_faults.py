"""Cross-component *synthetic* failure checks for TrainingDay custody R140.

These checks deliberately combine independently developed source prototypes.
None imports an actual host authority, starts a model, or grants training/resume.
The injected backends and observations are untrusted test doubles.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from trainingday import r140_host_custody_service as service
from trainingday import r140_linux_fd_custody as linux
from trainingday import r140_monotonic_authority as monotonic
from trainingday import r140_qwen_attestation as qwen
from trainingday import r140_windows_custody_contract as windows


def _request() -> service.CustodyRequest:
    return service.CustodyRequest(
        request_id="request-r141", caller_principal="trainer-r141",
        host_id="host-r141", lease_id="lease-r141", run_id="run-r141",
        run_contract_sha256="a" * 64, source_commit_sha256="b" * 64,
        checkpoint_weights_sha256="c" * 64,
        checkpoint_state_sha256="d" * 64,
        step=9, checkpoint_object_id="checkpoint-9", authority_epoch=4,
    )


class _SyntheticBackend:
    """Caller-owned fake; neither authenticates principals nor owns the host."""

    def __init__(self, *, times=(100, 100), expire_at=101,
                 ambiguous_commit=False) -> None:
        self.times = iter(times)
        self.expire_at = expire_at
        self.ambiguous_commit = ambiguous_commit
        self.commits = 0
        self.abandons: list[str] = []

    def host_time(self):
        return next(self.times)

    def reserve(self, request, digest):
        return service.HostReservation(
            request_sha256=digest, reservation_id="reservation-r141",
            caller_principal=request.caller_principal, host_id=request.host_id,
            lease_id=request.lease_id, authority_epoch=request.authority_epoch,
            expires_at=self.expire_at,
        )

    def snapshot(self, request, reservation):
        return service.HostSnapshot(
            request_sha256=request.digest(),
            reservation_id=reservation.reservation_id,
            source_commit_sha256=request.source_commit_sha256,
            checkpoint_weights_sha256=request.checkpoint_weights_sha256,
            checkpoint_state_sha256=request.checkpoint_state_sha256,
            object_id=request.checkpoint_object_id, step=request.step,
            descriptor_bound=True,
        )

    def commit(self, request, reservation, snapshot):
        self.commits += 1
        if self.ambiguous_commit:
            raise TimeoutError("synthetic response lost after possible commit")
        return service.HostCommit(
            request_sha256=request.digest(),
            reservation_id=reservation.reservation_id,
            snapshot_sha256=snapshot.digest(), receipt_id="receipt-r141",
            authority_epoch=request.authority_epoch, monotonic_sequence=1,
            state="HOST_BACKEND_COMMITTED_UNVERIFIED",
        )

    def abandon(self, reservation, reason):
        self.abandons.append(reason)


def _candidate(*, expected_epoch=0, expected_head=monotonic.GENESIS_HEAD,
               nonce="nonce-r141") -> monotonic.Candidate:
    return monotonic.Candidate(
        run_id="run-r141", host_principal="host-r141", step=9,
        checkpoint_sha256="e" * 64, nonce=nonce,
        expected_epoch=expected_epoch, expected_head=expected_head,
    )


def _windows_case() -> tuple[windows.WindowsCustodyIntent,
                             windows.WindowsCustodyObservation]:
    binding = dict(
        host_instance="a" * 32,
        volume_guid="{01234567-89ab-cdef-0123-456789abcdef}",
        parent_file_id="b" * 32, stage_file_id="c" * 32,
        job_token="d" * 64, lease_epoch=7,
        dispatch_nonce="e" * 64, target_basename="checkpoint-9",
    )
    controls = {
        name: True for name in (
            "pinned_ancestor_handles", "parent_and_stage_file_ids_held",
            "reparse_points_excluded", "same_volume_nonreplacing_publish",
            "job_kill_on_close_enforced", "job_breakaway_prohibited",
            "all_descendant_processes_enrolled",
            "prior_job_termination_acknowledged", "pending_writes_drained",
            "delayed_dispatch_replay_excluded", "lease_owner_acked",
            "external_nonce_witness_acknowledged",
        )
    }
    return (windows.WindowsCustodyIntent(**binding),
            windows.WindowsCustodyObservation(**binding, **controls))


def _stage_linux(parent: Path) -> str:
    stage = ".checkpoint-9-staging-r141"
    root = parent / stage
    root.mkdir()
    weights = b"synthetic-R141-training-bytes-not-real-model"
    (root / "lora-trainables.safetensors").write_bytes(weights)
    (root / "state.json").write_text(json.dumps({
        "schema": "R18_MEZO_STATE_V1",
        "status": "CHECKPOINT_BYTES_PINNED_NOT_RELOADED",
        "completed_steps": 9,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "run_contract_sha256": "a" * 64,
        "optimizer_window_seconds": 2,
        "counters": {"forward_passes": 3},
    }), encoding="utf-8")
    return stage


class R141CrossModuleFaults(unittest.TestCase):
    def assert_no_authority(self, value):
        self.assertIs(value.authorizes_resume, False)
        self.assertIs(value.authorizes_training, False)

    def test_custody_success_plus_monotonic_success_remain_two_source_claims(self):
        receipt = service.HostCustodyService(_SyntheticBackend()).execute(_request())
        witness = monotonic.assess_transition(_candidate(),
                                              monotonic.DemoVolatileWitness())
        self.assertEqual(receipt.status, "SOURCE_ONLY_HOST_AUTHORITY_UNVERIFIED")
        self.assertTrue(witness.synthetic_transition_observed)
        self.assert_no_authority(receipt)
        self.assert_no_authority(witness)
        self.assertFalse(witness.durable_antirollback_verified)
        self.assertFalse(receipt.physical_reload_verified)

    def test_lease_expires_after_snapshot_before_commit(self):
        backend = _SyntheticBackend(times=(100, 102), expire_at=101)
        with self.assertRaises(service.CustodyServiceError) as caught:
            service.HostCustodyService(backend).execute(_request())
        self.assertEqual(caught.exception.reason, "SNAPSHOT_REJECTED")
        self.assertEqual(backend.commits, 0)
        self.assertIn("SNAPSHOT_REJECTED", backend.abandons)

    def test_ambiguous_custody_commit_is_not_a_resumable_receipt(self):
        backend = _SyntheticBackend(ambiguous_commit=True)
        with self.assertRaises(service.CustodyServiceError) as caught:
            service.HostCustodyService(backend).execute(_request())
        self.assertEqual(caught.exception.reason, "COMMIT_OUTCOME_UNKNOWN")
        self.assertTrue(caught.exception.ambiguous_commit)
        self.assertEqual(backend.commits, 1)
        self.assertIn("COMMIT_OUTCOME_UNKNOWN", backend.abandons)
        witness = monotonic.assess_transition(_candidate(), None)
        self.assertEqual(witness.reason, "INDEPENDENT_WITNESS_UNAVAILABLE")
        self.assert_no_authority(witness)

    def test_monotonic_witness_commits_but_reply_times_out(self):
        underlying = monotonic.DemoVolatileWitness()

        class _LostReply:
            def compare_and_append(self, candidate):
                underlying.compare_and_append(candidate)
                raise TimeoutError("synthetic witness response lost")

        first = monotonic.assess_transition(_candidate(), _LostReply())
        self.assertEqual(first.reason, "WITNESS_OUTCOME_UNKNOWN_HOLD")
        self.assert_no_authority(first)
        self.assertEqual(underlying.demo_read("run-r141").epoch, 1)
        retry = monotonic.assess_transition(_candidate(), underlying)
        self.assertEqual(retry.reason, "NONCE_REPLAY")
        self.assert_no_authority(retry)

    def test_restored_application_snapshot_cannot_rewind_witness(self):
        witness = monotonic.DemoVolatileWitness()
        good = monotonic.assess_transition(_candidate(), witness)
        self.assertTrue(good.synthetic_transition_observed)
        stale = monotonic.assess_transition(
            _candidate(nonce="nonce-newer"), witness)
        self.assertEqual(stale.reason, "STALE_OR_DIVERGENT_SNAPSHOT")
        self.assert_no_authority(stale)

    def test_windows_perfect_caller_report_still_has_no_authority(self):
        intent, claimed = _windows_case()
        result = windows.assess_windows_custody_preflight(intent, claimed)
        self.assertTrue(result.synthetic_conditions_met)
        self.assert_no_authority(result)
        self.assertFalse(result.authoritative_host_binding_verified)
        self.assertFalse(result.physical_windows_tested)

    def test_windows_pending_writer_or_replay_negates_synthetic_admission(self):
        intent, claimed = _windows_case()
        for missing in ("pending_writes_drained", "delayed_dispatch_replay_excluded",
                        "all_descendant_processes_enrolled",
                        "prior_job_termination_acknowledged"):
            with self.subTest(missing=missing):
                altered = replace(claimed, **{missing: False})
                result = windows.assess_windows_custody_preflight(intent, altered)
                self.assertIn("UNPROVEN_" + missing.upper(), result.reasons)
                self.assertFalse(result.synthetic_conditions_met)
                self.assert_no_authority(result)

    def test_synthetic_custody_receipt_cannot_replace_qwen_source_pins(self):
        """A receipt-shaped object is no authority for an unpinned model load."""
        receipt = service.HostCustodyService(_SyntheticBackend()).execute(_request())
        self.assert_no_authority(receipt)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            # The missing independently expected model/config/tokenizer hashes
            # must reject before any attempt to load weights or invoke CUDA.
            with self.assertRaises(qwen.SnapshotAttestationError):
                qwen.audit_snapshot(
                    parent / "base-binding.json",
                    expected_binding_sha256=receipt.request_sha256,
                    expected_config_sha256="",
                    expected_tokenizer_sha256="",
                    source_root=parent,
                )

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-specific prototype")
    def test_linux_rename_timeout_after_publish_never_grants_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = _stage_linux(root)
            fd = linux.open_linux_directory_no_symlinks(root)
            real_rename = linux._rename_noreplace

            def rename_then_timeout(parent_fd, old, new):
                real_rename(parent_fd, old, new)
                raise TimeoutError("synthetic crash after rename, before reply")

            try:
                with patch.object(linux, "_rename_noreplace", rename_then_timeout):
                    with self.assertRaisesRegex(
                            linux.LinuxCustodyError, "POST_RENAME_CUSTODY_UNKNOWN"):
                        linux.publish_prepared_linux(fd, stage_name=stage, step=9)
            finally:
                os.close(fd)
            self.assertTrue((root / "checkpoint-9").is_dir())
            hold = monotonic.assess_transition(_candidate(), None)
            self.assertEqual(hold.reason, "INDEPENDENT_WITNESS_UNAVAILABLE")
            self.assert_no_authority(hold)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-specific prototype")
    def test_linux_same_bytes_replaced_inode_after_rename_holds_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = _stage_linux(root)
            fd = linux.open_linux_directory_no_symlinks(root)
            original = linux._readback
            calls = 0

            def swap_same_bytes(parent_fd, expected, object_ids):
                nonlocal calls
                calls += 1
                if calls == 2:
                    child = os.open("state.json", os.O_RDONLY | os.O_NOFOLLOW,
                                    dir_fd=parent_fd)
                    try:
                        raw = os.read(child, 1024 * 1024)
                    finally:
                        os.close(child)
                    os.unlink("state.json", dir_fd=parent_fd)
                    replacement = os.open("state.json", os.O_CREAT | os.O_EXCL |
                                          os.O_WRONLY, 0o600, dir_fd=parent_fd)
                    try:
                        self.assertEqual(os.write(replacement, raw), len(raw))
                    finally:
                        os.close(replacement)
                return original(parent_fd, expected, object_ids)

            try:
                with patch.object(linux, "_readback", swap_same_bytes):
                    with self.assertRaisesRegex(
                            linux.LinuxCustodyError, "POST_RENAME_CUSTODY_UNKNOWN"):
                        linux.publish_prepared_linux(fd, stage_name=stage, step=9)
            finally:
                os.close(fd)
            self.assertTrue((root / "checkpoint-9").is_dir())


if __name__ == "__main__":
    unittest.main()
