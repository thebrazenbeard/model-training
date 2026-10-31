"""CPU-only Windows custody contract tests, no Windows or pytest required."""

from dataclasses import FrozenInstanceError, asdict, replace
import unittest

from trainingday.r140_windows_custody_contract import (
    WindowsCustodyIntent,
    WindowsCustodyObservation,
    assess_windows_custody_preflight,
)


def _case():
    ids = dict(
        host_instance="a" * 32,
        volume_guid="{01234567-89ab-cdef-0123-456789abcdef}",
        parent_file_id="b" * 32,
        stage_file_id="c" * 32,
        job_token="d" * 64,
        lease_epoch=17,
        dispatch_nonce="e" * 64,
        target_basename="checkpoint-19",
    )
    flags = dict(
        pinned_ancestor_handles=True,
        parent_and_stage_file_ids_held=True,
        reparse_points_excluded=True,
        same_volume_nonreplacing_publish=True,
        job_kill_on_close_enforced=True,
        job_breakaway_prohibited=True,
        all_descendant_processes_enrolled=True,
        prior_job_termination_acknowledged=True,
        pending_writes_drained=True,
        delayed_dispatch_replay_excluded=True,
        lease_owner_acked=True,
        external_nonce_witness_acknowledged=True,
    )
    return WindowsCustodyIntent(**ids), WindowsCustodyObservation(**ids, **flags)


class WindowsCustodyContractTests(unittest.TestCase):
    def assert_no_authority(self, decision):
        self.assertIs(decision.authorizes_training, False)
        self.assertIs(decision.authorizes_resume, False)
        self.assertIs(decision.authorizes_filesystem_mutation, False)
        self.assertIs(decision.physical_windows_tested, False)
        self.assertIs(decision.authoritative_host_binding_verified, False)

    def test_perfect_synthetic_claim_never_authorizes(self):
        intent, report = _case()
        result = assess_windows_custody_preflight(intent, report)
        self.assertEqual(result.status, "SYNTHETIC_CONDITIONS_MET_NOT_AUTHORIZED")
        self.assertTrue(result.synthetic_conditions_met)
        self.assertEqual(result.reasons, ())
        self.assert_no_authority(result)

    def test_all_identity_mismatches_hold(self):
        cases = [
            ("host_instance", "f" * 32),
            ("volume_guid", "{ffffffff-ffff-ffff-ffff-ffffffffffff}"),
            ("parent_file_id", "f" * 32),
            ("stage_file_id", "f" * 32),
            ("job_token", "f" * 64),
            ("lease_epoch", 18),
            ("dispatch_nonce", "f" * 64),
            ("target_basename", "checkpoint-20"),
        ]
        for name, value in cases:
            with self.subTest(field=name):
                intent, report = _case()
                result = assess_windows_custody_preflight(
                    intent, replace(report, **{name: value}))
                self.assertIn("BINDING_MISMATCH_" + name.upper(), result.reasons)
                self.assertFalse(result.synthetic_conditions_met)
                self.assert_no_authority(result)

    def test_missing_windows_and_executor_controls_hold(self):
        flags = [
            "pinned_ancestor_handles", "parent_and_stage_file_ids_held",
            "reparse_points_excluded", "same_volume_nonreplacing_publish",
            "job_kill_on_close_enforced", "job_breakaway_prohibited",
            "all_descendant_processes_enrolled", "prior_job_termination_acknowledged",
            "pending_writes_drained", "delayed_dispatch_replay_excluded",
            "lease_owner_acked", "external_nonce_witness_acknowledged",
        ]
        for name in flags:
            with self.subTest(signal=name):
                intent, report = _case()
                result = assess_windows_custody_preflight(
                    intent, replace(report, **{name: False}))
                self.assertIn("UNPROVEN_" + name.upper(), result.reasons)
                self.assertFalse(result.synthetic_conditions_met)
                self.assert_no_authority(result)

    def test_truthy_reparse_claims_do_not_pass(self):
        for value in [1, "true", None]:
            with self.subTest(value=value):
                intent, report = _case()
                result = assess_windows_custody_preflight(
                    intent, replace(report, reparse_points_excluded=value))
                self.assertIn("UNPROVEN_REPARSE_POINTS_EXCLUDED", result.reasons)
                self.assert_no_authority(result)

    def test_invalid_canonical_identifiers_and_windows_names_hold(self):
        cases = [
            ("host_instance", "lappy"),
            ("host_instance", "A" * 32),
            ("volume_guid", "C:"),
            ("volume_guid", "\\\\?\\Volume{01234567-89ab-cdef-0123-456789abcdef}"),
            ("volume_guid", "{01234567-89AB-cdef-0123-456789abcdef}"),
            ("parent_file_id", "b" * 31),
            ("job_token", "d" * 63),
            ("lease_epoch", True),
            ("lease_epoch", 0),
            ("dispatch_nonce", "e" * 65),
            ("target_basename", "..\\checkpoint-19"),
            ("target_basename", "checkpoint-19:$DATA"),
            ("target_basename", "checkpoint-0"),
            ("target_basename", "checkpoint-2147483648"),
            ("target_basename", "checkpoint-01"),
            ("target_basename", "checkpoint-19 "),
        ]
        for name, value in cases:
            with self.subTest(field=name, value=value):
                intent, report = _case()
                result = assess_windows_custody_preflight(
                    replace(intent, **{name: value}), report)
                self.assertTrue(any(s.startswith("INVALID_") for s in result.reasons))
                self.assert_no_authority(result)

    def test_source_accepts_no_path_and_values_are_frozen(self):
        intent, report = _case()
        with self.assertRaises(TypeError):
            WindowsCustodyIntent(**{**asdict(intent), "stage_path": r"C:\Temp\stage"})
        with self.assertRaises(FrozenInstanceError):
            report.reparse_points_excluded = False

    def test_forged_untyped_objects_hold(self):
        for intent, report in [(None, None), ("intent", "claims")]:
            result = assess_windows_custody_preflight(intent, report)
            self.assertEqual(result.status, "HOLD_INVALID_REQUEST")
            self.assert_no_authority(result)


if __name__ == "__main__":
    unittest.main()
