"""R25 synthetic-only budget, power, heartbeat and lease collision tests."""

import unittest
from dataclasses import replace

from trainingday.r25_budget_lease_policy import (
    BudgetLimits, BudgetObservation, LeaseObservation, assess_budget_run,
)


def limits():
    return BudgetLimits(
        active_seconds=60.0, wall_seconds=70.0, energy_wh=50.0,
        cost_usd=3.0, thermal_c=85.0, peak_gpu_bytes=3_200_000_000,
        required_free_gpu_bytes=3_000_000_000, gpu_reserve_bytes=200_000_000,
        heartbeat_max_age_seconds=5.0, sample_max_gap_seconds=20.0,
        clock_divergence_seconds=2.0, lease_max_age_seconds=12.0,
        lease_max_remaining_seconds=120.0,
    )


def lease():
    return LeaseObservation(
        device_id="Lappy", resource_id="CUDA0", owner_id="r25-owner",
        lease_id="lease-001", observed_unix=1000.0, expires_unix=1050.0,
        active_lease_ids=("lease-001",),
    )


def observation(t=0.0, *, at=None):
    return BudgetObservation(
        monotonic_seconds=100.0 + t, wall_unix=1000.0 + t,
        supervisor_heartbeat_unix=1000.0 + t if at is None else at,
        optimizer_active_seconds=t, peak_gpu_bytes=3_100_000_000,
        free_gpu_bytes=3_300_000_000, temperature_c=70.0,
        energy_wh=0.5 * t, cost_usd=0.02 * t,
        connected=True, in_standby=False,
    )


class R25BudgetLeasePolicyTests(unittest.TestCase):
    def assess(self, *, a=None, b=None, lease_view=None, policy=None,
               complete=True, mode="TRAINING", renewals=()):
        return assess_budget_run(
            policy=limits() if policy is None else policy,
            lease=lease() if lease_view is None else lease_view,
            lease_renewals=renewals,
            observations=(observation(), observation(10)) if a is None and b is None
            else (observation() if a is None else a, observation(10) if b is None else b),
            expected_owner="r25-owner", completed=complete, run_mode=mode,
        )

    def test_clean_synthetic_policy_still_never_authorizes_gpu(self):
        decision = self.assess()
        self.assertEqual(decision["status"], "HOLD")
        self.assertEqual(decision["blockers"], [])
        self.assertTrue(decision["synthetic_constraints_satisfied"])
        self.assertFalse(decision["authorizes_gpu_start"])
        self.assertFalse(decision["host_lease_authenticated"])
        self.assertFalse(decision["valid_equal_time_neural_evidence"])

    def test_completed_training_requires_positive_declared_optimizer_progress(self):
        result = self.assess(a=observation(), b=observation(), complete=True)
        self.assertIn("NO_TRAINING_PROGRESS", result["blockers"])
        self.assertFalse(result["synthetic_constraints_satisfied"])

    def test_zero_update_control_is_explicit_and_not_trained(self):
        control = self.assess(a=observation(), b=observation(), mode="ZERO_UPDATE_CONTROL")
        self.assertEqual(control["blockers"], [])
        self.assertTrue(control["synthetic_constraints_satisfied"])
        self.assertFalse(control["valid_equal_time_neural_evidence"])
        self.assertFalse(control["authorizes_gpu_start"])
        with_update = self.assess(mode="ZERO_UPDATE_CONTROL")
        self.assertIn("ZERO_CONTROL_CONTAINS_OPTIMIZER_TIME", with_update["blockers"])

    def test_unknown_run_mode_is_rejected(self):
        result = self.assess(mode="CORRECT_LABEL_CONTROL")
        self.assertIn("MALFORMED_RUN_MODE", result["blockers"])

    def test_collision_and_stale_owner_are_rejected(self):
        for view, expected in (
            (replace(lease(), active_lease_ids=("lease-001", "lease-other")), "LEASE_COLLISION"),
            (replace(lease(), active_lease_ids=("lease-other",)), "LEASE_COLLISION"),
            (replace(lease(), owner_id="other-owner"), "LEASE_OWNER_MISMATCH"),
            (replace(lease(), observed_unix=985), "LEASE_STALE"),
            (replace(lease(), expires_unix=1009), "LEASE_EXPIRED"),
            (replace(lease(), expires_unix=1300), "LEASE_HORIZON_TOO_LONG"),
        ):
            with self.subTest(view=view):
                self.assertIn(expected, self.assess(lease_view=view)["blockers"])

    def test_connected_tunnel_does_not_equal_resource_lease(self):
        view = replace(lease(), active_lease_ids=())
        decision = self.assess(lease_view=view)
        self.assertIn("LEASE_COLLISION", decision["blockers"])
        self.assertFalse(decision["host_lease_authenticated"])

    def test_lease_fresh_at_start_but_stale_at_finish_fails_closed(self):
        decision = self.assess(b=observation(20))
        self.assertIn("LEASE_STALE", decision["blockers"])

    def test_valid_renewals_cover_31_60_and_531_seconds_synthetically(self):
        for duration in (31, 60, 531):
            with self.subTest(duration=duration):
                times = list(range(0, duration, 20)) + [duration]
                samples = tuple(observation(t) for t in times)
                renewals = tuple(replace(lease(), observed_unix=1000 + t,
                                         expires_unix=1050 + t)
                                 for t in range(20, duration, 20))
                result = assess_budget_run(
                    policy=replace(limits(), active_seconds=600, wall_seconds=600,
                                   energy_wh=500, cost_usd=50,
                                   lease_max_age_seconds=30),
                    lease=lease(), lease_renewals=renewals,
                    observations=samples, expected_owner="r25-owner",
                    completed=True, run_mode="TRAINING",
                )
                self.assertEqual(result["blockers"], [])
                self.assertTrue(result["synthetic_constraints_satisfied"])
                self.assertFalse(result["host_lease_authenticated"])
                self.assertFalse(result["authorizes_gpu_start"])

    def test_long_window_without_renewals_cannot_reuse_initial_lease(self):
        samples = tuple(observation(t) for t in range(0, 61, 20))
        result = assess_budget_run(
            policy=replace(limits(), active_seconds=100, wall_seconds=100,
                           energy_wh=100, cost_usd=10, lease_max_age_seconds=30),
            lease=lease(), lease_renewals=(), observations=samples,
            expected_owner="r25-owner", completed=True, run_mode="TRAINING",
        )
        self.assertIn("LEASE_STALE", result["blockers"])
        self.assertIn("LEASE_EXPIRED", result["blockers"])

    def test_lease_renewal_gap_owner_collision_and_replay_fail_closed(self):
        pairs = (
            ((replace(lease(), observed_unix=1040, expires_unix=1090),),
             "LEASE_RENEWAL_GAP_UNWITNESSED"),
            ((replace(lease(), observed_unix=1020, expires_unix=1070,
                      owner_id="other-owner"),), "LEASE_OWNER_CHANGE_UNAUTHORIZED"),
            ((replace(lease(), observed_unix=1020, expires_unix=1070,
                      lease_id="new-lease", active_lease_ids=("new-lease",)),),
             "LEASE_ID_CHANGE_UNAUTHORIZED"),
            ((replace(lease(), observed_unix=1020, expires_unix=1070,
                      active_lease_ids=("lease-001", "other-lease")),), "LEASE_COLLISION"),
            ((replace(lease(), observed_unix=1000, expires_unix=1070),),
             "NONMONOTONIC_LEASE_OBSERVATION"),
        )
        for renewals, expected in pairs:
            with self.subTest(expected=expected):
                result = self.assess(b=observation(40), renewals=renewals,
                                     policy=replace(limits(), lease_max_age_seconds=30))
                self.assertIn(expected, result["blockers"])

    def test_lease_expired_before_renewal_rejected(self):
        view = replace(lease(), expires_unix=1010)
        renewal = replace(lease(), observed_unix=1020, expires_unix=1070)
        result = self.assess(lease_view=view, renewals=(renewal,),
                             b=observation(40),
                             policy=replace(limits(), lease_max_age_seconds=30))
        self.assertIn("LEASE_EXPIRED_DURING_RUN", result["blockers"])

    def test_future_dated_renewal_is_not_witnessed_for_finished_window(self):
        renewal = replace(lease(), observed_unix=1030, expires_unix=1080)
        result = self.assess(renewals=(renewal,), b=observation(20),
                             policy=replace(limits(), lease_max_age_seconds=30))
        self.assertIn("LEASE_OBSERVATION_AFTER_RUN", result["blockers"])

    def test_wrong_device_or_malformed_lease_is_not_accepted(self):
        self.assertIn("RESOURCE_IDENTITY_UNVERIFIED",
                      self.assess(lease_view=replace(lease(), device_id="VN1"))["blockers"])
        self.assertIn("MALFORMED_LEASE",
                      self.assess(lease_view=replace(lease(), expires_unix=float("nan")))["blockers"])

    def test_lease_and_owner_ids_require_bounded_opaque_characters(self):
        for invalid in ("bad\nlease", "../owner", "white space", "A" * 129):
            with self.subTest(invalid=invalid):
                decision = self.assess(lease_view=replace(lease(), lease_id=invalid,
                                                          active_lease_ids=(invalid,)))
                self.assertIn("MALFORMED_LEASE", decision["blockers"])

    def test_standby_entry_and_disconnect_disqualify_run(self):
        for change, expected in (
            ({"in_standby": True}, "POWER_STANDBY_OBSERVED"),
            ({"connected": False}, "HOST_DISCONNECTED"),
        ):
            with self.subTest(change=change):
                decision = self.assess(b=replace(observation(10), **change))
                self.assertIn(expected, decision["blockers"])

    def test_r18_style_standby_never_retroactively_restores_budget(self):
        long_run = replace(observation(19_813.562), in_standby=True)
        decision = self.assess(
            policy=replace(limits(), active_seconds=531, wall_seconds=531,
                           energy_wh=20000, cost_usd=1000),
            b=long_run,
        )
        self.assertIn("ACTIVE_BUDGET_EXCEEDED", decision["blockers"])
        self.assertIn("WALL_BUDGET_EXCEEDED", decision["blockers"])
        self.assertIn("POWER_STANDBY_OBSERVED", decision["blockers"])
        self.assertFalse(decision["valid_equal_time_neural_evidence"])

    def test_heartbeat_gap_and_stale_heartbeat_fail_closed(self):
        decision = self.assess(b=observation(10, at=1000))
        self.assertIn("SUPERVISOR_HEARTBEAT_STALE", decision["blockers"])
        decision = self.assess(b=observation(40))
        self.assertIn("OBSERVATION_GAP_UNWITNESSED", decision["blockers"])

    def test_wall_clock_discontinuity_is_suspect_not_standby_proof(self):
        decision = self.assess(b=replace(observation(10), wall_unix=1050))
        self.assertIn("CLOCK_DIVERGENCE_SUSPECTED", decision["blockers"])
        self.assertNotIn("POWER_STANDBY_OBSERVED", decision["blockers"])

    def test_small_per_sample_drift_cannot_hide_large_cumulative_divergence(self):
        samples = tuple(replace(observation(0),
                                monotonic_seconds=100 + i * 2.8,
                                wall_unix=1000 + i * 1.4,
                                supervisor_heartbeat_unix=1000 + i * 1.4,
                                optimizer_active_seconds=i * 2.8)
                        for i in range(21))
        result = assess_budget_run(policy=limits(), lease=lease(),
                                   lease_renewals=(), observations=samples,
                                   expected_owner="r25-owner", completed=True,
                                   run_mode="TRAINING")
        self.assertNotIn("CLOCK_DIVERGENCE_SUSPECTED", result["blockers"])
        self.assertIn("CUMULATIVE_CLOCK_DIVERGENCE_SUSPECTED", result["blockers"])

    def test_active_wall_energy_cost_caps_reject_without_subtracting_sleep(self):
        samples = (
            (replace(observation(10), optimizer_active_seconds=61.0), "ACTIVE_BUDGET_EXCEEDED"),
            (replace(observation(10), wall_unix=1080), "WALL_BUDGET_EXCEEDED"),
            (replace(observation(10), energy_wh=55.0), "ENERGY_BUDGET_EXCEEDED"),
            (replace(observation(10), cost_usd=3.01), "COST_BUDGET_EXCEEDED"),
        )
        for second, expected in samples:
            with self.subTest(expected=expected):
                self.assertIn(expected, self.assess(b=second)["blockers"])

    def test_thermal_peak_memory_and_free_headroom_reject(self):
        cases = (
            ({"temperature_c": 90}, "THERMAL_LIMIT_EXCEEDED"),
            ({"peak_gpu_bytes": 3_300_000_000}, "PEAK_MEMORY_LIMIT_EXCEEDED"),
            ({"free_gpu_bytes": 3_000_000_000}, "GPU_HEADROOM_UNVERIFIED"),
        )
        for change, expected in cases:
            with self.subTest(change=change):
                self.assertIn(expected, self.assess(b=replace(observation(10), **change))["blockers"])

    def test_incomplete_window_is_not_valid_evidence(self):
        decision = self.assess(complete=False)
        self.assertIn("INCOMPLETE_RUN", decision["blockers"])
        self.assertFalse(decision["valid_equal_time_neural_evidence"])

    def test_counter_rollback_and_time_reversal_are_invalid(self):
        for first, second in (
            (observation(10), observation()),
            (replace(observation(), optimizer_active_seconds=20), observation(10)),
            (replace(observation(), energy_wh=20), observation(10)),
        ):
            with self.subTest(first=first):
                self.assertIn("NONMONOTONIC_RECEIPT", self.assess(a=first, b=second)["blockers"])

    def test_optimizing_faster_than_monotonic_time_is_rejected(self):
        decision = self.assess(b=replace(observation(10), optimizer_active_seconds=11))
        self.assertIn("IMPOSSIBLE_ACTIVE_TIME_ACCOUNTING", decision["blockers"])

    def test_malformed_nonfinite_or_boolean_numbers_fail_closed(self):
        for sample in (
            replace(observation(10), optimizer_active_seconds=float("nan")),
            replace(observation(10), wall_unix=float("inf")),
            replace(observation(10), peak_gpu_bytes=True),
            replace(observation(10), temperature_c=-1),
            replace(observation(10), wall_unix=10**1000),
        ):
            with self.subTest(sample=sample):
                result = self.assess(b=sample)
                self.assertIn("MALFORMED_OBSERVATION", result["blockers"])
                self.assertFalse(result["authorizes_gpu_start"])

    def test_invalid_policy_and_no_samples_fail_closed(self):
        invalid = replace(limits(), active_seconds=float("inf"))
        self.assertIn("MALFORMED_POLICY", self.assess(policy=invalid)["blockers"])
        decision = assess_budget_run(policy=limits(), lease=lease(), observations=(),
                                     expected_owner="r25-owner", completed=True,
                                     run_mode="TRAINING", lease_renewals=())
        self.assertIn("MISSING_WITNESS_SAMPLES", decision["blockers"])

    def test_caller_cannot_expand_lease_freshness_or_horizon_arbitrarily(self):
        for too_permissive in (
            replace(limits(), lease_max_age_seconds=1e308),
            replace(limits(), lease_max_remaining_seconds=1e308),
            replace(limits(), heartbeat_max_age_seconds=1e308),
            replace(limits(), sample_max_gap_seconds=1e308),
        ):
            with self.subTest(policy=too_permissive):
                self.assertIn("MALFORMED_POLICY",
                              self.assess(policy=too_permissive)["blockers"])

    def test_counterfeit_authentication_and_receipt_claims_cannot_authorize(self):
        result = self.assess()
        self.assertNotIn("trusted", result)
        self.assertFalse(result["authorizes_gpu_start"])
        self.assertFalse(result["host_lease_authenticated"])
        self.assertEqual(result["evidence_class"], "SYNTHETIC_POLICY_DECLARATIONS_ONLY")


if __name__ == "__main__":
    unittest.main()
