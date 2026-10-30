"""Public synthetic, non-authorizing continuous-learning policy regressions."""

from __future__ import annotations

import dataclasses
import unittest

from trainingday.r27_continuous_learning_policy import (
    ConsentClaim,
    CycleEvidence,
    CycleEvent,
    CycleRequest,
    FrozenBaseline,
    PolicyError,
    assess_cycle,
    baseline_digest,
)


def sha(ch: str) -> str:
    return ch * 64


def fixture():
    baseline = FrozenBaseline(
        base_sha256=sha("a"), adapter_sha256=sha("b"),
        checkpoint_sha256=sha("c"), evidence_sha256=sha("d"),
        rights_sha256=sha("e"),
    )
    request = CycleRequest(
        cycle_id="r27-cycle-001", requester_id="vera-local",
        objective_id="public-program-reasoning", source_sha256=sha("f"),
        baseline_sha256=baseline_digest(baseline),
        proposed_adapter_sha256=sha("1"),
        proposed_checkpoint_sha256=sha("2"),
    )
    claim = ConsentClaim(
        cycle_id=request.cycle_id, issuer_id="patrick-operator",
        issuer_role="HUMAN_OWNER_CLAIMED", decision="APPROVE_CLAIMED",
        baseline_sha256=request.baseline_sha256,
        source_sha256=request.source_sha256,
        objective_id=request.objective_id,
        proposed_adapter_sha256=request.proposed_adapter_sha256,
        proposed_checkpoint_sha256=request.proposed_checkpoint_sha256,
        issued_unix=100, expires_unix=200, nonce="unique-grant-001",
        max_active_seconds=60, max_cost_cents=1000,
        max_energy_mwh=30000,
    )
    evidence = CycleEvidence(
        cycle_id=request.cycle_id,
        previous_adapter_sha256=baseline.adapter_sha256,
        candidate_adapter_sha256=request.proposed_adapter_sha256,
        candidate_checkpoint_sha256=request.proposed_checkpoint_sha256,
        rights_sha256=baseline.rights_sha256,
        public_before_ppm=650000, public_after_ppm=750000,
        public_regressions=0, unseen_regressions=0,
        frozen_control_present=True, zero_control_present=True,
        corrupted_control_present=True,
        active_seconds=35, cost_cents=700, energy_mwh=11000,
        actual_tensor_bytes_independently_verified=False,
    )
    events = (
        CycleEvent(seq=1, kind="REQUEST", actor_id=request.requester_id, unix=101),
        CycleEvent(seq=2, kind="FREEZE", actor_id="evidence-custodian", unix=102),
        CycleEvent(seq=3, kind="CONSENT_CLAIM", actor_id=claim.issuer_id, unix=103),
        CycleEvent(seq=4, kind="PUBLIC_REVIEW", actor_id="public-reviewer", unix=120),
    )
    return baseline, request, claim, evidence, events


def check(*, baseline=None, request=None, claim=None, evidence=None, events=None,
          now=130):
    a, b, c, d, e = fixture()
    return assess_cycle(
        baseline=baseline if baseline is not None else a,
        request=request if request is not None else b,
        consent=claim if claim is not None else c,
        evidence=evidence if evidence is not None else d,
        events=events if events is not None else e,
        now_unix=now,
    )


class ContinuousLearningPolicyTests(unittest.TestCase):
    def test_source_complete_never_grants_weight_update_or_promotion(self):
        result = check()
        self.assertEqual(result["status"], "HOLD")
        self.assertTrue(result["synthetic_source_conditions_satisfied"])
        self.assertFalse(result["consent_independently_authenticated"])
        self.assertFalse(result["gpu_start_authorized"])
        self.assertFalse(result["continuous_weight_update_authorized"])
        self.assertFalse(result["model_promotion_authorized"])
        self.assertFalse(result["physical_rollback_proven"])

    def test_requester_cannot_self_approve(self):
        _, request, claim, _, _ = fixture()
        result = check(claim=dataclasses.replace(claim, issuer_id=request.requester_id))
        self.assertIn("CONSENT_PRINCIPAL_NOT_SEPARATE", result["blockers"])

    def test_consent_must_be_scoped_to_frozen_snapshot_and_source(self):
        _, _, claim, _, _ = fixture()
        self.assertIn("CONSENT_SCOPE_MISMATCH", check(claim=dataclasses.replace(
            claim, baseline_sha256=sha("0")))["blockers"])
        self.assertIn("CONSENT_SCOPE_MISMATCH", check(claim=dataclasses.replace(
            claim, source_sha256=sha("0")))["blockers"])

    def test_consent_cannot_be_reused_for_different_objective_or_adapter(self):
        _, request, _, _, _ = fixture()
        for field, value in (("objective_id", "private-empathy-experiment"),
                             ("proposed_adapter_sha256", sha("9")),
                             ("proposed_checkpoint_sha256", sha("9"))):
            with self.subTest(field=field):
                # The original claim is unchanged; only the requested operation moves.
                self.assertIn("CONSENT_SCOPE_MISMATCH", check(request=dataclasses.replace(
                    request, **{field: value}))["blockers"])

    def test_consent_expiration_blocks_even_if_all_other_receipts_match(self):
        self.assertIn("CONSENT_EXPIRED", check(now=201)["blockers"])

    def test_late_or_backdated_consent_event_blocks(self):
        _, _, _, _, ev = fixture()
        self.assertIn("CONSENT_NOT_CURRENT", check(events=tuple(
            dataclasses.replace(e, unix=99) if e.kind == "CONSENT_CLAIM" else e for e in ev
        ))["blockers"])

    def test_revocation_after_review_still_blocks_and_retains_previous_artifacts(self):
        _, _, _, _, ev = fixture()
        result = check(events=ev + (
            CycleEvent(seq=5, kind="RIGHTS_REVOKED", actor_id="rights-custodian", unix=125),
        ))
        self.assertIn("RIGHTS_REVOKED", result["blockers"])
        self.assertTrue(result["previous_snapshot_preserved"])
        self.assertFalse(result["unlearning_verified"])

    def test_revocation_before_review_blocks_later_review(self):
        _, _, _, _, ev = fixture()
        damaged = ev[:3] + (
            CycleEvent(seq=4, kind="RIGHTS_REVOKED", actor_id="rights-custodian", unix=110),
            CycleEvent(seq=5, kind="PUBLIC_REVIEW", actor_id="public-reviewer", unix=120),
        )
        self.assertIn("EVENT_AFTER_REVOCATION", check(events=damaged)["blockers"])

    def test_privacy_taint_and_rights_review_required_halt_cycle(self):
        _, _, _, e, ev = fixture()
        for status in ("REVIEW_REQUIRED", "REVOKED", "PRIVACY_TAINTED"):
            with self.subTest(status=status):
                self.assertIn("RIGHTS_NOT_CLEAR", check(evidence=dataclasses.replace(
                    e, rights_status=status))["blockers"])
        result = check(events=ev + (CycleEvent(
            seq=5, kind="PRIVACY_TAINTED", actor_id="rights-custodian", unix=125),))
        self.assertIn("PRIVACY_TAINTED", result["blockers"])
        self.assertFalse(result["model_promotion_authorized"])

    def test_invalid_rights_status_fails_closed(self):
        _, _, _, e, _ = fixture()
        self.assertRaises(PolicyError, check,
                          evidence=dataclasses.replace(e, rights_status="VERIFIED_HUMAN_GRANTED"))

    def test_public_and_unseen_regressions_block(self):
        _, _, _, e, _ = fixture()
        self.assertIn("PUBLIC_REGRESSION", check(evidence=dataclasses.replace(
            e, public_regressions=1))["blockers"])
        self.assertIn("UNSEEN_REGRESSION", check(evidence=dataclasses.replace(
            e, unseen_regressions=1))["blockers"])

    def test_declared_gain_not_a_physical_tensor_attestation(self):
        _, _, _, e, _ = fixture()
        # Even a claim of independent bytes is only caller input.
        result = check(evidence=dataclasses.replace(
            e, actual_tensor_bytes_independently_verified=True))
        self.assertFalse(result["tensor_bytes_independently_verified"])
        self.assertFalse(result["continuous_weight_update_authorized"])

    def test_missing_any_control_blocks(self):
        _, _, _, e, _ = fixture()
        for field in ("frozen_control_present", "zero_control_present", "corrupted_control_present"):
            with self.subTest(field=field):
                self.assertIn("CONTROL_INCOMPLETE", check(evidence=dataclasses.replace(
                    e, **{field: False}))["blockers"])

    def test_budget_overrun_blocks(self):
        _, _, _, e, _ = fixture()
        for field, value in (("active_seconds", 61), ("cost_cents", 1001),
                             ("energy_mwh", 30001)):
            with self.subTest(field=field):
                self.assertIn("DECLARED_BUDGET_OVERRUN", check(
                    evidence=dataclasses.replace(e, **{field: value}))["blockers"])

    def test_noop_adapter_and_wrong_checkpoint_block(self):
        b, _, _, e, _ = fixture()
        self.assertIn("NO_DECLARED_ADAPTER_CHANGE", check(evidence=dataclasses.replace(
            e, candidate_adapter_sha256=b.adapter_sha256))["blockers"])
        self.assertIn("CANDIDATE_IDENTITY_MISMATCH", check(evidence=dataclasses.replace(
            e, candidate_checkpoint_sha256=sha("9")))["blockers"])

    def test_zero_optimizer_progress_cannot_claim_a_learning_cycle(self):
        _, _, _, evidence, _ = fixture()
        self.assertIn("NO_DECLARED_TRAINING_PROGRESS", check(
            evidence=dataclasses.replace(evidence, active_seconds=0))["blockers"])

    def test_unauthorized_training_event_is_denied_even_with_claimed_consent(self):
        _, _, _, _, ev = fixture()
        self.assertIn("UNAUTHORIZED_TRAINING_EVENT", check(events=ev + (
            CycleEvent(seq=5, kind="TRAINING_STARTED", actor_id="vera-local", unix=121),
        ))["blockers"])

    def test_no_repeated_lifecycle_event_after_public_review(self):
        _, _, _, _, ev = fixture()
        actors = {
            "REQUEST": "vera-local",
            "FREEZE": "evidence-custodian",
            "CONSENT_CLAIM": "patrick-operator",
            "PUBLIC_REVIEW": "public-reviewer",
        }
        for kind, actor in actors.items():
            with self.subTest(kind=kind):
                result = check(events=ev + (
                    CycleEvent(seq=5, kind=kind, actor_id=actor, unix=125),
                ))
                self.assertIn("INVALID_EVENT_ORDER", result["blockers"])
                self.assertFalse(result["synthetic_source_conditions_satisfied"])

    def test_duplicate_reordered_or_changed_principals_fail_closed(self):
        _, _, _, _, ev = fixture()
        self.assertRaises(PolicyError, check, events=ev + (ev[-1],))
        self.assertIn("INVALID_EVENT_ORDER", check(events=ev[::-1])["blockers"])
        self.assertIn("EVENT_PRINCIPAL_MISMATCH", check(events=tuple(
            dataclasses.replace(e, actor_id="vera-local") if e.kind == "CONSENT_CLAIM" else e
            for e in ev))["blockers"])

    def test_malformed_types_and_extreme_inputs_rejected_without_exception_leak(self):
        b, r, c, e, ev = fixture()

        class AnyEqual:
            def __eq__(self, other):
                return True

        self.assertRaises(PolicyError, check, request=dataclasses.replace(r, objective_id=AnyEqual()))
        self.assertRaises(PolicyError, check, claim=dataclasses.replace(c, max_cost_cents=10**1000))
        self.assertRaises(PolicyError, check, evidence=dataclasses.replace(e, active_seconds=10**1000))
        self.assertRaises(PolicyError, check, events=ev + (CycleEvent(
            seq=5, kind=AnyEqual(), actor_id="vera-local", unix=121),))

    def test_original_evidence_hash_continuity_exposed_without_authorization(self):
        baseline, _, _, _, _ = fixture()
        result = check()
        self.assertEqual(result["original_baseline_sha256"], baseline_digest(baseline))
        self.assertTrue(result["previous_snapshot_preserved"])


if __name__ == "__main__":
    unittest.main()
