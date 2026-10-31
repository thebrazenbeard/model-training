"""Adversarial source-only evidence provenance checks for TrainingDay R141."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from trainingday.r141_evidence_gate import (
    EvidenceClaim, QualificationPacket, REQUIRED_PHYSICAL_PROBES,
    assess_qualification_packet,
)


HEAD = "a" * 40
CHECKPOINT = "b" * 64
TRAINER = "trainer-principal-01"
RUN_ID = "run-training-001"


def a_claim(name: str, *, evidence_class: str = "REAL_SYSTEM_PROBE",
            passed: bool = True, issuer: str | None = None) -> EvidenceClaim:
    return EvidenceClaim(
        requirement=name, evidence_class=evidence_class, git_head=HEAD,
        run_id=RUN_ID, checkpoint_sha256=CHECKPOINT,
        issuer=issuer or "host-owner-001",
        receipt_sha256=hashlib.sha256(name.encode("ascii")).hexdigest(),
        passed=passed,
    )


def packet(*claims: EvidenceClaim) -> QualificationPacket:
    return QualificationPacket(
        git_head=HEAD, run_id=RUN_ID, checkpoint_sha256=CHECKPOINT,
        trainer_principal=TRAINER, claims=tuple(claims),
    )


def all_claims() -> tuple[EvidenceClaim, ...]:
    return tuple(a_claim(name) for name in sorted(REQUIRED_PHYSICAL_PROBES))


class R141EvidenceGateTests(unittest.TestCase):
    def assert_permanent_hold(self, decision):
        self.assertFalse(decision.training_authorized)
        self.assertFalse(decision.checkpoint_resume_authorized)
        self.assertFalse(decision.sealed_eval_unseal_authorized)
        self.assertFalse(decision.model_learning_qualified)

    def test_empty_packet_missing_all_physical_evidence(self):
        d = assess_qualification_packet(packet())
        self.assertEqual(d.reason, "PHYSICAL_PROBE_REPORTS_MISSING_OR_WRONG_CLASS")
        self.assertEqual(set(d.missing_physical_claims), REQUIRED_PHYSICAL_PROBES)
        self.assertFalse(d.documentary_packet_claimed_complete)
        self.assert_permanent_hold(d)

    def test_source_tests_cannot_masquerade_as_physical_probes(self):
        d = assess_qualification_packet(packet(*(replace(c, evidence_class="TEST") for c in all_claims())))
        self.assertEqual(set(d.missing_physical_claims), REQUIRED_PHYSICAL_PROBES)
        self.assert_permanent_hold(d)

    def test_model_reviews_cannot_masquerade_as_physical_probes(self):
        d = assess_qualification_packet(packet(*(replace(c, evidence_class="MODEL_REVIEW") for c in all_claims())))
        self.assertEqual(set(d.missing_physical_claims), REQUIRED_PHYSICAL_PROBES)
        self.assert_permanent_hold(d)

    def test_all_purported_physical_probes_still_cannot_authorize(self):
        d = assess_qualification_packet(packet(*all_claims()))
        self.assertTrue(d.documentary_packet_claimed_complete)
        self.assertEqual(d.missing_physical_claims, ())
        self.assertEqual(d.reason, "ALL_PHYSICAL_REPORTS_ASSERTED_BUT_UNVERIFIED_HOST_AUTHORITY_HOLD")
        self.assert_permanent_hold(d)

    def test_host_lease_absent_fails_closed(self):
        d = assess_qualification_packet(packet(*(c for c in all_claims() if c.requirement != "EXCLUSIVE_HOST_LEASE")))
        self.assertEqual(d.missing_physical_claims, ("EXCLUSIVE_HOST_LEASE",))
        self.assert_permanent_hold(d)

    def test_immutable_witness_absent_fails_closed(self):
        d = assess_qualification_packet(packet(*(c for c in all_claims() if c.requirement != "INDEPENDENT_MONOTONIC_WITNESS")))
        self.assertEqual(d.missing_physical_claims, ("INDEPENDENT_MONOTONIC_WITNESS",))
        self.assert_permanent_hold(d)

    def test_qwen_loader_absent_fails_closed(self):
        d = assess_qualification_packet(packet(*(c for c in all_claims() if c.requirement != "PINNED_QWEN_WEIGHT_LOAD")))
        self.assertEqual(d.missing_physical_claims, ("PINNED_QWEN_WEIGHT_LOAD",))
        self.assert_permanent_hold(d)

    def test_checkpoint_reload_absent_fails_closed(self):
        d = assess_qualification_packet(packet(*(c for c in all_claims() if c.requirement != "CHECKPOINT_RELOAD")))
        self.assertEqual(d.missing_physical_claims, ("CHECKPOINT_RELOAD",))
        self.assert_permanent_hold(d)

    def test_optimizer_rng_parity_absent_fails_closed(self):
        d = assess_qualification_packet(packet(*(c for c in all_claims() if c.requirement != "OPTIMIZER_RNG_CONTINUITY")))
        self.assertEqual(d.missing_physical_claims, ("OPTIMIZER_RNG_CONTINUITY",))
        self.assert_permanent_hold(d)

    def test_sealed_eval_absent_fails_closed(self):
        d = assess_qualification_packet(packet(*(c for c in all_claims() if c.requirement != "SEALED_HELDOUT_EVALUATION")))
        self.assertEqual(d.missing_physical_claims, ("SEALED_HELDOUT_EVALUATION",))
        self.assert_permanent_hold(d)

    def test_malformed_git_head_rejected(self):
        d = assess_qualification_packet(replace(packet(*all_claims()), git_head="a" * 39))
        self.assertEqual(d.reason, "EXACT_GIT_HEAD_REQUIRED")
        self.assert_permanent_hold(d)

    def test_malformed_checkpoint_sha_rejected(self):
        d = assess_qualification_packet(replace(packet(*all_claims()), checkpoint_sha256="b" * 63))
        self.assertEqual(d.reason, "CHECKPOINT_DIGEST_REQUIRED")
        self.assert_permanent_hold(d)

    def test_wrong_head_evidence_rejected(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(replace(reports[0], git_head="d" * 40), *reports[1:]))
        self.assertEqual(d.reason, "EXACT_ARTIFACT_BINDING_MISMATCH")
        self.assert_permanent_hold(d)

    def test_wrong_run_binding_rejected(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(replace(reports[0], run_id="other-run-001"), *reports[1:]))
        self.assertEqual(d.reason, "EXACT_ARTIFACT_BINDING_MISMATCH")

    def test_wrong_checkpoint_digest_binding_rejected(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(replace(reports[0], checkpoint_sha256="d" * 64), *reports[1:]))
        self.assertEqual(d.reason, "EXACT_ARTIFACT_BINDING_MISMATCH")

    def test_duplicate_requirement_rejected(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(*reports, reports[0]))
        self.assertEqual(d.reason, "DUPLICATE_OR_CONFLICTING_REQUIREMENT")

    def test_false_result_does_not_satisfy_requirement(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(replace(reports[0], passed=False), *reports[1:]))
        self.assertEqual(d.missing_physical_claims, (reports[0].requirement,))
        self.assert_permanent_hold(d)

    def test_a_numeric_true_is_not_boolean_success(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(replace(reports[0], passed=1), *reports[1:]))
        self.assertEqual(d.reason, "MALFORMED_PROBE_RESULT")
        self.assert_permanent_hold(d)

    def test_unknown_requirement_rejected(self):
        d = assess_qualification_packet(packet(a_claim("NEW_UNAPPROVED_GAP")))
        self.assertEqual(d.reason, "UNKNOWN_REQUIREMENT")

    def test_unhashable_requirement_rejected_without_crashing(self):
        d = assess_qualification_packet(packet(replace(all_claims()[0], requirement=[])))
        self.assertEqual(d.reason, "MALFORMED_EVIDENCE_CLAIM")

    def test_unhashable_evidence_class_rejected_without_crashing(self):
        d = assess_qualification_packet(packet(replace(all_claims()[0], evidence_class=[])))
        self.assertEqual(d.reason, "UNKNOWN_EVIDENCE_CLASS")

    def test_bad_receipt_hash_rejected(self):
        d = assess_qualification_packet(packet(replace(all_claims()[0], receipt_sha256="not-a-hash")))
        self.assertEqual(d.reason, "MALFORMED_RECEIPT_IDENTITY")

    def test_one_receipt_cannot_be_reused_to_assert_two_distinct_probes(self):
        reports = all_claims()
        d = assess_qualification_packet(packet(
            reports[0], replace(reports[1], receipt_sha256=reports[0].receipt_sha256),
            *reports[2:],
        ))
        self.assertEqual(d.reason, "REPLAYED_RECEIPT_ACROSS_REQUIREMENTS")
        self.assert_permanent_hold(d)

    def test_host_lease_claim_must_not_be_issued_by_trainer(self):
        reports = all_claims()
        modified = [replace(c, issuer=TRAINER) if c.requirement == "EXCLUSIVE_HOST_LEASE" else c for c in reports]
        d = assess_qualification_packet(packet(*modified))
        self.assertEqual(d.reason, "INDEPENDENT_ISSUER_REQUIRED")

    def test_trainer_cannot_self_attest_independent_witness(self):
        reports = all_claims()
        modified = [replace(c, issuer=TRAINER) if c.requirement == "INDEPENDENT_MONOTONIC_WITNESS" else c for c in reports]
        d = assess_qualification_packet(packet(*modified))
        self.assertEqual(d.reason, "INDEPENDENT_ISSUER_REQUIRED")

    def test_trainer_cannot_self_attest_sealed_evaluation(self):
        reports = all_claims()
        modified = [replace(c, issuer=TRAINER) if c.requirement == "SEALED_HELDOUT_EVALUATION" else c for c in reports]
        d = assess_qualification_packet(packet(*modified))
        self.assertEqual(d.reason, "INDEPENDENT_ISSUER_REQUIRED")

    def test_list_instead_of_immutable_tuple_rejected(self):
        d = assess_qualification_packet(replace(packet(*all_claims()), claims=list(all_claims())))
        self.assertEqual(d.reason, "MALFORMED_CLAIM_SET")

    def test_too_many_claims_rejected_before_loop(self):
        d = assess_qualification_packet(replace(packet(*all_claims()), claims=all_claims() * 22))
        self.assertEqual(d.reason, "MALFORMED_CLAIM_SET")

    def test_non_packet_input_rejected(self):
        d = assess_qualification_packet({"git_head": HEAD})
        self.assertEqual(d.reason, "MALFORMED_PACKET")
        self.assert_permanent_hold(d)


if __name__ == "__main__":
    unittest.main()
