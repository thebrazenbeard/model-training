"""Source-only host launch declarations: never a physical launch capability."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest

from trainingday.host_launch_request import (
    HostLeaseDeclaration,
    LaunchRequest,
    RightsDeclaration,
    RunPlan,
    SourceBinding,
    assess_launch_request,
)
from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable,
    require_physical_launch_authority,
)


def valid_request():
    return LaunchRequest(
        schema="TRAININGDAY_HOST_LAUNCH_DECLARATION_V1",
        source=SourceBinding(
            source_commit_sha1="a" * 40,
            source_tree_sha1="b" * 40,
            corpus_manifest_sha256="c" * 64,
            corpus_producer_commit_sha1="d" * 40,
            training_input_sha256="e" * 64,
            base_binding_sha256="f" * 64,
            base_model_id="Qwen/Qwen3.5-4B",
            base_revision_sha1="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ),
        rights=RightsDeclaration(
            decision_id="rights-20261101-a",
            decision_sha256="1" * 64,
            covered_manifest_sha256="c" * 64,
            declared_disposition="TRAINING_PERMITTED_CLAIM",
            purported_owner="owner-principal-a",
        ),
        plan=RunPlan(
            run_id="r16-correct-a",
            protocol="R16_CHECKPOINTABLE",
            mode="correct",
            optimizer_steps=16,
            selected_cases=128,
            seed=41,
            learning_rate_ppb=20_000,
            expected_zero_digest_sha256="2" * 64,
        ),
        lease=HostLeaseDeclaration(
            host_instance="host-instance-20261101",
            caller_principal="caller-principal-a",
            gpu_lease_id="gpu-lease-20261101-a",
            gpu_lease_epoch=4,
            request_nonce="3" * 64,
            issued_at_unix=1_793_497_000,
            expires_at_unix=1_793_497_600,
            expected_witness_epoch=3,
            expected_witness_head_sha256="4" * 64,
        ),
    )


class HostLaunchRequestTests(unittest.TestCase):
    def test_valid_declaration_digest_is_deterministic_and_never_authorizes(self):
        request = valid_request()
        first = assess_launch_request(request)
        second = assess_launch_request(valid_request())
        self.assertEqual(first.status, "SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED")
        self.assertEqual(first.reasons, ())
        self.assertEqual(first.request_sha256, second.request_sha256)
        self.assertEqual(len(first.request_sha256), 64)
        self.assertFalse(first.authorizes_training)
        self.assertFalse(first.authorizes_resume)
        self.assertFalse(first.host_identity_verified)
        self.assertFalse(first.dataset_rights_verified)
        self.assertFalse(first.gpu_lease_verified)
        self.assertFalse(first.replay_prevention_verified)
        self.assertFalse(first.physical_training_observed)
        # A separately computed canonical digest is the oracle.
        self.assertEqual(first.request_sha256, hashlib.sha256(json.dumps({
            "schema": request.schema,
            "source": vars_for(request.source),
            "rights": vars_for(request.rights),
            "plan": vars_for(request.plan),
            "lease": vars_for(request.lease),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")).hexdigest())
        with self.assertRaises(FrozenInstanceError):
            request.plan = request.plan

    def test_source_or_rights_mutation_changes_digest_without_grant(self):
        original = valid_request()
        changed_source = replace(
            original, source=replace(original.source, source_tree_sha1="9" * 40))
        changed_rights = replace(
            original, rights=replace(original.rights, declared_disposition="HOLD_CLAIM"))
        for changed in (changed_source, changed_rights):
            with self.subTest(changed=changed):
                outcome = assess_launch_request(changed)
                self.assertEqual(outcome.status, "SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED")
                self.assertNotEqual(outcome.request_sha256,
                                    assess_launch_request(original).request_sha256)
                self.assertFalse(outcome.authorizes_training)

    def test_wrong_types_hashes_or_model_binding_fail_closed(self):
        original = valid_request()
        bad_sources = [
            replace(original.source, source_commit_sha1="A" * 40),
            replace(original.source, source_tree_sha1="g" * 40),
            replace(original.source, corpus_manifest_sha256="0" * 63),
            replace(original.source, corpus_producer_commit_sha1="a" * 39),
            replace(original.source, training_input_sha256="N" * 64),
            replace(original.source, base_binding_sha256="f" * 63),
            replace(original.source, base_model_id="wrong/donor"),
            replace(original.source, base_revision_sha1="0" * 40),
        ]
        for source in bad_sources:
            with self.subTest(source=source):
                outcome = assess_launch_request(replace(original, source=source))
                self.assertEqual(outcome.status, "SOURCE_DECLARATION_REJECTED")
                self.assertIsNone(outcome.request_sha256)
        for rights in (
            replace(original.rights, decision_id="../traversal"),
            replace(original.rights, decision_sha256="N" * 64),
            replace(original.rights, covered_manifest_sha256="a" * 64),
            replace(original.rights, declared_disposition="REAL_AUTHORIZATION"),
            replace(original.rights, purported_owner=""),
        ):
            with self.subTest(rights=rights):
                self.assertEqual(assess_launch_request(
                    replace(original, rights=rights)).status,
                    "SOURCE_DECLARATION_REJECTED")

    def test_run_plan_has_bounded_zero_r15_r16_modes_and_exact_integer_types(self):
        original = valid_request()
        bad_plans = [
            replace(original.plan, optimizer_steps=True),
            replace(original.plan, optimizer_steps=15),
            replace(original.plan, selected_cases=120),
            replace(original.plan, learning_rate_ppb=0),
            replace(original.plan, learning_rate_ppb=200_001),
            replace(original.plan, protocol="R17_PRETEND"),
            replace(original.plan, mode="deploy"),
            replace(original.plan, seed=-1),
            replace(original.plan, expected_zero_digest_sha256=None),
            replace(original.plan, run_id="../escape"),
            replace(original.plan, protocol="R15_FRESH", optimizer_steps=16),
            replace(original.plan, mode="zero", optimizer_steps=0),
        ]
        for plan in bad_plans:
            with self.subTest(plan=plan):
                self.assertEqual(assess_launch_request(
                    replace(original, plan=plan)).status,
                    "SOURCE_DECLARATION_REJECTED")
        valid_zero = replace(original.plan, mode="zero", optimizer_steps=0,
                             selected_cases=8, expected_zero_digest_sha256=None)
        self.assertEqual(assess_launch_request(
            replace(original, plan=valid_zero)).status,
            "SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED")
        valid_r15 = replace(original.plan, protocol="R15_FRESH", optimizer_steps=8)
        self.assertEqual(assess_launch_request(
            replace(original, plan=valid_r15)).status,
            "SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED")

    def test_lease_expiry_nonce_epoch_and_replay_preconditions_are_declarations(self):
        original = valid_request()
        bad_leases = [
            replace(original.lease, host_instance="../host"),
            replace(original.lease, caller_principal=""),
            replace(original.lease, gpu_lease_id="a"),
            replace(original.lease, gpu_lease_epoch=True),
            replace(original.lease, gpu_lease_epoch=0),
            replace(original.lease, request_nonce="0" * 63),
            replace(original.lease, issued_at_unix=0),
            replace(original.lease, expires_at_unix=original.lease.issued_at_unix),
            replace(original.lease, expires_at_unix=original.lease.issued_at_unix + 3601),
            replace(original.lease, expected_witness_epoch=-1),
            replace(original.lease, expected_witness_head_sha256="g" * 64),
            replace(original.lease, expected_witness_epoch=0),
        ]
        for lease in bad_leases:
            with self.subTest(lease=lease):
                self.assertEqual(assess_launch_request(
                    replace(original, lease=lease)).status,
                    "SOURCE_DECLARATION_REJECTED")
        genesis = replace(original.lease, expected_witness_epoch=0,
                          expected_witness_head_sha256="0" * 64)
        assessment = assess_launch_request(replace(original, lease=genesis))
        self.assertEqual(assessment.status, "SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED")
        self.assertFalse(assessment.replay_prevention_verified)

    def test_spoofed_containers_and_rights_claim_cannot_unlock_physical_guard(self):
        original = valid_request()
        class HostileString(str):
            def __eq__(self, other):
                raise AssertionError("validator evaluated hostile equality")

        for altered in (
            replace(original, schema="TRAININGDAY_HOST_LAUNCH_DECLARATION_V0"),
            replace(original, schema=HostileString(original.schema)),
            replace(original, source=replace(
                original.source,
                base_revision_sha1=HostileString(original.source.base_revision_sha1))),
            replace(original, source=vars_for(original.source)),
            replace(original, rights=vars_for(original.rights)),
            replace(original, plan=vars_for(original.plan)),
            replace(original, lease=vars_for(original.lease)),
        ):
            with self.subTest(altered=altered):
                self.assertEqual(
                    assess_launch_request(altered).status,
                    "SOURCE_DECLARATION_REJECTED")
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            require_physical_launch_authority(operation="R16 neural training")


def vars_for(value):
    from dataclasses import fields
    return {field.name: getattr(value, field.name) for field in fields(value)}


if __name__ == "__main__":
    unittest.main()
