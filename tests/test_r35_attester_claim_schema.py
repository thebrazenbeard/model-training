"""Synthetic source-only R35 attester-claim contract checks.

No signing key, bank plaintext, GPU, host lease, or real attester exists here.
"""

import unittest
from unittest.mock import patch

import trainingday.r35_attester_claim_schema as schema_module
from trainingday.r35_attester_claim_schema import (
    ARMS, CLAIM_SCHEMA, assess_claim_shape,
)


H = {"plan_sha256": "a" * 64, "corpus_sha256": "b" * 64,
     "base_sha256": "c" * 64}


def sample_claim():
    arms = []
    for index, arm_id in enumerate(sorted(ARMS)):
        run_id = f"r35-synthetic-{index:02d}"
        digest = f"{index + 1:064x}"
        trainable_before = "e" * 64
        trainable_after = f"{index + 101:064x}"
        arms.append({
            "arm_id": arm_id,
            "declared": {"run_id": run_id, "adapter_after_sha256": digest,
                         "trainable_before_sha256": trainable_before,
                         "trainable_after_sha256": trainable_after},
            "purported_witness": {
                "arm_id": arm_id, "run_id": run_id,
                "corpus_sha256": H["corpus_sha256"],
                "base_sha256": H["base_sha256"],
                "adapter_after_sha256": digest,
                "trainable_before_sha256": trainable_before,
                "trainable_after_sha256": trainable_after,
                "host_id": "synthetic-host", "lease_id": f"synthetic-lease-{index}",
                "observed_start_ms": 100 + index * 10,
                "observed_end_ms": 105 + index * 10,
            },
        })
    return {"schema": CLAIM_SCHEMA, **H,
            "purported_attester_id": "attacker-self-signer", "arms": arms}


class AttesterClaimShapeTests(unittest.TestCase):
    def test_self_consistent_forged_witnesses_are_never_authority(self):
        result = assess_claim_shape(sample_claim(), expected=H)
        self.assertTrue(result.structurally_consistent)
        self.assertFalse(result.independent_attestation_verified)
        self.assertFalse(result.private_bank_unseal_authorized)
        self.assertIn("UNTRUSTED_PURPORTED_WITNESS", result.blockers)

    def test_missing_witness_fails_structure_closed(self):
        claim = sample_claim()
        del claim["arms"][0]["purported_witness"]
        result = assess_claim_shape(claim, expected=H)
        self.assertFalse(result.structurally_consistent)
        self.assertFalse(result.private_bank_unseal_authorized)
        self.assertIn("INVALID_ARM_RECORD", result.blockers)

    def test_witness_mismatched_arm_or_digest_fails_closed(self):
        for field, value in (("arm_id", "qlora:wrong"),
                             ("adapter_after_sha256", "f" * 64),
                             ("trainable_after_sha256", "f" * 64)):
            with self.subTest(field=field):
                claim = sample_claim()
                claim["arms"][0]["purported_witness"][field] = value
                result = assess_claim_shape(claim, expected=H)
                self.assertFalse(result.structurally_consistent)
                self.assertIn("DECLARATION_WITNESS_MISMATCH", result.blockers)

    def test_reused_run_id_and_missing_eighth_arm_fail_closed(self):
        claim = sample_claim()
        claim["arms"][1]["declared"]["run_id"] = claim["arms"][0]["declared"]["run_id"]
        claim["arms"][1]["purported_witness"]["run_id"] = claim["arms"][0]["declared"]["run_id"]
        self.assertIn("DUPLICATE_RUN_ID", assess_claim_shape(claim, expected=H).blockers)
        claim = sample_claim()
        claim["arms"].pop()
        self.assertIn("INVALID_ARM_SET", assess_claim_shape(claim, expected=H).blockers)

    def test_invalid_monotonic_times_and_corpus_binding_fail_closed(self):
        claim = sample_claim()
        claim["arms"][0]["purported_witness"]["observed_end_ms"] = 100
        self.assertIn("INVALID_OBSERVATION", assess_claim_shape(claim, expected=H).blockers)
        claim = sample_claim()
        claim["arms"][0]["purported_witness"]["corpus_sha256"] = "d" * 64
        self.assertIn("DECLARATION_WITNESS_MISMATCH", assess_claim_shape(claim, expected=H).blockers)

    def test_substituted_expected_plan_is_not_accepted(self):
        claim = sample_claim()
        result = assess_claim_shape(claim, expected={**H, "plan_sha256": "f" * 64})
        self.assertIn("EXPECTED_BINDING_MISMATCH", result.blockers)
        self.assertFalse(result.private_bank_unseal_authorized)

    def test_bool_timestamps_and_injected_authorization_are_rejected(self):
        claim = sample_claim()
        claim["arms"][0]["purported_witness"]["observed_end_ms"] = True
        self.assertIn("INVALID_OBSERVATION", assess_claim_shape(claim, expected=H).blockers)
        claim = sample_claim()
        claim["private_bank_unseal_authorized"] = True
        self.assertIn("INVALID_CLAIM_RECORD", assess_claim_shape(claim, expected=H).blockers)

    def test_custom_comparison_object_cannot_impersonate_witness(self):
        class Poison:
            def __eq__(self, other):
                raise AssertionError("untrusted comparison callback executed")

            def __ne__(self, other):
                raise AssertionError("untrusted comparison callback executed")

        claim = sample_claim()
        claim["arms"][0]["purported_witness"]["run_id"] = Poison()
        result = assess_claim_shape(claim, expected=H)
        self.assertFalse(result.structurally_consistent)
        self.assertIn("INVALID_ARM_RECORD", result.blockers)
        self.assertFalse(result.private_bank_unseal_authorized)

    def test_shared_exclusive_lease_claim_cannot_overlap_itself(self):
        claim = sample_claim()
        for arm in claim["arms"]:
            witness = arm["purported_witness"]
            witness["lease_id"] = "one-exclusive-gpu-lease"
            witness["observed_start_ms"] = 100
            witness["observed_end_ms"] = 200
        result = assess_claim_shape(claim, expected=H)
        self.assertFalse(result.structurally_consistent)
        self.assertIn("EXCLUSIVE_LEASE_OVERLAP", result.blockers)
        self.assertFalse(result.independent_attestation_verified)
        self.assertFalse(result.private_bank_unseal_authorized)

    def test_same_declared_initial_adapter_cannot_have_distinct_trainable_start(self):
        claim = sample_claim()
        arm = claim["arms"][0]
        arm["declared"]["trainable_before_sha256"] = "d" * 64
        arm["purported_witness"]["trainable_before_sha256"] = "d" * 64
        result = assess_claim_shape(claim, expected=H)
        self.assertFalse(result.structurally_consistent)
        self.assertIn("INITIAL_TRAINABLE_DIGEST_MISMATCH", result.blockers)
        self.assertFalse(result.private_bank_unseal_authorized)

    def test_reused_lease_with_sequential_nonoverlapping_observations_is_shape_valid(self):
        claim = sample_claim()
        for index, arm in enumerate(claim["arms"]):
            witness = arm["purported_witness"]
            witness["lease_id"] = "one-exclusive-gpu-lease"
            witness["observed_start_ms"] = 100 + index * 10
            witness["observed_end_ms"] = 110 + index * 10
        result = assess_claim_shape(claim, expected=H)
        self.assertTrue(result.structurally_consistent)
        self.assertFalse(result.independent_attestation_verified)

    def test_one_lease_id_cannot_switch_host_clock_domains(self):
        claim = sample_claim()
        claim["arms"][1]["purported_witness"]["lease_id"] = (
            claim["arms"][0]["purported_witness"]["lease_id"])
        claim["arms"][1]["purported_witness"]["host_id"] = "different-host"
        result = assess_claim_shape(claim, expected=H)
        self.assertIn("LEASE_HOST_MISMATCH", result.blockers)
        self.assertFalse(result.structurally_consistent)

    def test_malformed_top_binding_cannot_invoke_attacker_comparison_hook(self):
        class Poison:
            def __eq__(self, other):
                raise RuntimeError("malformed binding comparison executed")

            def __ne__(self, other):
                raise RuntimeError("malformed binding comparison executed")

        claim = sample_claim()
        claim["corpus_sha256"] = Poison()
        result = assess_claim_shape(claim, expected=H)
        self.assertFalse(result.structurally_consistent)
        self.assertIn("INVALID_CLAIM_BINDING", result.blockers)
        self.assertFalse(result.private_bank_unseal_authorized)

    def test_invalid_arm_count_fails_without_traversing_all_arms(self):
        for size in (1024, 2048):
            with self.subTest(size=size):
                claim = sample_claim()
                claim["arms"] = [claim["arms"][0]] * size
                with patch.object(schema_module, "_object", wraps=schema_module._object) as shape:
                    result = assess_claim_shape(claim, expected=H)
                self.assertIn("INVALID_ARM_SET", result.blockers)
                self.assertFalse(result.private_bank_unseal_authorized)
                # Only expected+claim shapes may be inspected; invalid array
                # cardinality should short-circuit before per-arm validation.
                self.assertEqual(shape.call_count, 2)


if __name__ == "__main__":
    unittest.main()
