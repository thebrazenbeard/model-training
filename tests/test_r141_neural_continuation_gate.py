"""Adversarial source-only tests; synthetic receipts grant no authority."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from trainingday.r141_neural_continuation_gate import (
    SCHEMA, EvidenceError, assess_source_claims, read_claims,
)


def h(char: str) -> str:
    return char * 64


def fixture() -> dict:
    shared = {
        "base_revision": "a" * 40,
        "model_config_sha256": h("1"),
        "base_frozen_weights_sha256": h("0"),
        "train_manifest_sha256": h("2"),
        "selected_ids_sha256": h("3"),
        "seed": 42,
        "hyperparameters_sha256": h("4"),
        "initial_optimizer_sha256": h("1"),
        "initial_rng_sha256": h("3"),
    }
    arms = {
        name: {
            **shared,
            "mode": name,
            "optimizer_steps": 0 if name == "zero" else 8,
            "initial_fp32_digest": h("5"),
            "final_fp32_digest": {"zero": h("5"), "correct": h("8"),
                                   "corrupt": h("9")}[name],
            "adapter_artifact_sha256": {"zero": h("a"),
                                        "correct": h("b"), "corrupt": h("c")}[name],
            "training_input_sha256": h("2") if name != "corrupt" else h("f"),
        }
        for name in ("zero", "correct", "corrupt")
    }
    arms["correct"]["claimed_fp32_l2_delta"] = 0.125
    arms["corrupt"]["corrupt_label_audit_sha256"] = h("e")
    return {
        "schema": SCHEMA,
        "source_git_head": "d" * 40,
        "run_contract_sha256": h("d"),
        "final_checkpoint_sha256": h("e"),
        "arms": arms,
        "continuation_segments": [
            {
                "start_step": 0, "end_step": 4,
                "model_in_fp32_digest": h("5"), "model_out_fp32_digest": h("6"),
                "optimizer_in_sha256": h("1"), "optimizer_out_sha256": h("2"),
                "rng_in_sha256": h("3"), "rng_out_sha256": h("4"),
                "checkpoint_in_sha256": None, "checkpoint_out_sha256": h("7"),
                "run_contract_sha256": h("d"),
            },
            {
                "start_step": 4, "end_step": 8,
                "model_in_fp32_digest": h("6"), "model_out_fp32_digest": h("8"),
                "optimizer_in_sha256": h("2"), "optimizer_out_sha256": h("0"),
                "rng_in_sha256": h("4"), "rng_out_sha256": h("f"),
                "checkpoint_in_sha256": h("7"), "checkpoint_out_sha256": h("e"),
                "run_contract_sha256": h("d"),
            },
        ],
        "evaluation": {
            "frozen_reference_sha256": h("0"),
            "sealed_bank_sha256": h("c"),
            "sealed_commitment_sha256": h("b"),
            "examiner_result_sha256": h("a"),
            "examiner_identity": "separate-examiner-runtime",
            "independent_examiner_claimed": True,
            "heldout_case_count": 100,
            "frozen_case_count": 100,
            "zero_accuracy": 0.50,
            "correct_accuracy": 0.75,
            "corrupt_accuracy": 0.51,
            "frozen_baseline_accuracy": 0.88,
            "frozen_after_accuracy": 0.86,
            "zero_correct_count": 50,
            "correct_correct_count": 75,
            "corrupt_correct_count": 51,
            "frozen_baseline_correct_count": 88,
            "frozen_after_correct_count": 86,
            "correct_frozen_kl": 0.08,
            "max_frozen_kl": 0.10,
            "max_frozen_accuracy_drop": 0.03,
        },
    }


class ContinuationEvidenceTests(unittest.TestCase):
    def reject(self, mutate, reason: str):
        envelope = fixture()
        mutate(envelope)
        result = assess_source_claims(envelope)
        self.assertFalse(result["source_claims_consistent"])
        self.assertEqual(result["reason_codes"], [reason])
        self.assertFalse(result["proves_neural_learning"])
        self.assertFalse(result["authorizes_training"])
        self.assertFalse(result["authorizes_resume"])

    def test_consistent_claim_is_still_not_qualifying_physical_evidence(self):
        outcome = assess_source_claims(fixture())
        self.assertTrue(outcome["source_claims_consistent"])
        self.assertEqual(outcome["status"],
                         "SOURCE_CLAIMS_CONSISTENT_PHYSICAL_VERIFICATION_HOLD")
        for flag in ("authorizes_training", "authorizes_resume",
                     "authorizes_graduation", "proves_neural_learning",
                     "proves_physical_reload"):
            self.assertIs(outcome[flag], False)

    def test_zero_update_arm_cannot_change(self):
        self.reject(lambda e: e["arms"]["zero"].update(final_fp32_digest=h("f")),
                    "ZERO_ARM_UPDATED")

    def test_different_initial_digest_fails(self):
        self.reject(lambda e: e["arms"]["correct"].update(initial_fp32_digest=h("f")),
                    "TRAINED_ARM_DIFFERENT_INITIAL_TENSORS")

    def test_corrupt_label_control_must_be_matched(self):
        self.reject(lambda e: e["arms"]["corrupt"].update(seed=87),
                    "CONTROL_SOURCE_OR_SEED_MISMATCH")

    def test_corrupt_training_input_differentiates_label_control(self):
        self.reject(lambda e: e["arms"]["corrupt"].update(training_input_sha256=h("2")),
                    "CORRUPTED_LABEL_INPUT_NOT_DISTINCTLY_ATTESTED")

    def test_corrupt_label_derivation_requires_audit_binding(self):
        self.reject(lambda e: e["arms"]["corrupt"].pop("corrupt_label_audit_sha256"),
                    "CORRUPTED_LABEL_INPUT_NOT_DISTINCTLY_ATTESTED")

    def test_corrupt_control_must_update_separately(self):
        self.reject(lambda e: e["arms"]["corrupt"].update(final_fp32_digest=h("5")),
                    "CORRUPT_CONTROL_NOT_UPDATED")

    def test_correct_digest_cannot_match_corrupt(self):
        self.reject(lambda e: e["arms"]["correct"].update(final_fp32_digest=h("9")),
                    "NO_DISTINCT_CORRECT_WEIGHT_CHANGE_CLAIM")

    def test_nontrivial_tensor_delta_claim_required(self):
        self.reject(lambda e: e["arms"]["correct"].update(claimed_fp32_l2_delta=0),
                    "NONTRIVIAL_TENSOR_DELTA_UNESTABLISHED")

    def test_numerically_trivial_tensor_delta_claim_rejected(self):
        self.reject(lambda e: e["arms"]["correct"].update(claimed_fp32_l2_delta=1e-12),
                    "NONTRIVIAL_TENSOR_DELTA_UNESTABLISHED")

    def test_no_optimizer_step_gap(self):
        self.reject(lambda e: e["continuation_segments"][1].update(start_step=5),
                    "CONTINUATION_STEP_GAP")

    def test_model_state_must_chain(self):
        self.reject(lambda e: e["continuation_segments"][1].update(
            model_in_fp32_digest=h("f")), "CONTINUATION_MODEL_STATE_BROKEN")

    def test_optimizer_must_chain(self):
        self.reject(lambda e: e["continuation_segments"][1].update(
            optimizer_in_sha256=h("f")),
            "CONTINUATION_OPTIMIZER_RNG_OR_CHECKPOINT_BROKEN")

    def test_initial_optimizer_state_requires_same_zero_control(self):
        self.reject(lambda e: e["continuation_segments"][0].update(
            optimizer_in_sha256=h("f")), "INITIAL_OPTIMIZER_OR_RNG_NOT_BOUND")

    def test_initial_rng_state_requires_same_zero_control(self):
        self.reject(lambda e: e["continuation_segments"][0].update(
            rng_in_sha256=h("f")), "INITIAL_OPTIMIZER_OR_RNG_NOT_BOUND")

    def test_rng_must_chain(self):
        self.reject(lambda e: e["continuation_segments"][1].update(
            rng_in_sha256=h("e")),
            "CONTINUATION_OPTIMIZER_RNG_OR_CHECKPOINT_BROKEN")

    def test_resume_checkpoint_must_chain(self):
        self.reject(lambda e: e["continuation_segments"][1].update(
            checkpoint_in_sha256=h("f")),
            "CONTINUATION_OPTIMIZER_RNG_OR_CHECKPOINT_BROKEN")

    def test_run_contract_cannot_mutate(self):
        self.reject(lambda e: e["continuation_segments"][1].update(
            run_contract_sha256=h("f")), "CONTINUATION_CONTRACT_CHANGED")

    def test_new_final_checkpoint_is_exactly_bound(self):
        self.reject(lambda e: e.update(final_checkpoint_sha256=h("f")),
                    "CONTINUATION_FINAL_STATE_MISMATCH")

    def test_sealed_bank_must_be_distinct(self):
        self.reject(lambda e: e["evaluation"].update(sealed_bank_sha256=h("2")),
                    "EVALUATION_NOT_SEPARATE_FROM_TRAIN")

    def test_frozen_reference_requires_exact_base(self):
        self.reject(lambda e: e["evaluation"].update(frozen_reference_sha256=h("e")),
                    "FROZEN_REFERENCE_NOT_PINNED_TO_BASE")

    def test_examiner_claim_missing(self):
        self.reject(lambda e: e["evaluation"].update(independent_examiner_claimed=False),
                    "INDEPENDENT_EXAMINER_NOT_IDENTIFIED")

    def test_corrupt_control_improves_equally(self):
        self.reject(lambda e: e["evaluation"].update(corrupt_accuracy=0.76,
                                                   corrupt_correct_count=76),
                    "LEARNING_NOT_DISTINCT_FROM_CONTROLS")

    def test_frozen_kl_violation(self):
        self.reject(lambda e: e["evaluation"].update(correct_frozen_kl=0.11),
                    "FROZEN_COMPETENCE_RETENTION_FAILED")

    def test_frozen_competence_drop(self):
        self.reject(lambda e: e["evaluation"].update(frozen_after_accuracy=0.80,
                                                    frozen_after_correct_count=80),
                    "FROZEN_COMPETENCE_RETENTION_FAILED")

    def test_one_case_cannot_claim_fractional_accuracy(self):
        self.reject(lambda e: e["evaluation"].update(heldout_case_count=1),
                    "ACCURACY_DENOMINATOR_OR_COUNT_MISMATCH")

    def test_accuracy_cannot_disagree_with_correct_count(self):
        self.reject(lambda e: e["evaluation"].update(correct_correct_count=74),
                    "ACCURACY_DENOMINATOR_OR_COUNT_MISMATCH")

    def test_all_correct_counts_required(self):
        self.reject(lambda e: e["evaluation"].pop("frozen_baseline_correct_count"),
                    "ACCURACY_DENOMINATOR_OR_COUNT_MISMATCH")

    def test_boolean_numerator_not_integer(self):
        self.reject(lambda e: e["evaluation"].update(zero_correct_count=True),
                    "ACCURACY_DENOMINATOR_OR_COUNT_MISMATCH")

    def test_frozen_evaluation_uses_its_own_denominator(self):
        claim = fixture()
        claim["evaluation"].update(frozen_case_count=50,
                                   frozen_baseline_correct_count=44,
                                   frozen_after_correct_count=43)
        self.assertTrue(assess_source_claims(claim)["source_claims_consistent"])

    def test_nan_in_evaluation_is_rejected(self):
        self.reject(lambda e: e["evaluation"].update(correct_frozen_kl=float("nan")),
                    "CLAIMS_INVALID")

    def test_enormous_json_integer_cannot_crash_numeric_gate(self):
        for field in ("zero_accuracy", "correct_accuracy", "corrupt_accuracy",
                      "frozen_baseline_accuracy", "frozen_after_accuracy",
                      "correct_frozen_kl", "max_frozen_kl", "max_frozen_accuracy_drop"):
            with self.subTest(field=field):
                claim = fixture()
                claim["evaluation"][field] = 10**1000
                result = assess_source_claims(claim)
                self.assertFalse(result["source_claims_consistent"])
                self.assertFalse(result["authorizes_resume"])
                self.assertFalse(result["proves_neural_learning"])

    def test_nested_infinite_float_rejected(self):
        claim = fixture()
        claim["evaluation"]["forged_nested"] = {"k": [float("inf")]}
        outcome = assess_source_claims(claim)
        self.assertEqual(outcome["reason_codes"], ["CLAIMS_INVALID"])
        self.assertFalse(outcome["authorizes_training"])

    def test_bool_optimizer_steps_is_not_integer(self):
        self.reject(lambda e: e["arms"]["correct"].update(optimizer_steps=True),
                    "CONTROL_STEPS_INVALID")

    def test_read_claims_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "claims.json"
            file.write_text('{"schema":"A","schema":"B"}', encoding="utf-8")
            with self.assertRaises(EvidenceError):
                read_claims(file)

    def test_read_claims_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "claims.json"
            file.write_text('{"actual":NaN}', encoding="utf-8")
            with self.assertRaises(EvidenceError):
                read_claims(file)

    def test_read_claims_valid_json_does_not_grant_authority(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "claims.json"
            file.write_text(json.dumps(fixture()), encoding="utf-8")
            self.assertTrue(assess_source_claims(read_claims(file))[
                "source_claims_consistent"])
            self.assertFalse(assess_source_claims(read_claims(file))[
                "authorizes_resume"])


if __name__ == "__main__":
    unittest.main()
