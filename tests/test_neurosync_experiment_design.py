"""Offline source-contract tests: no model calls or filesystem effects."""

from copy import deepcopy
import json
import unittest
from pathlib import Path

if Path(__file__).resolve().parent.name == "tests":
    # Published model-training/tests: never fall back to a similarly named
    # unrelated module when the intended production sibling is missing.
    from trainingday.neurosync_experiment_design import (
        ARMS, IndependentResultAuthorityUnavailable, PlanHold,
        example_source_plan, require_independent_result_qualification,
        validate_study_plan,
    )
else:
    # Isolated scratch layout: module and tests next to each other.
    from neurosync_experiment_design import (
        ARMS, IndependentResultAuthorityUnavailable, PlanHold,
        example_source_plan, require_independent_result_qualification,
        validate_study_plan,
    )


class SevenArmPreregistrationTests(unittest.TestCase):
    def setUp(self):
        self.plan = example_source_plan()

    def assert_held(self, modified):
        with self.assertRaises(PlanHold):
            validate_study_plan(modified)

    def test_fixture_is_source_only_with_seven_arms(self):
        validated = validate_study_plan(self.plan)
        self.assertEqual(tuple(ARMS), validated.arm_ids)
        self.assertEqual(50, validated.heldout_item_count)
        self.assertEqual("SOURCE_PLAN_VALID__EXECUTION_UNVERIFIED", validated.disposition)
        self.assertEqual(64, len(validated.canonical_sha256))
        self.assertEqual(validated.canonical_sha256, validate_study_plan(json.dumps(self.plan)).canonical_sha256)

    def test_missing_arm(self):
        self.plan["arms"].pop()
        self.assert_held(self.plan)

    def test_duplicate_arm(self):
        self.plan["arms"][-1]["arm_id"] = "untouched"
        self.assert_held(self.plan)

    def test_unrecognized_arm(self):
        self.plan["arms"][-1]["arm_id"] = "uncontrolled"
        self.assert_held(self.plan)

    def test_arm_cannot_substitute_wrong_mechanism(self):
        self.plan["arms"][3]["intervention"] = "APPROVED_OUTPUTS_RECIPIENT_NATIVE_TRAIN"
        self.assert_held(self.plan)

    def test_control_cannot_claim_planned_weight_change(self):
        self.plan["arms"][1]["weight_modification_planned"] = True
        self.assert_held(self.plan)

    def test_sham_baseline_cannot_receive_donor_context(self):
        self.plan["arms"][0]["exposure_guard"] = "ALLOW_DONOR_HINTS"
        self.assert_held(self.plan)

    def test_context_baseline_cannot_retain_context_at_cold_eval(self):
        self.plan["arms"][1]["exposure_guard"] = "ALLOW_RETRIEVAL_AFTER_RESTART"
        self.assert_held(self.plan)

    def test_no_selective_ablation_of_shuffled_or_copy_contrast(self):
        self.plan["required_contrasts"].pop(5)
        self.assert_held(self.plan)

    def test_self_reported_100_percent_result_cannot_be_qualified(self):
        proof = {"claimed_score": 1.0, "signature": "self-signed", "all_arms_pass": True}
        with self.assertRaises(IndependentResultAuthorityUnavailable):
            require_independent_result_qualification(plan_sha256="a" * 64, purported_receipt=proof)

    def test_malformed_surrogate_is_plan_hold_not_unicode_error(self):
        self.assert_held(chr(0xD800))

    def test_cannot_claim_executed_or_authorized(self):
        self.plan["evidence_class"] = "RUNTIME_VERIFIED"
        self.assert_held(self.plan)
        self.plan["evidence_class"] = "SOURCE_ONLY_PREREGISTRATION"
        self.plan["arms"][3]["disposition"] = "PASS"
        self.assert_held(self.plan)

    def test_no_hidden_answers_or_prompts_allowed(self):
        for injected_key in ("heldout_answers", "prompts", "hidden_answer_key"):
            plan = deepcopy(self.plan)
            plan["dataset"][injected_key] = "LEAK"
            self.assert_held(plan)

    def test_separate_split_commitments(self):
        self.plan["dataset"]["sealed_heldout_manifest_sha256"] = self.plan["dataset"]["approved_train_manifest_sha256"]
        self.assert_held(self.plan)

    def test_separate_donor_receiver(self):
        self.plan["models"]["recipient_id"] = self.plan["models"]["donor_id"]
        self.assert_held(self.plan)

    def test_heldout_strata_and_blinding_required(self):
        self.plan["scoring"]["strata"]["self_identification"] = 9
        self.assert_held(self.plan)
        self.plan = example_source_plan()
        self.plan["scoring"]["blind_assignment_commit_sha256"] = "bad"
        self.assert_held(self.plan)

    def test_no_self_grading(self):
        self.plan["scoring"]["grading_policy"] = "DONOR_SELF_GRADE"
        self.assert_held(self.plan)

    def test_cold_receiver_and_disconnected_donor_mandatory(self):
        for key in ("cold_restart_before_heldout", "donor_disconnected_during_heldout",
                    "external_state_disabled_during_heldout", "independent_blinded_scorer"):
            plan = example_source_plan()
            plan["controls"][key] = False
            self.assert_held(plan)

    def test_reject_duplicate_json_keys(self):
        doc = json.dumps(self.plan)
        duplicated = doc.replace('"schema":', '"schema":"bogus", "schema":', 1)
        self.assert_held(duplicated)

    def test_reject_nonfinite_oversized_and_invalid_hash(self):
        self.plan["models"]["tokenizer_sha256"] = "g" * 64
        self.assert_held(self.plan)
        self.assert_held("{" + " " * 65537 + "}")
        self.assert_held('{"n":NaN}')

    def test_reject_unverifiable_host_rights_claim_field(self):
        self.plan["host_authorized"] = True
        self.assert_held(self.plan)


if __name__ == "__main__":
    unittest.main()
