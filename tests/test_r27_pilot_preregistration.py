"""Public, answer-free pilot protocol tests; NEVER a private OOD evaluator."""

import hashlib
import builtins
import unittest
from dataclasses import replace
from unittest.mock import patch

import trainingday.r27_pilot_preregistration as pilot_module

from trainingday.r27_pilot_preregistration import (
    PilotCase, PilotProtocol, PilotError, preregister_pilot,
    score_synthetic_pilot,
)


def h(s):
    return hashlib.sha256(s.encode("ascii")).hexdigest()


class PilotPreregistrationTests(unittest.TestCase):
    def setUp(self):
        families = ("ARITHMETIC", "PROGRAM_REASONING", "SOURCE_PRIORITY")
        self.cases = tuple(
            PilotCase(
                case_id=f"pilot-case-{i:03d}",
                family=families[i % 3],
                prompt_sha256=h(f"future-public-prompt-{i}"),
                rights_evidence_sha256=h(f"rights-reference-{i}"),
            ) for i in range(120)
        )
        self.spec = PilotProtocol(
            protocol_id="pilot-r27-public-001",
            frozen_base_sha256=h("frozen-base"),
            tokenizer_sha256=h("tokenizer"),
            admitted_train_corpus_sha256=h("admitted-train-corpus"),
            known_training_prompt_sha256=(h("known-train-prompt"),),
            cases=self.cases,
            opaque_arm_ids=("blind-A01", "blind-B02", "blind-C03", "blind-D04"),
            min_paired_gain=0.10,
            one_sided_alpha=0.05,
        )
        self.receipt = preregister_pilot(self.spec)
        self.roles = {
            "blind-A01": "FROZEN",
            "blind-B02": "ZERO_UPDATE",
            "blind-C03": "CORRECT_LABEL",
            "blind-D04": "CORRUPTED_LABEL",
        }

    def outputs(self, *, positive=True):
        rows = [c.case_id for c in self.cases]
        baseline = {k: (i % 2 == 0) for i, k in enumerate(rows)}
        good = {k: (i % 10 != 0) for i, k in enumerate(rows)} if positive else baseline
        corrupt = {k: (i % 3 == 0) for i, k in enumerate(rows)}
        return {
            "blind-A01": baseline,
            "blind-B02": dict(baseline),
            "blind-C03": good,
            "blind-D04": corrupt,
        }

    def test_public_answer_free_manifest_is_deterministic(self):
        again = preregister_pilot(replace(self.spec, cases=tuple(reversed(self.cases))))
        self.assertEqual(again.manifest_sha256, self.receipt.manifest_sha256)
        self.assertEqual(again.case_count, 120)
        self.assertTrue(again.declared_training_overlap_absent)
        self.assertFalse(again.training_inventory_complete_attested)
        self.assertFalse(again.rights_authenticated)
        self.assertFalse(again.blind_allocation_authenticated)
        self.assertFalse(again.release_authorized)
        self.assertNotIn("answer", self.receipt.canonical_json)
        self.assertNotIn("prompt_text", self.receipt.canonical_json)
        self.assertNotIn("future-public-prompt", self.receipt.canonical_json)

    def test_public_synthetic_threshold_only_never_neural_claim(self):
        result = score_synthetic_pilot(self.spec, self.receipt, self.roles, self.outputs())
        self.assertEqual(result.status, "SYNTHETIC_THRESHOLD_MET_NO_CLAIM")
        self.assertGreater(result.correct_vs_frozen_gain, 0.30)
        self.assertLess(result.correct_vs_frozen_p_value, 0.05)
        self.assertEqual(len(result.by_family_accuracy), 3)
        self.assertFalse(result.physical_neural_improvement_demonstrated)
        self.assertFalse(result.protected_eval_access_authorized)
        self.assertFalse(result.model_promotion_authorized)

    def test_no_gain_and_corrupted_control_cannot_pass(self):
        r = score_synthetic_pilot(self.spec, self.receipt, self.roles, self.outputs(positive=False))
        self.assertEqual(r.status, "NO_IMPROVEMENT")
        outputs = self.outputs()
        outputs["blind-D04"] = dict(outputs["blind-C03"])
        r = score_synthetic_pilot(self.spec, self.receipt, self.roles, outputs)
        self.assertEqual(r.status, "NO_IMPROVEMENT")

    def test_three_control_familywise_adjustment_blocks_marginal_p_value(self):
        # Ten wins/two losses among 120 paired examples give p≈0.0193:
        # significant at .05, but NOT at Bonferroni-adjusted .05/3.
        baseline = {c.case_id: (i % 2 == 0) for i, c in enumerate(self.cases)}
        improved = dict(baseline)
        for i in range(1, 20, 2):
            improved[self.cases[i].case_id] = True
        for i in (0, 2):
            improved[self.cases[i].case_id] = False
        outputs = {
            "blind-A01": dict(baseline), "blind-B02": dict(baseline),
            "blind-C03": improved, "blind-D04": dict(baseline),
        }
        spec = replace(self.spec, min_paired_gain=0.05)
        result = score_synthetic_pilot(spec, preregister_pilot(spec), self.roles, outputs)
        self.assertGreater(result.correct_vs_frozen_p_value, 0.05 / 3)
        self.assertLess(result.correct_vs_frozen_p_value, 0.05)
        self.assertEqual(result.status, "NO_IMPROVEMENT")

    def test_overall_gain_cannot_hide_critical_family_regression(self):
        # Overall candidate improves, and every family clears absolute .50,
        # but arithmetic loses 18/40 previously correct cases: no claim.
        arith_ids = [c.case_id for c in self.cases if c.family == "ARITHMETIC"]
        baseline = {c.case_id: c.family == "ARITHMETIC" for c in self.cases}
        improved = {c.case_id: c.family != "ARITHMETIC" for c in self.cases}
        for k in arith_ids[:22]:
            improved[k] = True
        outputs = {
            "blind-A01": baseline, "blind-B02": dict(baseline),
            "blind-C03": improved,
            "blind-D04": {c.case_id: False for c in self.cases},
        }
        result = score_synthetic_pilot(self.spec, self.receipt, self.roles, outputs)
        self.assertGreater(result.correct_vs_frozen_gain, 0.30)
        self.assertEqual(result.status, "NO_IMPROVEMENT")

    def test_known_train_overlap_and_internal_duplicate_rejected(self):
        self.assertRaises(PilotError, preregister_pilot, replace(
            self.spec, known_training_prompt_sha256=(self.cases[0].prompt_sha256,)
        ))
        self.assertRaises(PilotError, preregister_pilot, replace(
            self.spec, cases=self.cases[:-1] + (replace(
                self.cases[-1], prompt_sha256=self.cases[0].prompt_sha256
            ),)
        ))

    def test_ambiguous_scope_and_unreviewed_rights_cannot_be_claimed_authenticated(self):
        self.assertRaises(PilotError, preregister_pilot, replace(
            self.spec, cases=self.cases[:-1] + (replace(self.cases[-1], family="EMPATHY"),)
        ))
        self.assertRaises(PilotError, preregister_pilot, replace(
            self.spec, cases=self.cases[:-1] + (replace(self.cases[-1], rights_evidence_sha256=""),)
        ))

    def test_tampered_plan_against_precommit_rejected(self):
        altered = replace(self.spec, min_paired_gain=0.0)
        with self.assertRaises(PilotError):
            score_synthetic_pilot(altered, self.receipt, self.roles, self.outputs())

    def test_unblinded_role_mapping_and_case_count_rejected(self):
        with self.assertRaises(PilotError):
            score_synthetic_pilot(self.spec, self.receipt,
                                  {**self.roles, "cheat": "FROZEN"}, self.outputs())
        with self.assertRaises(PilotError):
            score_synthetic_pilot(self.spec, self.receipt, self.roles,
                                  {**self.outputs(), "blind-A01": {}})

    def test_non_boolean_results_rejected_before_scoring(self):
        outputs = self.outputs()
        outputs["blind-C03"][self.cases[0].case_id] = 1
        with self.assertRaises(PilotError):
            score_synthetic_pilot(self.spec, self.receipt, self.roles, outputs)

    def test_oversized_per_arm_map_rejected_without_expensive_item_iteration(self):
        outputs = self.outputs()
        # First ordered arm avoids unrelated valid-arm iteration preceding the
        # adversarial one; the old source still calls all() on 100k entries.
        outputs["blind-A01"].update({f"extra-{i:06d}": False for i in range(100_000)})
        calls = 0

        def sentinel_all(iterable):
            nonlocal calls
            calls += 1
            if calls > 2:  # role-map and outer-arm checks may use all()
                raise AssertionError("EXPENSIVE_PER_ARM_ITERATION_BEFORE_SIZE_GATE")
            return builtins.all(iterable)

        with (patch.object(pilot_module, "preregister_pilot", return_value=self.receipt),
              patch.object(pilot_module, "all", side_effect=sentinel_all, create=True)):
            with self.assertRaisesRegex(PilotError, "UNPAIRED_OR_MALFORMED_CASE_OUTCOMES"):
                score_synthetic_pilot(self.spec, self.receipt, self.roles, outputs)
        self.assertLessEqual(calls, 2)

    def test_oversized_outer_arm_map_rejected_before_key_iteration(self):
        outputs = self.outputs()
        outputs.update({f"extra-arm-{i:06d}": {} for i in range(100_000)})
        calls = 0

        def sentinel_all(iterable):
            nonlocal calls
            calls += 1
            if calls > 1:  # valid role-map only; outer large map must not be iterated
                raise AssertionError("EXPENSIVE_OUTER_ITERATION_BEFORE_SIZE_GATE")
            return builtins.all(iterable)

        with (patch.object(pilot_module, "preregister_pilot", return_value=self.receipt),
              patch.object(pilot_module, "all", side_effect=sentinel_all, create=True)):
            with self.assertRaisesRegex(PilotError, "MISSING_OR_EXTRA_ARM_OUTCOMES"):
                score_synthetic_pilot(self.spec, self.receipt, self.roles, outputs)
        self.assertEqual(calls, 1)

    def test_oversized_role_map_rejected_before_role_item_iteration(self):
        roles = dict(self.roles)
        roles.update({f"extra-role-{i:06d}": "CORRECT_LABEL" for i in range(100_000)})

        def sentinel_all(_iterable):
            raise AssertionError("EXPENSIVE_ROLE_ITERATION_BEFORE_SIZE_GATE")

        with (patch.object(pilot_module, "preregister_pilot", return_value=self.receipt),
              patch.object(pilot_module, "all", side_effect=sentinel_all, create=True)):
            with self.assertRaisesRegex(PilotError, "ARM_MAPPING_MALFORMED_OR_UNBLINDED"):
                score_synthetic_pilot(self.spec, self.receipt, roles, self.outputs())

    def test_nonfinite_threshold_and_extreme_values_controlled(self):
        for value in (float("nan"), float("inf"), 10**1000, 0.0, 1.1, True):
            with self.subTest(value=str(value)[:12]):
                with self.assertRaises(PilotError):
                    preregister_pilot(replace(self.spec, one_sided_alpha=value))


if __name__ == "__main__":
    unittest.main()
