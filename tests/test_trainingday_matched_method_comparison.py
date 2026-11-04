"""Source-only adversarial contract tests; no model, GPU, or private DEV read."""

import copy
import hashlib
import json
import unittest

from trainingday.matched_method_comparison import (
    MatchedStudyError, build_study_plan, summarize_paired_binary_outcomes,
    validate_untrusted_receipts,
)


def spec():
    return dict(
        official_weight_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        train_sha256="3" * 64,
        private_dev_commitment_sha256="4" * 64,
        lora_target_spec_sha256="5" * 64,
        seeds=(11, 13, 17),
        wall_clock_budget_seconds=90,
    )


def receipts(plan):
    result = []
    for arm in plan["arms"]:
        zero = arm["label_mode"] == "zero"
        base = arm["method"] == "OFFICIAL_BASE"
        result.append(dict(
            arm_id=arm["arm_id"], plan_sha256=plan["plan_sha256"],
            elapsed_ms=80_000, optimizer_steps=0 if zero else 10,
            examples_seen=0 if zero or base else 100,
            tokens_seen=0 if zero or base else 1000,
            forward_calls=1 if base else 0 if zero else 20,
            backward_calls=0 if zero or base or arm["method"] == "MEZO_LORA" else 10,
            peak_vram_bytes=2_700_000_000,
            peak_host_ram_bytes=3_200_000_000,
            checkpoint_reloaded=True, interruption_resumed=not zero and not base,
            adapter_sha256=None if base else "7" * 64,
            loaded_weight_sha256=plan["official_weight_sha256"],
            train_order_sha256=None if zero or base else "6" * 64,
        ))
        if base:
            result[-1]["optimizer_steps"] = 0
    return result


def scores(ids, successful):
    return {f"case-{i:016x}": i in successful for i in range(ids)}


class MatchedMethodStudyTests(unittest.TestCase):
    def test_plan_has_seed_paired_methods_three_label_controls_and_hard_hold(self):
        a = build_study_plan(**spec())
        self.assertEqual(a, build_study_plan(**spec()))
        self.assertEqual(len(a["arms"]), 21)
        self.assertEqual(len({arm["arm_id"] for arm in a["arms"]}), 21)
        for seed in (11, 13, 17):
            subset = [x for x in a["arms"] if x["seed"] == seed
                      and x["method"] != "OFFICIAL_BASE"]
            self.assertEqual({x["method"] for x in subset},
                             {"MEZO_LORA", "QLORA_ADAMW"})
            self.assertEqual({x["label_mode"] for x in subset},
                             {"correct", "corrupted", "zero"})
            references = [x for x in a["arms"] if x["seed"] == seed
                          and x["method"] == "OFFICIAL_BASE"]
            self.assertEqual(len(references), 1)
            self.assertEqual(references[0]["trainable_subspace"],
                             "NO_ADAPTER_UNMODIFIED_OFFICIAL_BASE")
        for k in ("source_rights_admitted", "host_owned_lease_verified",
                  "protected_evaluator_custody_verified", "authorizes_training",
                  "authorizes_eval_unseal", "allow_full_parameter_mezo_as_matched_arm"):
            self.assertIs(a[k], False)
        self.assertTrue(a["base_zero_update_reference_required"])

    def test_plan_rejects_spoofed_hashes_seeds_and_bounds(self):
        for key, wrong in (("official_weight_sha256", "z" * 64),
                           ("tokenizer_sha256", True),
                           ("train_sha256", "4" * 64),
                           ("seeds", [1, 1, 2]),
                           ("seeds", [1, 2]),
                           ("seeds", [1, True, 3]),
                           ("seeds", [-1, 2, 3]),
                           ("wall_clock_budget_seconds", True),
                           ("wall_clock_budget_seconds", 10)):
            proposed = spec()
            proposed[key] = wrong
            with self.subTest(key=key, wrong=wrong), self.assertRaises(
                    MatchedStudyError):
                build_study_plan(**proposed)

    def test_receipt_shape_covers_all_arms_but_does_not_authenticate(self):
        plan = build_study_plan(**spec())
        result = validate_untrusted_receipts(plan, receipts(plan))
        self.assertEqual(result["arms_checked"], 21)
        self.assertTrue(result["covers_all_declared_arms"])
        for k in ("physical_execution_independently_verified", "data_rights_verified",
                  "protected_eval_verified", "proves_learning"):
            self.assertIs(result[k], False)

    def test_receipts_reject_plan_tamper_and_duplicate_or_missing_arm(self):
        plan = build_study_plan(**spec())
        baseline = receipts(plan)
        tampered = copy.deepcopy(plan)
        tampered["equal_wall_clock_budget_seconds"] = 400
        candidates = [
            (plan, baseline[:-1]),
            (plan, baseline[:-1] + [baseline[0]]),
            (tampered, baseline),
        ]
        for p, r in candidates:
            with self.subTest(candidate=r[-1]["arm_id"]), self.assertRaises(
                    MatchedStudyError):
                validate_untrusted_receipts(p, r)

    def test_rehashed_incomplete_plan_or_forged_positive_authority_is_rejected(self):
        original = build_study_plan(**spec())
        for change in ("omit_method", "omit_base", "forge_authority"):
            plan = copy.deepcopy(original)
            if change == "omit_method":
                plan["arms"] = [x for x in plan["arms"] if x["method"] != "MEZO_LORA"]
            elif change == "omit_base":
                plan["arms"] = [x for x in plan["arms"] if x["method"] != "OFFICIAL_BASE"]
            else:
                plan["authorizes_training"] = True
                plan["protected_evaluator_custody_verified"] = True
            plan.pop("plan_sha256")
            plan["plan_sha256"] = hashlib.sha256(json.dumps(
                plan, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest()
            with self.subTest(change=change), self.assertRaises(MatchedStudyError):
                validate_untrusted_receipts(plan, receipts(original))

    def test_untrusted_receipt_does_not_allow_private_data_or_fake_work(self):
        plan = build_study_plan(**spec())
        for field, value in (
            ("dev_prompt", "private information"),
            ("optimizer_steps", True),
            ("elapsed_ms", 90_001),
            ("checkpoint_reloaded", False),
            ("interruption_resumed", False),
            ("examples_seen", 0),
            ("tokens_seen", 0),
            ("forward_calls", 0),
            ("peak_host_ram_bytes", 0),
            ("peak_vram_bytes", 0),
            ("train_order_sha256", "bad"),
        ):
            r = receipts(plan)
            record = next(x for x in r if ":correct:" in x["arm_id"])
            record[field] = value
            with self.subTest(field=field), self.assertRaises(MatchedStudyError):
                validate_untrusted_receipts(plan, r)
        r = receipts(plan)
        zero = next(x for x in r if ":zero:" in x["arm_id"])
        zero["optimizer_steps"] = 1
        with self.assertRaises(MatchedStudyError):
            validate_untrusted_receipts(plan, r)
        r = receipts(plan)
        mezo = next(x for x in r if x["arm_id"].startswith("MEZO_LORA:correct:"))
        mezo["backward_calls"] = 1
        with self.assertRaises(MatchedStudyError):
            validate_untrusted_receipts(plan, r)
        r = receipts(plan)
        mezo = next(x for x in r if x["arm_id"].startswith("MEZO_LORA:correct:"))
        mezo["forward_calls"] = 1
        with self.assertRaises(MatchedStudyError):
            validate_untrusted_receipts(plan, r)
        r = receipts(plan)
        qlora = next(x for x in r if x["arm_id"].startswith("QLORA_ADAMW:correct:"))
        qlora["backward_calls"] = 1
        with self.assertRaises(MatchedStudyError):
            validate_untrusted_receipts(plan, r)

    def test_cross_method_exposure_and_unmodified_base_identity_enforced(self):
        plan = build_study_plan(**spec())
        for field, wrong in (("examples_seen", 10000), ("tokens_seen", 42),
                             ("train_order_sha256", "8" * 64)):
            r = receipts(plan)
            m = next(x for x in r if x["arm_id"] == "MEZO_LORA:correct:11")
            m[field] = wrong
            with self.subTest(field=field), self.assertRaises(MatchedStudyError):
                validate_untrusted_receipts(plan, r)
        r = receipts(plan)
        base = next(x for x in r if x["arm_id"] == "OFFICIAL_BASE:reference:11")
        base["adapter_sha256"] = "9" * 64
        with self.assertRaises(MatchedStudyError):
            validate_untrusted_receipts(plan, r)

    def test_mcnemar_is_paired_not_deducible_from_marginals(self):
        # Same historical 3/64 vs 5/64 totals; different case identities
        # produce different discordance and exact McNemar probabilities.
        base = scores(64, {0, 1, 2})
        nested = summarize_paired_binary_outcomes(base, scores(64, {0, 1, 2, 3, 4}))
        disjoint = summarize_paired_binary_outcomes(base, scores(64, {3, 4, 5, 6, 7}))
        self.assertEqual((nested["reference_correct"], nested["challenger_correct"]), (3, 5))
        self.assertEqual((disjoint["reference_correct"], disjoint["challenger_correct"]), (3, 5))
        self.assertEqual(nested["mcnemar_exact_two_sided_p"], 0.5)
        self.assertNotEqual(nested["mcnemar_exact_two_sided_p"],
                            disjoint["mcnemar_exact_two_sided_p"])
        self.assertFalse(nested["generalization_proven"])
        self.assertFalse(disjoint["independent_eval_verified"])
        self.assertNotIn("case-", str(nested))  # no private case IDs in output

    def test_paired_outcomes_reject_case_identity_leakage_and_bad_shape(self):
        valid = scores(64, {0, 1, 2})
        for wrong in (
            scores(63, {0, 1}),
            {**valid, "prompt=secret answer=5": True},
            {**valid, "case-0000000000000000": 1},
            {**valid, "case-0000000000000000": "correct"},
            {f"case-{i:016x}": False for i in range(31)},
        ):
            with self.subTest(kind=len(wrong)), self.assertRaises(MatchedStudyError):
                summarize_paired_binary_outcomes(valid, wrong)


if __name__ == "__main__":
    unittest.main()
