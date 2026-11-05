"""Original public synthetic strings only; no model inference or held-out data."""

import unittest

from trainingday.public_multiturn_controls import (
    ARMS, BUDGETS, CASES, CausalControlError, PUBLIC_PLAN_SCHEMA,
    audit_public_study, public_plan,
)


def fixture_rows(*, censor=False):
    rows = []
    adapters = {
        "base": "NO_ADAPTER", "persona_base": "NO_ADAPTER",
        "r16_zero_step": "b" * 64, "r16_correct8": "c" * 64,
        "r16_corrupt8": "d" * 64,
    }
    for budget in BUDGETS:
        for case_id, script in CASES.items():
            for arm in ARMS:
                rows.append({
                    "arm": arm,
                    "case_id": f"{case_id}_b{budget}",
                    "session_id": f"{arm}_{case_id}_b{budget}",
                    "source_claim": "mock_test",
                    "origin_claim": "fake Qwen model — synthetics only",
                    "base_sha256": "a" * 64,
                    "adapter_sha256": adapters[arm],
                    "prompt_sha256": "e" * 64 if arm == "base" else "f" * 64,
                    "generation": {"context_window_tokens": 4096,
                                   "max_reply_tokens": budget,
                                   "temperature_milli": 700,
                                   "rng_seed": 17},
                    "turns": [{"user": prompt,
                               "assistant": None if censor else "mock output, not Vera",
                               "outcome": "timeout" if censor else "complete"}
                              for prompt in script],
                })
    return rows


class PublicMultiTurnControlsTests(unittest.TestCase):
    def test_public_plan_five_arms_three_budgets_two_fixed_cases(self):
        plan = public_plan()
        self.assertEqual(plan["schema"], PUBLIC_PLAN_SCHEMA)
        self.assertEqual(plan["planned_cell_count"], 30)
        self.assertEqual(plan["arm_roles"], list(ARMS))
        self.assertEqual(plan["reply_budgets"], list(BUDGETS))
        self.assertEqual(len(plan["public_case_scripts"]), 2)
        self.assertEqual(plan["physical_inference_runs"], 0)
        self.assertIsNone(plan["semantic_quality_score"])
        self.assertFalse(plan["heldout"])
        self.assertFalse(plan["training_authorized"])

    def test_feedback_script_has_explicit_correction_only_in_feedback_arm(self):
        self.assertIn("Correction:", CASES["public_feedback"][1])
        self.assertTrue(all("Correction:" not in prompt for prompt in CASES["public_control"]))
        self.assertTrue(all(len(script) == 3 for script in CASES.values()))
        self.assertEqual(CASES["public_feedback"][0], CASES["public_control"][0])
        self.assertEqual(CASES["public_feedback"][2], CASES["public_control"][2])

    def test_mock_records_only_validate_declarations(self):
        report = audit_public_study(fixture_rows())
        self.assertEqual(report["declared_slots_checked"], 30)
        self.assertEqual(report["declared_completed_turns"], 90)
        self.assertEqual(report["source_claim"], "mock_test")
        self.assertFalse(report["authentic_model_inference_attested"])
        self.assertFalse(report["prompt_profile_physically_applied_attested"])
        self.assertFalse(report["semantic_improvement_measured"])
        self.assertFalse(report["protected_holdout_used"])
        self.assertTrue(report["declared_same_base_digest_origin_claim_and_sampling"])
        self.assertFalse(report["official_base_revision_attested"])
        self.assertNotIn("declared_same_base_revision_and_sampling", report)

    def test_censored_mock_attempts_cannot_count_as_model_success(self):
        report = audit_public_study(fixture_rows(censor=True))
        self.assertEqual(report["declared_completed_turns"], 0)
        self.assertEqual(report["declared_technical_censored_turns"], 90)
        self.assertFalse(report["semantic_improvement_measured"])

    def test_different_base_hash_among_arms_breaks_causal_control(self):
        rows = fixture_rows()
        for row in rows:
            if row["arm"] == "r16_correct8":
                row["base_sha256"] = "9" * 64
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_missing_corrupted_label_control_breaks_causal_control(self):
        rows = [r for r in fixture_rows() if r["arm"] != "r16_corrupt8"]
        with self.assertRaises(ValueError):
            audit_public_study(rows)

    def test_missing_zero_step_control_breaks_causal_control(self):
        rows = [r for r in fixture_rows() if r["arm"] != "r16_zero_step"]
        with self.assertRaises(ValueError):
            audit_public_study(rows)

    def test_persona_prompt_claim_must_match_all_adapter_arms(self):
        rows = fixture_rows()
        for row in rows:
            if row["arm"] == "r16_corrupt8":
                row["prompt_sha256"] = "1" * 64
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_base_without_profile_must_differ_from_persona(self):
        rows = fixture_rows()
        for row in rows:
            if row["arm"] == "base":
                row["prompt_sha256"] = "f" * 64
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_adapter_arms_must_have_distinct_declared_digest(self):
        rows = fixture_rows()
        for row in rows:
            if row["arm"] == "r16_corrupt8":
                row["adapter_sha256"] = "c" * 64
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_baseline_adapter_cannot_be_nonzero(self):
        rows = fixture_rows()
        for row in rows:
            if row["arm"] == "base":
                row["adapter_sha256"] = "b" * 64
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_altered_second_turn_breaks_script_and_pairing(self):
        rows = fixture_rows()
        rows[1]["turns"][1]["user"] = "changed cue"
        with self.assertRaises(ValueError):
            audit_public_study(rows)

    def test_mismatched_sampling_seed_breaks_study(self):
        rows = fixture_rows()
        rows[1]["generation"]["rng_seed"] = 999
        with self.assertRaises(ValueError):
            audit_public_study(rows)

    def test_missing_128_budget_breaks_study(self):
        rows = [r for r in fixture_rows() if r["generation"]["max_reply_tokens"] != 128]
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_mismatched_budget_case_id_breaks_study(self):
        rows = fixture_rows()
        rows[0]["generation"]["max_reply_tokens"] = 96
        with self.assertRaises(ValueError):
            audit_public_study(rows)

    def test_cannot_mix_mock_and_claimed_public_inference(self):
        rows = fixture_rows()
        for row in rows:
            if row["arm"] == "persona_base":
                row["source_claim"] = "public_exposed"
        with self.assertRaises(ValueError):
            audit_public_study(rows)

    def test_private_unverified_source_is_not_public_fixture(self):
        rows = fixture_rows()
        for row in rows:
            row["source_claim"] = "private_unverified"
        with self.assertRaises(CausalControlError):
            audit_public_study(rows)

    def test_capture_does_not_emit_private_or_mock_assistant_text(self):
        report = audit_public_study(fixture_rows())
        self.assertNotIn("mock output", repr(report))
        self.assertNotIn("fake Qwen", repr(report))
        self.assertNotIn("scores", report)


if __name__ == "__main__":
    unittest.main()
