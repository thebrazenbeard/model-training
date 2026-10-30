"""Pure CPU tests: estimates cannot authorize an R18 rerun or protected eval."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.r18_benchmark import (
    EVAL_SOURCE, FIT_POLICY, MODEL_ID, PLAN_SCHEMA, fit_candidates,
    validate_plan,
)
from trainingday.r22_equal_cost import main, preregister_work_forecast, target_token_mean


def plan():
    return {
        "schema": PLAN_SCHEMA, "base_model": MODEL_ID,
        "base_revision": "a" * 40,
        "base_binding_sha256": "a" * 64,
        "tokenizer_binding_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "selected_id_sha256": "d" * 64, "selected_count": 1024,
        "train_content_sha256": "e" * 64,
        "source_admission_status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
        "eval_bank_source": EVAL_SOURCE,
        "eval_custody_sha256": "f" * 64,
        "eval_source_manifest_sha256": "1" * 64,
        "eval_index_sha256": "2" * 64, "eval_examples": 64,
        "r16_eval_custody_sha256": "3" * 64,
        "r16_holdout_id_sha256": "4" * 64,
        "seed": 17, "training_budget_seconds": 531,
        "probe_budget_seconds": 10,
        "fit_policy": FIT_POLICY, "fit_candidates": fit_candidates(),
        "primary_training_config": fit_candidates()[6],
        "no_eval_access_before_arm_freeze": True,
        "preregistered_checkpoint_selection": "END_OF_TIME_BUDGET_ONLY",
        "promotion_policy": "NEVER_AUTOPROMOTE_R18_REQUIRES_INDEPENDENT_REPLICATION",
    }


def profiles(p):
    common = {
        "plan_sha256": validate_plan(p),
        "selected_id_sha256": p["selected_id_sha256"],
        "train_content_sha256": p["train_content_sha256"],
        "seed": p["seed"],
        "source": "DECLARED_UNTRUSTED_CALIBRATION",
        "data_partition": "TRAIN_ONLY", "eval_access": False,
        "elapsed_seconds": 10, "optimizer_steps": 2,
        "example_presentations": 16, "input_tokens_presented": 320,
        "supervised_target_tokens": 32,
    }
    return [
        {**common, "arm": "mezo", "forward_sample_tokens": 640,
         "backward_sample_tokens": 0, "padded_forward_tokens": 768,
         "padded_backward_tokens": 0},
        {**common, "arm": "qlora", "forward_sample_tokens": 320,
         "backward_sample_tokens": 320, "padded_forward_tokens": 384,
         "padded_backward_tokens": 384},
    ]


class R22EqualCostTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan()
        self.profiles = profiles(self.plan)

    def test_target_weighting_differs_from_microbatch_mean(self):
        # Old QLoRA micro-loss averaging gave 5; R19 common target objective 6.5.
        self.assertEqual(target_token_mean([2, 8], [1, 3]), 6.5)
        self.assertNotEqual(target_token_mean([2, 8], [1, 3]), (2 + 8) / 2)

    def test_common_budget_and_unequal_token_work_not_confused(self):
        report = preregister_work_forecast(self.plan, self.profiles)
        self.assertEqual(report["equal_optimizer_wall_budget_seconds"],
                         {"mezo": 531, "qlora": 531})
        self.assertEqual(report["controls"], ["frozen", "zero", "correct", "corrupt"])
        self.assertEqual(report["objective"], "SUPERVISED_TARGET_TOKEN_WEIGHTED_MEAN_BOTH_ARMS")
        m = report["forecast"]["mezo"]["estimated_at_equal_optimizer_wall_time"]
        q = report["forecast"]["qlora"]["estimated_at_equal_optimizer_wall_time"]
        self.assertEqual(m["forward_sample_tokens"], 33984)
        self.assertEqual(q["forward_sample_tokens"], 16992)
        self.assertEqual(m["backward_sample_tokens"], 0)
        self.assertEqual(q["backward_sample_tokens"], 16992)
        self.assertFalse(report["equal_sample_token_exposure_asserted"])
        for flag in ("authorizes_gpu_start", "model_improvement_proven",
                     "promotion_allowed", "historical_r18_overrun_comparison_valid"):
            self.assertFalse(report[flag])

    def test_order_independent_hash_and_forecast(self):
        self.assertEqual(preregister_work_forecast(self.plan, self.profiles),
                         preregister_work_forecast(self.plan, list(reversed(self.profiles))))

    def test_misbound_or_duplicated_controls_rejected(self):
        for mutation in (
            lambda a: a[1].update(arm="mezo"),
            lambda a: a[0].update(seed=21),
            lambda a: a[0].update(selected_id_sha256="0" * 64),
            lambda a: a[1].update(train_content_sha256="0" * 64),
            lambda a: a[1].update(source="INDEPENDENT_PHYSICAL_ATTESTATION"),
        ):
            with self.subTest(mutation=repr(mutation)):
                bad = copy.deepcopy(self.profiles)
                mutation(bad)
                with self.assertRaises(AdmissionError):
                    preregister_work_forecast(self.plan, bad)

    def test_private_eval_access_or_unregistered_schema_rejected(self):
        for mutation in (
            lambda a: a[0].update(eval_access=True),
            lambda a: a[1].update(data_partition="PRIVATE_EVAL"),
            lambda a: a[0].update(host_lease_authenticated=True),
        ):
            bad = copy.deepcopy(self.profiles)
            mutation(bad)
            with self.assertRaises(AdmissionError):
                preregister_work_forecast(self.plan, bad)

    def test_probe_or_historical_wall_overrun_rejected(self):
        for elapsed in (10.0001, 19813.562, float("nan"), 0, -1, 10**1000):
            with self.subTest(elapsed=elapsed):
                bad = copy.deepcopy(self.profiles)
                bad[0]["elapsed_seconds"] = elapsed
                with self.assertRaises(AdmissionError):
                    preregister_work_forecast(self.plan, bad)

    def test_subnormal_elapsed_and_overflowing_projection_fail_closed(self):
        for elapsed, padded in ((5e-324, 768), (1e-300, 2**53)):
            with self.subTest(elapsed=elapsed, padded=padded):
                bad = copy.deepcopy(self.profiles)
                bad[0]["elapsed_seconds"] = elapsed
                bad[0]["padded_forward_tokens"] = padded
                with self.assertRaises(AdmissionError):
                    preregister_work_forecast(self.plan, bad)

    def test_subnormal_cli_produces_strict_json_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            plan_path, profiles_path = base / "plan.json", base / "profiles.json"
            plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
            bad = copy.deepcopy(self.profiles)
            bad[0]["elapsed_seconds"] = 5e-324
            profiles_path.write_text(json.dumps(bad), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--plan", str(plan_path), "--profiles", str(profiles_path)])
            self.assertEqual(code, 2)
            report = json.loads(output.getvalue(), parse_constant=lambda c: self.fail(c))
            self.assertEqual(report["status"], "HOLD_INVALID_DECLARED_FORECAST")
            self.assertFalse(report["authorizes_gpu_start"])

    def test_algorithm_physical_token_accounting_rejected_when_impossible(self):
        for arm, change in (
            (0, {"forward_sample_tokens": 319}),
            (0, {"backward_sample_tokens": 320}),
            (0, {"padded_backward_tokens": 1}),
            (1, {"backward_sample_tokens": 0}),
            (1, {"padded_backward_tokens": 319}),
            (1, {"padded_forward_tokens": 10**1000}),
        ):
            bad = copy.deepcopy(self.profiles)
            bad[arm].update(change)
            with self.subTest(arm=arm, change=change), self.assertRaises(AdmissionError):
                preregister_work_forecast(self.plan, bad)

    def test_loss_zero_counts_or_nonnumeric_rejected(self):
        for losses, counts in (([2, 8], [0, 3]), ([2], [1, 3]),
                               ([float("inf"), 8], [1, 3]), ([2, 8], [True, 3]),
                               ([10**1000, 8], [1, 3]), ([2, 8], [10**1000, 3])):
            with self.assertRaises(AdmissionError):
                target_token_mean(losses, counts)

    def test_cli_complete_forecast_still_exits_non_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            p, c = base / "plan.json", base / "profiles.json"
            p.write_text(json.dumps(self.plan), encoding="utf-8")
            c.write_text(json.dumps(self.profiles), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--plan", str(p), "--profiles", str(c)])
            report = json.loads(output.getvalue())
            self.assertEqual(code, 3)
            self.assertFalse(report["authorizes_gpu_start"])
            bad = copy.deepcopy(self.profiles)
            bad[0]["elapsed_seconds"] = 19813.562
            c.write_text(json.dumps(bad), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--plan", str(p), "--profiles", str(c)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"],
                             "HOLD_INVALID_DECLARED_FORECAST")


if __name__ == "__main__":
    unittest.main()
