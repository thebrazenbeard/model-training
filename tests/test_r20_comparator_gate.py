"""R20 preregistered comparator gate: CPU-only negative cases."""
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from trainingday.r18_benchmark import canonical_sha256, make_plan, selected_id_digest
from trainingday.r18_qlora import effective_batch_token_loss_scale
from trainingday.r20_comparator_gate import assess_precomparison, main


def _fixture():
    rows = [{"id": f"train-{i}", "split": "train", "training_use": "ADMITTED",
             "prompt": f"Compute {i}+2", "answer": str(i + 2)} for i in range(4)]
    eval_index = [{"id": f"external-{i}", "prompt_sha256": hashlib.sha256(
        f"external-{i}-new".encode()).hexdigest()} for i in range(64)]
    plan = make_plan(
        selected_rows=rows, admission_selected_id_sha256=selected_id_digest(rows),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=eval_index, r16_holdout_ids=[f"r16-{i}" for i in range(64)],
        base_revision="1" * 40, base_binding_sha256="2" * 64,
        tokenizer_binding_sha256="3" * 64, corpus_manifest_sha256="4" * 64,
        eval_custody_sha256="5" * 64, eval_source_manifest_sha256="6" * 64,
        r16_eval_custody_sha256="7" * 64, seed=41,
        training_budget_seconds=531.0, probe_budget_seconds=30.0,
    )
    evidence = {
        "schema": "VERA_TRAININGDAY_R20_PRECOMPARISON_V1",
        "plan_sha256": canonical_sha256(plan),
        "source_head": "a" * 40,
        "source": {
            "objective_contract": "SUPERVISED_TOKEN_WEIGHTED_MEAN_BOTH_ARMS",
            "objective_parity_tests_pass": True,
            "checkpoint_policy": "END_ONLY_NO_RESUME",
            "checkpoint_failure_tests_pass": True,
            "verified_source_head": "a" * 40,
        },
        "data": {
            "corpus_manifest_sha256": plan["corpus_manifest_sha256"],
            "selected_id_sha256": plan["selected_id_sha256"],
            "train_content_sha256": plan["train_content_sha256"],
            "eval_access_before_arm_freeze": False,
        },
        "initial": {
            "mezo_adapter_sha256": "b" * 64,
            "qlora_adapter_sha256": "b" * 64,
            "mezo_trainable_digest": "c" * 64,
            "qlora_trainable_digest": "c" * 64,
            "mezo_base_binding_sha256": plan["base_binding_sha256"],
            "qlora_base_binding_sha256": plan["base_binding_sha256"],
        },
        "host": {
            "device_id": "Lappy", "resource_id": "CUDA0",
            "exclusive_owner_lease": True, "lease_id": "fresh-local-lease",
            "lease_observed_at_unix": 1990, "lease_expires_at_unix": 2100,
            "awake_observed_at_unix": 1995,
            "lid_open": True, "suspend_guard_active": True,
            "standby_since_plan": False,
            "thermal": {"observed_at_unix": 1995,
                        "thermal_warning": False,
                        "thermal_throttling_active": False},
            "gpu_memory": {"observed_at_unix": 1995,
                           "free_bytes": 3_000_000_000,
                           "mezo_fit_peak_bytes": 1_800_000_000,
                           "qlora_fit_peak_bytes": 2_500_000_000,
                           "reserve_bytes": 200_000_000},
            "active_resource_lease_ids": ["fresh-local-lease"],
        },
        "experiment": {
            "fit_mezo_sha256": "d" * 64, "fit_qlora_sha256": "e" * 64,
            "mezo_budget_seconds": 531.0, "qlora_budget_seconds": 531.0,
            "mezo_probe_budget_seconds": 30.0, "qlora_probe_budget_seconds": 30.0,
            "abort_policy": "FAIL_CLOSED_ON_OVERRUN_OR_SUSPEND",
            "arms": ["mezo", "qlora"],
            "controls": ["frozen", "zero", "correct", "corrupt"],
        },
    }
    return plan, evidence


class R20ComparatorGateTests(unittest.TestCase):
    def test_synthetic_ga8_microbatch_loss_equals_mezo_target_weighted_loss(self):
        """Uneven completion lengths cannot make QLoRA average examples equally.

        Both arm formulas must reduce to the same supervised-token-weighted
        scalar for a complete effective eight-example update; no model/eval
        or accelerator is required to test this algebraic source contract.
        """
        target_counts = [1, 7, 3, 9, 2, 5, 4, 12]
        sample_losses = [0.25, 0.9, 1.2, 0.5, 2.0, 1.25, 0.3, 1.7]
        effective = [{"target_tokens": tokens} for tokens in target_counts]
        denominator = sum(target_counts)
        mezo_objective = sum(loss * tokens for loss, tokens in
                             zip(sample_losses, target_counts)) / denominator
        for microbatch_size in (1, 2, 4, 8):
            with self.subTest(microbatch_size=microbatch_size):
                qlora_objective = 0.0
                for start in range(0, 8, microbatch_size):
                    microbatch = effective[start:start + microbatch_size]
                    counts = target_counts[start:start + microbatch_size]
                    losses = sample_losses[start:start + microbatch_size]
                    mean_loss = sum(loss * count for loss, count in
                                    zip(losses, counts)) / sum(counts)
                    qlora_objective += mean_loss * effective_batch_token_loss_scale(
                        microbatch, effective)
                self.assertAlmostEqual(qlora_objective, mezo_objective,
                                       places=13)

    def _invoke_cli(self, plan, evidence):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_file = Path(temp_dir) / "plan.json"
            evidence_file = Path(temp_dir) / "evidence.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            evidence_file.write_text(json.dumps(evidence), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exit_status = main([
                    "--plan", str(plan_file), "--evidence", str(evidence_file),
                    "--now-unix", "2000",
                ])
        return exit_status, json.loads(out.getvalue())

    def test_complete_declarations_have_nonzero_exit_and_explicit_launch_denial(self):
        # Synthetically perfect declarations cannot authorize a CUDA launch.
        plan, evidence = _fixture()
        exit_status, report = self._invoke_cli(plan, evidence)
        self.assertEqual(exit_status, 3)
        self.assertEqual(report["decision_scope"], "DECLARED_EVIDENCE_CONSISTENCY_ONLY")
        self.assertEqual(report["gpu_launch_decision"], "NOT_AUTHORIZED")
        self.assertFalse(report["authorizes_gpu_start"])
        self.assertEqual(report["blockers"], [])

    def test_incomplete_declarations_keep_distinct_failure_exit(self):
        plan, evidence = _fixture()
        evidence["host"]["exclusive_owner_lease"] = False
        exit_status, report = self._invoke_cli(plan, evidence)
        self.assertEqual(exit_status, 2)
        self.assertEqual(report["gpu_launch_decision"], "NOT_AUTHORIZED")
        self.assertIn("HOST_LEASE_UNVERIFIED", report["blockers"])

    def test_complete_synthetic_bundle_is_still_not_gpu_authority(self):
        plan, evidence = _fixture()
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(report["status"], "EVIDENCE_SHAPES_COMPLETE_OWNER_VERIFICATION_REQUIRED")
        self.assertEqual(report["blockers"], [])
        self.assertFalse(report["authorizes_gpu_start"])

    def test_missing_or_stale_host_lease_fails_closed(self):
        plan, evidence = _fixture()
        evidence["host"]["exclusive_owner_lease"] = False
        evidence["host"]["lease_expires_at_unix"] = 1999
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(result["status"], "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
        self.assertIn("HOST_LEASE_UNVERIFIED", result["blockers"])

    def test_standby_and_sleep_cannot_be_subtracted_to_make_valid_budget(self):
        plan, evidence = _fixture()
        evidence["host"]["standby_since_plan"] = True
        evidence["host"]["awake_observed_at_unix"] = 1000
        evidence["host"]["suspend_guard_active"] = False
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("HOST_POWER_UNSAFE", result["blockers"])

    def test_thermal_stress_and_stale_sensor_readout_fail_closed(self):
        for alteration in (
            {"thermal_warning": True},
            {"thermal_throttling_active": True},
            {"observed_at_unix": 1900},
        ):
            with self.subTest(alteration=alteration):
                plan, evidence = _fixture()
                evidence["host"]["thermal"].update(alteration)
                result = assess_precomparison(plan, evidence, now_unix=2000)
                self.assertIn("HOST_THERMAL_UNSAFE", result["blockers"])
                self.assertFalse(result["authorizes_gpu_start"])

    def test_insufficient_fit_margin_and_stale_memory_sample_fail_closed(self):
        for alteration in (
            {"free_bytes": 2_600_000_000},
            {"reserve_bytes": 0},
            {"reserve_bytes": -1},
            {"qlora_fit_peak_bytes": True},
            {"observed_at_unix": 1900},
        ):
            with self.subTest(alteration=alteration):
                plan, evidence = _fixture()
                evidence["host"]["gpu_memory"].update(alteration)
                result = assess_precomparison(plan, evidence, now_unix=2000)
                self.assertIn("GPU_MEMORY_FIT_UNVERIFIED", result["blockers"])
                self.assertFalse(result["authorizes_gpu_start"])

    def test_missing_and_overlapping_gpu_lease_set_fails_closed(self):
        for leases in ([], ["fresh-local-lease", "other-owner"],
                       ["fresh-local-lease", "fresh-local-lease"],
                       ["other-owner"], "fresh-local-lease"):
            with self.subTest(leases=leases):
                plan, evidence = _fixture()
                evidence["host"]["active_resource_lease_ids"] = leases
                result = assess_precomparison(plan, evidence, now_unix=2000)
                self.assertIn("GPU_LEASE_COLLISION_UNVERIFIED", result["blockers"])
                self.assertFalse(result["authorizes_gpu_start"])

    def test_absent_resource_observations_never_count_as_complete(self):
        plan, evidence = _fixture()
        for field in ("thermal", "gpu_memory", "active_resource_lease_ids"):
            evidence["host"].pop(field)
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("HOST_THERMAL_UNSAFE", result["blockers"])
        self.assertIn("GPU_MEMORY_FIT_UNVERIFIED", result["blockers"])
        self.assertIn("GPU_LEASE_COLLISION_UNVERIFIED", result["blockers"])
        self.assertFalse(result["authorizes_gpu_start"])

    def test_mismatched_initial_weights_or_objective_fail_closed(self):
        plan, evidence = _fixture()
        evidence["initial"]["qlora_trainable_digest"] = "f" * 64
        evidence["source"]["objective_contract"] = "PER_EXAMPLE_MEAN"
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("INITIAL_ADAPTER_MISMATCH", result["blockers"])
        self.assertIn("OBJECTIVE_PARITY_UNVERIFIED", result["blockers"])

    def test_wrong_corpus_or_unsealed_eval_fail_closed(self):
        plan, evidence = _fixture()
        evidence["data"]["selected_id_sha256"] = "f" * 64
        evidence["data"]["eval_access_before_arm_freeze"] = True
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("TRAIN_SELECTION_UNBOUND", result["blockers"])

    def test_wrong_budget_controls_or_checkpoint_gate_fail_closed(self):
        plan, evidence = _fixture()
        evidence["experiment"]["qlora_budget_seconds"] = 500
        evidence["experiment"]["controls"] = ["correct"]
        evidence["source"]["checkpoint_failure_tests_pass"] = False
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("BUDGET_OR_CONTROLS_UNMATCHED", result["blockers"])
        self.assertIn("CHECKPOINT_POLICY_UNVERIFIED", result["blockers"])

    def test_malformed_bundle_fails_closed_instead_of_throwing_key_errors(self):
        plan, evidence = _fixture()
        evidence.pop("source")
        self.assertEqual(assess_precomparison(plan, evidence, now_unix=2000)["status"],
                         "HOLD_MISSING_PRECOMPARISON_EVIDENCE")

    def test_nonfinite_unexpected_evidence_is_structured_hold(self):
        plan, evidence = _fixture()
        evidence["additional_observations"] = {"untrusted_metric": math.nan}
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(report["status"], "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])
        self.assertFalse(report["authorizes_gpu_start"])

    def test_unserializable_evidence_is_structured_hold(self):
        plan, evidence = _fixture()
        evidence["additional_observations"] = {"invalid": object()}
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])
        self.assertFalse(report["authorizes_gpu_start"])

    def test_ten_thousand_nested_plain_lists_are_structured_hold(self):
        """A small-node-count deep object must not crash json.dumps hashing."""
        plan, evidence = _fixture()
        nested = "observation"
        for _ in range(10_000):
            nested = [nested]
        evidence["additional_observations"] = nested
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(report["status"], "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])
        self.assertFalse(report["authorizes_gpu_start"])

    def test_deep_evidence_json_cli_yields_machine_readable_hold(self):
        """The JSON decoder's own recursion limit is also fail-closed."""
        plan, evidence = _fixture()
        evidence["additional_observations"] = None
        original = json.dumps(evidence)
        encoded = '"additional_observations": null'
        self.assertIn(encoded, original)
        nested = "[" * 10_000 + "null" + "]" * 10_000
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            evidence_path = Path(directory) / "evidence.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            evidence_path.write_text(original.replace(encoded,
                '"additional_observations": ' + nested), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--plan", str(plan_path), "--evidence",
                                  str(evidence_path), "--now-unix", "2000"])
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])

    def test_depth_boundary_and_cycles_retained_as_distinct_outcomes(self):
        plan, evidence = _fixture()
        nested = "observation"
        # The root evidence object has depth zero, and its observation is at
        # depth one.  Sixty-three nested lists put this leaf at depth 64.
        for _ in range(63):
            nested = [nested]
        evidence["additional_observations"] = nested
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(report["status"],
                         "EVIDENCE_SHAPES_COMPLETE_OWNER_VERIFICATION_REQUIRED")
        self.assertIsNotNone(report["declared_evidence_sha256"])

        evidence["additional_observations"] = [nested]
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])

        cyclic = []
        cyclic.append(cyclic)
        evidence["additional_observations"] = cyclic
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])

    def test_numeric_subclasses_are_not_canonical_evidence(self):
        class ForgedNumber(float):
            def __eq__(self, other):
                return True

        class ForgedInteger(int):
            def __eq__(self, other):
                return True

        for untrusted in (ForgedNumber(1.0), ForgedInteger(1)):
            with self.subTest(untrusted=type(untrusted).__name__):
                plan, evidence = _fixture()
                evidence["additional_observations"] = {"value": untrusted}
                report = assess_precomparison(plan, evidence, now_unix=2000)
                self.assertEqual(report["status"],
                                 "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
                self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
                self.assertIsNone(report["declared_evidence_sha256"])

    def test_forged_mapping_get_cannot_qualify_declared_source(self):
        plan, evidence = _fixture()
        authentic = copy.deepcopy(evidence["source"])

        class ForgedSource(dict):
            def get(self, key, default=None):
                return authentic.get(key, default)

        evidence["source"] = ForgedSource({
            "objective_contract": "WRONG_ON_WIRE",
            "checkpoint_policy": "WRONG_ON_WIRE",
        })
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(report["status"], "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])

    def test_forged_controls_list_never_qualifies(self):
        plan, evidence = _fixture()

        class ForgedControls(list):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        evidence["experiment"]["controls"] = ForgedControls(["correct"])
        report = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertEqual(report["status"], "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])

    def test_huge_now_timestamp_is_rejected_without_overflow(self):
        plan, evidence = _fixture()
        with self.assertRaises(ValueError):
            assess_precomparison(plan, evidence, now_unix=10**400)

    def test_boolean_budget_cannot_equal_numeric_preregistration(self):
        plan, evidence = _fixture()
        plan["training_budget_seconds"] = 1.0
        plan["probe_budget_seconds"] = 0.5
        evidence["plan_sha256"] = canonical_sha256(plan)
        for arm in ("mezo", "qlora"):
            evidence["experiment"][f"{arm}_budget_seconds"] = 1.0
            evidence["experiment"][f"{arm}_probe_budget_seconds"] = 0.5
        evidence["experiment"]["mezo_budget_seconds"] = True
        result = assess_precomparison(plan, evidence, now_unix=2000)
        self.assertIn("BUDGET_OR_CONTROLS_UNMATCHED", result["blockers"])

    def test_plan_dict_subclass_cannot_spoof_validated_fields(self):
        plan, evidence = _fixture()
        authentic = copy.deepcopy(plan)

        class ForgedPlan(dict):
            def get(self, key, default=None):
                return authentic.get(key, default)

        forged = ForgedPlan(plan)
        forged["base_model"] = "WRONG_ON_WIRE"
        evidence["plan_sha256"] = canonical_sha256(forged)
        with self.assertRaises(ValueError):
            assess_precomparison(forged, evidence, now_unix=2000)

    def test_cli_duplicate_evidence_keys_yield_machine_readable_hold(self):
        plan, evidence = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            evidence_path = Path(directory) / "evidence.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            plain = json.dumps(evidence)
            original = '"controls": ["frozen", "zero", "correct", "corrupt"]'
            self.assertIn(original, plain)
            ambiguous = plain.replace(original,
                '"controls": ["correct"], ' + original, 1)
            evidence_path.write_text(ambiguous, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--plan", str(plan_path), "--evidence",
                                  str(evidence_path), "--now-unix", "2000"])
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertIsNone(report["declared_evidence_sha256"])

    def test_cli_nonfinite_json_input_emits_machine_readable_hold(self):
        plan, evidence = _fixture()
        evidence["additional_observations"] = {"untrusted_metric": math.nan}
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            evidence_path = Path(directory) / "evidence.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--plan", str(plan_path), "--evidence",
                                  str(evidence_path), "--now-unix", "2000"])
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "HOLD_MISSING_PRECOMPARISON_EVIDENCE")
        self.assertIn("EVIDENCE_NOT_CANONICAL_JSON", report["blockers"])
        self.assertFalse(report["authorizes_gpu_start"])

if __name__ == "__main__":
    unittest.main()
