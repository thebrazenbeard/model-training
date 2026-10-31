"""R20 preregistered comparator gate: CPU-only negative cases."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from trainingday.r18_benchmark import canonical_sha256, make_plan, selected_id_digest
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
    def _invoke_cli(self, plan, evidence):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_file = Path(temp_dir) / "plan.json"
            evidence_file = Path(temp_dir) / "evidence.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            evidence_file.write_text(json.dumps(evidence), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
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


if __name__ == "__main__":
    unittest.main()
