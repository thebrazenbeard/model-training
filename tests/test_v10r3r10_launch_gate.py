"""R10 Qwen-only guarded launch and memory-gate tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from successor.experiments.preflight_v10r3r10_launch import (
    classify, validate_subject, EXPECTED_CORPUS_SHA, EXPECTED_R9_SHA,
)


def facts(**overrides):
    result = {
        "model_type": "qwen3_5",
        "adapter_sha": EXPECTED_R9_SHA,
        "corpus_sha": EXPECTED_CORPUS_SHA,
        "free_ram_gib": 14,
        "free_virtual_gib": 16,
        "free_gpu_mib": 3962,
        "live_trainer_count": 0,
        "output_already_exists": False,
        "step_count": 8,
    }
    result.update(overrides)
    return result


class QwenOnlyLaunchTests(unittest.TestCase):
    def test_healthy_exact_qwen_r9_8steps_only_ready_for_local_run(self):
        item = classify(**facts())
        self.assertEqual(item["status"], "READY_FOR_R10_LOCAL_ONLY")
        self.assertEqual(item["optimizer_steps_executed"], 0)
        self.assertFalse(item["weights_changed"])
        self.assertFalse(item["deployed"])

    def test_wrong_dolphin_model_never_trains(self):
        item = classify(**facts(model_type="mistral"))
        self.assertEqual(item["status"], "HOLD_NO_WEIGHT_CHANGE")
        self.assertIn("WRONG_MODEL_NOT_QWEN35", item["reasons"])

    def test_wrong_parent_or_training_corpus_never_trains(self):
        for change, reason in [
            ({"adapter_sha": "0" * 64}, "WRONG_PARENT_NOT_R9"),
            ({"corpus_sha": "f" * 64}, "WRONG_CORPUS_NOT_VERA_R10"),
            ({"step_count": 20}, "UNAPPROVED_OPTIMIZER_STEP_COUNT")]:
            with self.subTest(change=change):
                result = classify(**facts(**change))
                self.assertIn(reason, result["reasons"])
                self.assertEqual(result["status"], "HOLD_NO_WEIGHT_CHANGE")

    def test_host_commit_exhaustion_prevents_weight_changes(self):
        result = classify(**facts(free_virtual_gib=0.18,
                                  free_ram_gib=6.51))
        self.assertEqual(result["status"], "HOLD_NO_WEIGHT_CHANGE")
        self.assertIn("LOW_WINDOWS_VIRTUAL_MEMORY_COMMIT", result["reasons"])
        self.assertIn("LOW_HOST_RAM", result["reasons"])

    def test_competing_model_or_active_output_refused(self):
        for change, reason in [
            ({"live_trainer_count": 1}, "EXISTING_TRAINER_DO_NOT_COMPETE"),
            ({"output_already_exists": True}, "OUTPUT_EXISTS_NO_OVERWRITE"),
            ({"free_gpu_mib": 3000}, "LOW_GPU_FREE_MEMORY")]:
            with self.subTest(change=change):
                result = classify(**facts(**change))
                self.assertIn(reason, result["reasons"])

    def test_real_source_and_parent_spec_bound(self):
        spec = ROOT / "successor/experiments/V10R3R10_VERA_VOICE_EXECUTION_SPEC_20261013_V1.json"
        corpus = Path(r"D:\VERA\.scratch\vera-v10r3r10-qwen35-identity-20261013-v1\train.jsonl")
        if not corpus.is_file():
            self.skipTest("R10 scratch corpus absent")
        data, parent = validate_subject(ROOT, spec, corpus)
        self.assertEqual(data["source_subject"]["train_sha256"],
                         EXPECTED_CORPUS_SHA)
        self.assertEqual(data["development_window"]["start_row"], 224)
        self.assertEqual(parent["adapter_model_sha256"], EXPECTED_R9_SHA)
        self.assertEqual(parent["source_cumulative_optimizer_steps"], 28)


if __name__ == "__main__":
    unittest.main()
