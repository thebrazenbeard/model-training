"""R11 retry-after-pagefile exact source-binding regression tests."""
from __future__ import annotations
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"successor/experiments/V10R3R11_VERA_VOICE_EXECUTION_SPEC_20261014_V1.json"
RETRY=ROOT/"successor/experiments/V10R3R11_VERA_VOICE_EXECUTION_SPEC_20261015_PAGEFILE_RETRY1.json"
OLD=Path(r"D:\VERA\models\adapters\v10r3r11-vera-voice-qwen35-r10-resume-20261014")
NEW=Path(r"D:\VERA\models\adapters\v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1")


class RetrySpecTests(unittest.TestCase):
    def test_exactly_output_namespace_changes(self):
        before=json.loads(ORIGINAL.read_text(encoding="utf8"))
        after=json.loads(RETRY.read_text(encoding="utf8"))
        self.assertEqual(before["output"]["namespace"],str(OLD))
        self.assertEqual(after["output"]["namespace"],str(NEW))
        patched=json.loads(json.dumps(before))
        patched["output"]["namespace"]=str(NEW)
        self.assertEqual(after,patched)

    def test_parent_and_training_constraints_are_unchanged(self):
        spec=json.loads(RETRY.read_text(encoding="utf8"))
        self.assertEqual(spec["resume_adapter"]["source_cumulative_optimizer_steps"],36)
        self.assertEqual(spec["trainer"]["max_optimizer_steps"],8)
        self.assertEqual(spec["trainer"]["learning_rate"],2e-6)
        self.assertEqual(spec["resume_adapter"]["adapter_model_sha256"],
                         "c69ed0e4bd78e09752b5cbb4dfaf19ebea8a4174735bf88d2676947ef04b2739")
        self.assertEqual(spec["source_subject"]["train_sha256"],
                         "f73fe5a0d2c21bdd3ab3cc2cee3238d3297d0941611d31514c543ab560ee491e")
        self.assertFalse(spec["authority"]["paid_compute_authorized"])
        self.assertFalse(spec["authority"]["merge_authorized"])
        self.assertFalse(spec["authority"]["deploy_activate_authorized"])

    def test_old_attempt_and_r10_parent_not_retired(self):
        parent=Path(r"D:\VERA\models\adapters\v10r3r10-vera-voice-qwen35-r9-resume-20261013\adapter\adapter_model.safetensors")
        self.assertTrue(parent.is_file())
        self.assertTrue(OLD.is_dir())
        self.assertNotEqual(OLD,NEW)


if __name__=="__main__":
    unittest.main()
