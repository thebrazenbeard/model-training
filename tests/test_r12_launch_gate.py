"""R12 rejects contested GPU and protected output rather than evicting services."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from successor.experiments import preflight_r12_launch
from successor.experiments.preflight_r12_launch import classify, validated_receipt


class R12LaunchGateTests(unittest.TestCase):
    def test_busy_gpu_never_admits_training(self):
        result = classify(ram=12, virtual=60, gpu=1608, active=0, output_exists=False)
        self.assertEqual(result["status"], "HOLD_NO_OPTIMIZER")
        self.assertIn("GPU_OCCUPIED_PRESERVE_SERVING_PROCESS", result["reasons"])
        self.assertFalse(result["services_interrupted"])
        self.assertEqual(result["optimizer_steps_executed"], 0)

    def test_existing_output_and_other_trainer_fail_closed(self):
        result = classify(ram=12, virtual=60, gpu=3962, active=1, output_exists=True)
        self.assertEqual(result["status"], "HOLD_NO_OPTIMIZER")
        self.assertIn("ANOTHER_TRAINER_ACTIVE", result["reasons"])
        self.assertIn("OUTPUT_EXISTS_PRESERVE_ADAPTER", result["reasons"])

    def test_retry_receipt_is_distinct_and_scoped(self):
        # The production default is a Windows D: path. Use a real in-scope
        # receipt so the duplicate-path assertion works on POSIX and Windows.
        with TemporaryDirectory() as temp_dir:
            scope = Path(temp_dir)
            receipt = scope / "VERA_R12_FP32_LAUNCH_GATE_20261015_V1.json"
            receipt.write_text("{}", encoding="utf-8")
            later = scope / "VERA_R12_FP32_LAUNCH_GATE_20261015_V999999.json"
            with patch.object(preflight_r12_launch, "RECEIPT", receipt):
                self.assertEqual(validated_receipt(later), later.resolve())
                with self.assertRaisesRegex(RuntimeError, "exists"):
                    validated_receipt(receipt)
                with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                    validated_receipt(scope / ".." / later.name)
                with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                    validated_receipt(scope / "VERA_R12_FP32_LAUNCH_GATE_20261015_V0.json")

    def test_resource_thresholds_and_ready_path(self):
        held = classify(ram=8.99, virtual=11.99, gpu=3899, active=0, output_exists=False)
        self.assertEqual(len(held["reasons"]), 3)
        ready = classify(ram=10, virtual=60, gpu=3962, active=0, output_exists=False)
        self.assertEqual(ready["status"], "READY_LOCAL_R12")
        self.assertEqual(ready["reasons"], [])
        self.assertEqual(ready["optimizer_steps_executed"], 0)
