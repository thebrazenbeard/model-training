"""V10 cannot use local preflight as a physical Qwen launch grant.

These tests use no model, CUDA, installed host, credentials, or training data.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from successor.experiments import train_v10_qwen35_authorized as runner


READY = {
    "status": "READY_PRECONDITIONS",
    "training_allowed": True,
    "reasons": [],
}


class V10PhysicalLaunchHoldTests(unittest.TestCase):
    def test_execute_holds_before_preflight_artifacts_model_or_output(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            output = root / "output-must-not-exist"
            called = []

            def model_loader():
                called.append("model")
                raise AssertionError("model import and CUDA must not occur")

            with patch.object(runner, "assess_preflight", side_effect=AssertionError(
                "preflight must not be read before physical authority"
            )), patch.object(runner, "_read_json", side_effect=AssertionError(
                "no authority or corpus file may be read"
            )), patch.dict(os.environ, {
                "TRAININGDAY_AUTHORIZED": "true",
                "VERA_GPU_LEASE": "self-declared",
            }):
                with self.assertRaisesRegex(runner.TrainingHold, "HARD HOLD"):
                    runner.execute_authorized_training(
                        root,
                        train_jsonl=root / "nonexistent-train.jsonl",
                        validation_jsonl=root / "nonexistent-dev.jsonl",
                        output_dir=output,
                        training_stack_loader=model_loader,
                    )

            self.assertEqual(called, [])
            self.assertFalse(output.exists())

    def test_spoofed_ready_plan_is_physical_hold_without_model_import(self):
        with patch.object(runner, "assess_preflight", return_value=dict(READY)):
            with patch.object(runner, "load_training_stack", side_effect=AssertionError(
                "plan must not import model stack"
            )):
                result = runner.plan_execution(
                    Path("/unreadable-unused-source"),
                    training_stack_loader=lambda: self.fail("model loader invoked"),
                )

        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["preflight"], READY)
        self.assertFalse(result["training_stack_loaded"])
        self.assertTrue(any("HARD HOLD" in r for r in result["reasons"]))

    def test_existing_preflight_hold_remains_a_hold(self):
        held = {"status": "HOLD", "training_allowed": False,
                "reasons": ["unadmitted-private-dev"]}
        with patch.object(runner, "assess_preflight", return_value=held):
            result = runner.plan_execution(
                Path("/no-model"),
                training_stack_loader=lambda: self.fail("model loader invoked"),
            )
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["reasons"], ["unadmitted-private-dev"])
        self.assertFalse(result["training_stack_loaded"])


if __name__ == "__main__":
    unittest.main()
