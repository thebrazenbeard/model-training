"""CPU-only tests for the source-level R15 process watchdog."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainingday.r38_budget_guard import BudgetHold, assess_budget, guarded_child


class BudgetGuardTests(unittest.TestCase):
    def test_invalid_or_missing_sensor_fails_closed(self):
        with self.assertRaises(BudgetHold):
            assess_budget(elapsed_s=1, timeout_s=10, gpu_temp_c=None, max_temp_c=82)
        with self.assertRaises(BudgetHold):
            assess_budget(elapsed_s=1, timeout_s=10, gpu_temp_c=float("nan"), max_temp_c=82)
        self.assertEqual(
            assess_budget(elapsed_s=1, timeout_s=10, gpu_temp_c=81, max_temp_c=82),
            "PASS",
        )

    def test_exact_deadline_and_thermal_threshold_abort(self):
        self.assertEqual(assess_budget(10, 10, 60, 82), "WALLCLOCK_EXCEEDED")
        self.assertEqual(assess_budget(0, 10, 82, 82), "THERMAL_LIMIT_REACHED")

    def test_unavailable_temperature_prevents_child_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "run-one"
            outcome = guarded_child(
                [sys.executable, "-c", "print('never')"], receipt,
                timeout_s=1.0, max_temp_c=80, poll_s=0.02,
                temperature=lambda: None,
            )
            self.assertEqual(outcome["status"], "HOLD_NO_LAUNCH")
            self.assertFalse((receipt / "stdout.log").exists())
            self.assertFalse((receipt / "launch.json").exists())

    def test_owned_child_timeout_preserves_incident_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "run-two"
            outcome = guarded_child(
                [sys.executable, "-c", "import time; print('go',flush=True); time.sleep(5)"],
                receipt, timeout_s=0.06, max_temp_c=82, poll_s=0.01,
                temperature=lambda: 55,
            )
            self.assertEqual(outcome["status"], "INCOMPLETE_ABORTED")
            self.assertEqual(outcome["abort_reason"], "WALLCLOCK_EXCEEDED")
            self.assertTrue((receipt / "launch.json").is_file())
            self.assertEqual(json.loads((receipt / "result.json").read_text())["status"], "INCOMPLETE_ABORTED")
            self.assertIn("go", (receipt / "stdout.log").read_text())

    def test_thermal_abort_and_no_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "run-three"
            temperatures = iter([60, 83, 83])
            outcome = guarded_child(
                [sys.executable, "-c", "import time; time.sleep(2)"], receipt,
                timeout_s=3, max_temp_c=82, poll_s=0.01,
                temperature=lambda: next(temperatures, 83),
            )
            self.assertEqual(outcome["abort_reason"], "THERMAL_LIMIT_REACHED")
            with self.assertRaises(BudgetHold):
                guarded_child(
                    [sys.executable, "-c", "pass"], receipt,
                    timeout_s=3, max_temp_c=82, poll_s=0.01,
                    temperature=lambda: 60,
                )

    def test_success_is_process_exit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "run-four"
            outcome = guarded_child(
                [sys.executable, "-c", "print('done')"], receipt,
                timeout_s=3, max_temp_c=82, poll_s=0.01,
                temperature=lambda: 42,
            )
            self.assertEqual(outcome["status"], "INCOMPLETE_PROCESS_TREE_UNVERIFIED")
            self.assertEqual(outcome["child_returncode"], 0)
            self.assertFalse(outcome["guard_enforcement_qualified"])
            self.assertEqual(outcome["process_tree_containment"], "UNVERIFIED_DIRECT_CHILD_ONLY")

    def test_child_exit_after_deadline_between_samples_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "run-late"
            outcome = guarded_child(
                [sys.executable, "-c", "import time; time.sleep(0.07)"], receipt,
                timeout_s=0.04, max_temp_c=82, poll_s=0.2,
                temperature=lambda: 50,
            )
            self.assertEqual(outcome["status"], "INCOMPLETE_ABORTED")
            self.assertEqual(outcome["abort_reason"], "WALLCLOCK_EXCEEDED")
            self.assertFalse(outcome["guard_enforcement_qualified"])

    def test_nested_receipt_symlink_ancestor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            (real / "sub").mkdir(parents=True)
            alias = root / "link"
            try:
                os.symlink(real, alias, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation unavailable on this host")
            receipt = alias / "sub" / "run"
            with self.assertRaisesRegex(BudgetHold, "symlink"):
                guarded_child(
                    [sys.executable, "-c", "print('never')"], receipt,
                    timeout_s=3, max_temp_c=82, poll_s=0.01,
                    temperature=lambda: 50,
                )
            self.assertFalse((real / "sub" / "run").exists())

    def test_failed_spawn_preserves_attempt_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "run-five"
            with self.assertRaises(FileNotFoundError):
                guarded_child(
                    [str(Path(tmp) / "missing-executable")], receipt,
                    timeout_s=2, max_temp_c=82, poll_s=0.01,
                    temperature=lambda: 55,
                )
            result = json.loads((receipt / "result.json").read_text())
            self.assertEqual(result["status"], "INCOMPLETE_NO_CHILD")
            self.assertTrue(result["attempt_consumed"])

    def test_cli_cannot_monitor_a_different_gpu_than_r15(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "unlaunched"
            result = subprocess.run([
                sys.executable, "-m", "trainingday.r38_budget_guard",
                "--receipt-dir", str(receipt), "--max-seconds", "2",
                "--max-gpu-temp-c", "82", "--gpu-index", "1", "--",
                "--mode", "correct",
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("device 0", result.stderr)
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
