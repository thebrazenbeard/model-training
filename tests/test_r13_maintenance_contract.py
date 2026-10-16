"""Guard the reversible, service-scoped R13 maintenance controller."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "successor/experiments"
CONTROL = ROOT / "run_r13_with_preactive_restore_20261016.ps1"
RECOVER = ROOT / "restore_preactive_r13_20261016.ps1"


class MaintenanceContractTests(unittest.TestCase):
    def test_dry_run_never_writes_execution_receipt(self):
        code = CONTROL.read_text(encoding="utf8")
        self.assertIn("if(-not $PreflightOnly){\n $receiptData=[ordered]@{", code)
        self.assertIn("VERA_R13_TRAIN_20261016_CONTROLLER_V3.json", code)
        self.assertIn("VERA_R13_BEHAVIOR_LAUNCH_GATE_20261016_V3.json", code)
        self.assertNotIn("VERA_R13_TRAIN_20261016_CONTROLLER.json", code)

    def test_recovery_task_registered_before_any_preactive_stop(self):
        code = CONTROL.read_text(encoding="utf8")
        self.assertLess(code.index("Register-ScheduledTask -TaskName $taskName"),
                        code.index("Stop-ScheduledTask -TaskName $name"))
        self.assertLess(code.index("$watchRegistered=$true"),
                        code.index("$paused=$true"))
        self.assertIn("finally {", code)
        self.assertIn("RestorePreActive", code)
        self.assertIn("Unregister-ScheduledTask -TaskName $taskName", code)

    def test_only_named_preactive_tasks_and_exact_owned_trainers(self):
        control = CONTROL.read_text(encoding="utf8")
        recover = RECOVER.read_text(encoding="utf8")
        for text in (control, recover):
            self.assertIn("PreActive Window Supervisor", text)
            self.assertIn("PreActive Qwen Endpoint", text)
            self.assertIn("PreActive Daemon", text)
            self.assertIn("Contains($repo)", text)
            self.assertIn("Contains($spec)", text)
            self.assertNotIn("Disable-ScheduledTask", text)
            self.assertNotIn("Stop-Process -Name", text)
            self.assertNotIn("Stop-Service", text)
        self.assertIn("OUTSIDE_PREACTIVE_AUTHORIZED_WINDOW_NO_START", recover)

    def test_frozen_binding_and_final_runtime_checks(self):
        script = CONTROL.read_text(encoding="utf8")
        for value in (
            "ad2a3d56c9a5ee648764e9469676b78155bb993de690edda9627189a767b0fdf",
            "90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5",
            "a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c",
            "source_head",
            "cumulative_optimizer_steps -ne 84",
            "preactive_restored_verified",
        ):
            self.assertIn(value, script)
        self.assertIn("-PreflightOnly", script.replace("[switch]$PreflightOnly", "-PreflightOnly"))
        self.assertIn("$PreflightOnly", script)
        self.assertIn("original_gpu_free_mib", script.replace("initial_gpu_free_mib", "original_gpu_free_mib"))


if __name__ == "__main__":
    unittest.main()
