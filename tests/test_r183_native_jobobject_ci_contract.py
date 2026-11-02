"""Guard the native Windows custody source test from CI omission."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

class NativeJobObjectCITests(unittest.TestCase):
    def test_real_windows_test_is_executed_in_native_job(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  windows-stage0-source:")
        end = workflow.index("  pytest-synthetic-merge:", start)
        job = workflow[start:end]
        self.assertIn("runs-on: windows-latest", job)
        self.assertIn("JobObjectChildCustody.cs", job)
        self.assertIn("native_jobobject_child_custody.cs", job)
        self.assertIn("NATIVE_JOB_CUSTODY_PASS", job)
        self.assertIn("NativeCustodyTest.exe", job)
        self.assertIn("HostOwnedProbeReservation.cs", job)
        self.assertIn("native_broker_probe_reservation.cs", job)
        self.assertIn("NATIVE_PROBE_RESERVATION_PASS", job)
        self.assertNotIn("continue-on-error", job)
        self.assertNotIn("|| true", job)

if __name__ == "__main__":
    unittest.main()
