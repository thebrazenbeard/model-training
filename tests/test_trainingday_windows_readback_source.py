"""Portable source guard for the fail-closed Windows host readback helper.

These static assertions are NOT a Windows runtime identity/ACL qualification.
"""

from pathlib import Path
import re
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] / "trainingday" /
          "windows_host_broker" / "ReadOnlyHostReadback.ps1")


class WindowsHostReadbackSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_only_expected_read_only_host_surfaces(self):
        source = self.source
        for required in (
            "Get-CimInstance -ClassName Win32_Service",
            "Get-CimInstance -ClassName Win32_Process",
            "Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid",
            "Get-FileHash -LiteralPath $path -Algorithm SHA256",
            "Get-Acl -LiteralPath $path",
            "Get-ExecutablePath ([string]$svc.PathName)",
            "Get-DirectoryAclEvidence $expectedRoot",
            "$expectedRoot = 'D:\\VERA'",
            "$serviceName = 'VeraTrainingDayHostBroker'",
            "ConvertTo-Json -Depth 6",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_no_host_mutation_network_or_credential_handling(self):
        # A source-only tripwire; review the actual diff and run on Windows
        # before trusting this helper's real process/ACL observations.
        commands = (
            "Start-Service", "Stop-Service", "Restart-Service", "Set-Service",
            "New-Service", "sc.exe", "Set-Acl", "Set-Item", "New-Item",
            "Remove-Item", "Move-Item", "Copy-Item", "Out-File", "Tee-Object",
            "Set-Content", "Add-Content", "Export-Clixml", "Set-ExecutionPolicy",
            "Invoke-WebRequest", "Invoke-RestMethod", "Invoke-Command",
            "Start-Process", "Enter-PSSession", "curl.exe", "ssh.exe",
            "-SkipCertificateCheck", "-Insecure", "-k ",
        )
        for forbidden in commands:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)
        self.assertNotIn("C:\\VERA", self.source)
        self.assertNotRegex(self.source, r"(?i)convertto-securestring|pscredential")

    def test_unknown_host_evidence_always_remains_hold(self):
        source = self.source
        for name in (
            "rights_admitted", "evaluator_custody_authenticated",
            "gpu_lease_granted", "physical_training_authorized",
        ):
            with self.subTest(name=name):
                self.assertRegex(source, rf"(?m)^\s*{re.escape(name)} = \$false$")
        self.assertIn("status = 'HOLD'", source)
        self.assertIn("process_owner_sid_evidence_available = $false", source)
        self.assertIn("process_service_pid_stable = $false", source)
        self.assertIn("$serviceAfter = Get-CimInstance -ClassName Win32_Service", source)
        self.assertIn("$process.CreationDate -eq $processAfter.CreationDate", source)
        self.assertNotIn("process_token_evidence_available", source)
        self.assertIn("SERVICE_CIM_UNAVAILABLE", source)
        self.assertIn("BINARY_READ_UNAVAILABLE", source)
        self.assertNotIn("Get-ChildItem -Recurse", source)


if __name__ == "__main__":
    unittest.main()
