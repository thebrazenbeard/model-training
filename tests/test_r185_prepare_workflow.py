"""Guard the isolated Windows PREPARE_V2 source-only qualification lane.

Passing this contract is not proof of Windows host integration, peer identity
capture, NAS permission, protected evaluation custody, or GPU ownership.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/trainingday-r185-prepare-source.yml"


class PrepareV2WindowsWorkflowTests(unittest.TestCase):
    def workflow(self):
        return WORKFLOW.read_text(encoding="utf-8")

    def test_isolated_trigger_read_only_checkout_and_windows_runner(self):
        source = self.workflow()
        self.assertIn('name: TRAININGDAY R185 PREPARE_V2 source-only Windows smoke', source)
        self.assertIn('  pull_request: {}', source)
        self.assertIn('  workflow_dispatch:', source)
        self.assertIn('  contents: read', source)
        self.assertIn('runs-on: windows-latest', source)
        self.assertIn('actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1', source)
        self.assertIn('ref: ${{ github.event.pull_request.head.sha || github.sha }}', source)
        self.assertIn('persist-credentials: false', source)
        self.assertIn('git rev-parse --verify HEAD', source)
        self.assertIn('EXPECTED_SOURCE_SHA', source)

    def test_native_csharp_compiler_runs_source_and_checks_real_output(self):
        source = self.workflow()
        self.assertIn('Microsoft.NET\\Framework64\\v4.0.30319\\csc.exe', source)
        self.assertIn('PrepareV2PeerIdentity.cs', source)
        self.assertIn('/target:library', source)
        self.assertIn('/r:System.Security.dll', source)
        self.assertIn('/r:System.Core.dll', source)
        self.assertIn('if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $output))', source)
        self.assertIn('[System.Reflection.Assembly]::LoadFrom($output)', source)
        self.assertIn("GetRawConstantValue() -cne 'HOLD'", source)
        self.assertIn("GetProperty('AuthorizesTraining')", source)
        self.assertIn("GetMethod('DenyClaim'", source)
        self.assertIn("'HOLD_PROCESS_EXITED_OR_UNOBSERVABLE'", source)
        self.assertIn('R185_PREPARE_SOURCE_SMOKE_PASS', source)

    def test_source_regressions_run_and_failures_are_not_masked(self):
        source = self.workflow()
        self.assertIn('tests.test_r185_prepare_identity_source', source)
        self.assertIn('tests.test_r185_prepare_workflow', source)
        self.assertIn('python -m unittest -q', source)
        for forbidden in (
            'continue-on-error:',
            '|| true',
            'exit 0',
            'Start-Process',
            'Install-Service',
            'New-Service',
            'secrets.',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_claims_remain_source_only_not_host_runtime_qualification(self):
        source = self.workflow()
        self.assertIn('SOURCE_ONLY_NOT_HOST_QUALIFIED', source)
        self.assertIn('no service installation, credential grants, GPU use or training', source)
        self.assertNotIn('HOST_RUNTIME_QUALIFIED', source)
        self.assertNotIn('TRAINING_AUTHORIZED', source)


if __name__ == "__main__":
    unittest.main()
