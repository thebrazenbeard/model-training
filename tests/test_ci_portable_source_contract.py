"""Guard the distinct, source-only Linux CI signal against accidental dilution."""

from pathlib import Path
import re
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"


def _job(name: str) -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z_][\w-]*:\s*$|\Z)",
        workflow,
    )
    if match is None:
        raise AssertionError(f"Missing CI job: {name}")
    return match.group("body")


class PortableSourceCIContractTests(unittest.TestCase):
    def test_r155_draft_base_has_exact_pull_request_trigger(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        match = re.search(
            r'(?m)^  pull_request:\n    branches: \[(?P<branches>[^\]]+)\]$',
            workflow,
        )
        self.assertIsNotNone(match)
        self.assertIn('"draft/r154-cognitive-source-intake-20261101"',
                      match.group("branches"))
        self.assertIn('"draft/r155-independent-rights-pair-source-20261101"',
                      match.group("branches"))
        self.assertNotIn('"draft/**"', match.group("branches"))

    def test_portable_source_job_is_exact_head_and_pinned_linux(self):
        body = _job("portable-source-contracts")
        self.assertIn("runs-on: ubuntu-24.04", body)
        self.assertIn("github.event.repository.private == false", body)
        self.assertIn("github.event.pull_request.head.sha || github.sha", body)
        self.assertIn("persist-credentials: false", body)
        self.assertIn("git rev-parse --verify HEAD", body)
        self.assertIn("git rev-parse HEAD^{tree}", body)
        self.assertIn("GITHUB_STEP_SUMMARY", body)
        self.assertNotIn("continue-on-error", body)

    def test_portable_manifest_is_explicit_and_source_only(self):
        body = _job("portable-source-contracts")
        self.assertIn("python -m unittest -q", body)
        self.assertIn("tests.test_r143_wire_json_strict", body)
        self.assertIn("tests.test_r144_physical_launch_hard_hold", body)
        self.assertIn("tests.test_r18_checkpoint_commit", body)
        self.assertIn("tests.test_owner_historical_marker_contract", body)
        self.assertIn("tests.test_ci_r16_source_contract", body)
        self.assertIn("tests.test_cognitive_sft_intake", body)
        self.assertIn("tests.test_rights_pair_evidence_r155", body)
        self.assertIn("tests.test_rights_registry_evidence_r164", body)
        self.assertIn("tests.test_owner_historical_evidence_preflight", body)
        self.assertIn("tests.test_trainingday_readme_contract", body)
        self.assertIn("tests.test_trainingday_host_launch_request", body)
        self.assertIn("tests.test_trainingday_r16_checkpoint", body)
        self.assertNotIn("unittest discover", body)
        self.assertNotIn("tests/test_*.py", body)
        self.assertNotIn("pytest -q", body)

    def test_cognitive_sft_intake_has_live_portable_unittest_bridge(self):
        body = _job("portable-source-contracts")
        self.assertEqual(body.count("tests.test_cognitive_sft_intake"), 1)
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "tests.test_cognitive_sft_intake"
        )
        self.assertGreaterEqual(suite.countTestCases(), 15)

    def test_r155_rights_evidence_has_live_portable_unittest_bridge(self):
        body = _job("portable-source-contracts")
        self.assertEqual(body.count("tests.test_rights_pair_evidence_r155"), 1)
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "tests.test_rights_pair_evidence_r155"
        )
        # Count methods, not their outcome: if optional Ed25519 is absent,
        # nine crypto tests SKIP. This is *not* independent admission proof.
        self.assertGreaterEqual(suite.countTestCases(), 10)

    def test_r164_rights_registry_has_live_portable_unittest_bridge(self):
        body = _job("portable-source-contracts")
        self.assertEqual(body.count("tests.test_rights_registry_evidence_r164"), 1)
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "tests.test_rights_registry_evidence_r164"
        )
        self.assertGreaterEqual(suite.countTestCases(), 9)

    def test_full_pytest_remains_independent_red_diagnostic(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        body = _job("pytest")
        self.assertIn("runs-on: ubuntu-24.04", body)
        self.assertIn("python -m pytest -q", body)
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("fail-fast: false", workflow)

    def test_active_trainingday_r14_has_its_own_exact_head_source_gate(self):
        body = _job("trainingday-r14-source")
        self.assertIn("runs-on: ubuntu-24.04", body)
        self.assertIn("github.event.pull_request.head.sha || github.sha", body)
        self.assertIn("persist-credentials: false", body)
        self.assertIn("git rev-parse --verify HEAD", body)
        self.assertIn("git rev-parse HEAD^{tree}", body)
        self.assertIn("GITHUB_STEP_SUMMARY", body)
        self.assertIn("pytest==8.3.5", body)
        self.assertIn("tests/test_trainingday_admission.py", body)
        self.assertIn("tests/test_trainingday_labyrinth_intake.py", body)
        self.assertIn("tests/test_trainingday_neural.py", body)
        self.assertNotIn("continue-on-error", body)
        self.assertNotIn("--preflight-only", body)
        self.assertNotIn("train_neural --", body)


if __name__ == "__main__":
    unittest.main()
