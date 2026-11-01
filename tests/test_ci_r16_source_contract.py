"""Static guard for the distinct, source-only TrainingDay R16/R18 CI lane."""

from pathlib import Path
import re
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"
SOURCE_TESTS = (
    "tests/test_ci_r16_source_contract.py",
    "tests/test_trainingday_reload_verify.py",
    "tests/test_trainingday_r16_checkpoint.py",
    "tests/test_r141_neural_continuation_gate.py",
    "tests/test_r18_checkpoint_commit.py",
    "tests/test_r18_eval.py",
    "tests/test_r18_mezo.py",
    "tests/test_r18_qlora.py",
    "tests/test_r18_eval_source.py",
    "tests/test_r152_physical_eval_guards.py",
    "tests/test_r18_benchmark.py",
)


def _job(name: str) -> str:
    source = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z_][\w-]*:\s*$|\Z)",
        source,
    )
    if match is None:
        raise AssertionError(f"Missing CI job: {name}")
    return match.group("body")


class R16SourceCIContractTests(unittest.TestCase):
    def test_r16_job_is_independent_exact_head_pinned_source_gate(self):
        body = _job("trainingday-r16-source")
        self.assertIn("github.event.repository.private == false", body)
        self.assertIn("runs-on: ubuntu-24.04", body)
        self.assertIn("python-version: \"3.12\"", body)
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", body)
        self.assertIn("github.event.pull_request.head.sha || github.sha", body)
        self.assertIn("persist-credentials: false", body)
        self.assertIn("git rev-parse --verify HEAD", body)
        self.assertIn("git rev-parse HEAD^{tree}", body)
        self.assertIn('"$actual_sha" != "$EXPECTED_SOURCE_SHA"', body)
        self.assertIn("GITHUB_STEP_SUMMARY", body)
        self.assertNotIn("continue-on-error", body)

    def test_r16_job_has_explicit_pytest_only_source_manifest(self):
        body = _job("trainingday-r16-source")
        self.assertEqual(
            re.findall(r"(?:^|\s)(tests/[\w./-]+\.py)(?=\s|$)", body),
            list(SOURCE_TESTS),
        )
        self.assertEqual(
            re.findall(r"(?:pip|pip3) install [^\n]+", body),
            ["pip install --disable-pip-version-check pytest==8.3.5"],
        )
        self.assertIn("python -m pytest -q", body)
        for forbidden in (
            "requirements.txt", "torch", "--preflight-only", "train_neural --",
            "unittest discover", "tests/test_*.py", "--ignore", "--deselect",
        ):
            self.assertNotIn(forbidden, body)

    def test_other_diagnostics_and_hard_hold_remain(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python -m pytest -q", _job("pytest"))
        self.assertIn("python -m unittest -q", _job("portable-source-contracts"))
        self.assertIn("tests/test_trainingday_neural.py", _job("trainingday-r14-source"))
        self.assertIn("windows-latest", _job("windows-stage0-source"))
        self.assertIn("tests.test_r144_physical_launch_hard_hold", workflow)
        self.assertNotIn("continue-on-error", workflow)

    def test_new_status_downgrade_regressions_are_in_r16_manifest(self):
        self.assertIn(
            "tests/test_trainingday_reload_verify.py",
            _job("trainingday-r16-source"),
        )
        tests = (WORKFLOW.parents[2] / "tests/test_trainingday_reload_verify.py").read_text(
            encoding="utf-8"
        )
        for name in (
            "test_r16_legacy_status_downgrade_never_bypasses_checkpoint_custody",
            "test_r16_downgrade_cannot_strip_fields_while_sidecar_remains",
            "test_r16_checkpoint_directory_itself_blocks_legacy_downgrade",
            "test_r16_recipe_marker_without_sidecars_cannot_use_legacy_status",
        ):
            self.assertRegex(tests, rf"(?m)^def {name}\(")


if __name__ == "__main__":
    unittest.main()
