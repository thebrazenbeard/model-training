"""Host-independent adversarial checks for the opt-in full-suite guard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import ci_full_suite_guard as guard
import yaml


def stats(**counts):
    return {key: [object()] * n for key, n in counts.items()}


class FakeReporter:
    def __init__(self, outcomes):
        self.stats = outcomes
        self.messages = []

    def write_sep(self, _, message, **kwargs):
        self.messages.append(message)


def session(collected, reporter):
    manager = SimpleNamespace(get_plugin=lambda _: reporter)
    return SimpleNamespace(testscollected=collected,
                           config=SimpleNamespace(pluginmanager=manager),
                           exitstatus=0)


class FullSuiteGuardTests(unittest.TestCase):
    def test_only_exact_nonempty_all_pass_is_clean(self):
        clean, detail = guard.audit_counts(3, stats(passed=3))
        self.assertTrue(clean, detail)

    def test_empty_and_miscounted_collections_are_rejected(self):
        for collected, reported in [(0, stats()), (3, stats(passed=2)),
                                    (1, stats(passed=2))]:
            with self.subTest(collected=collected, reported=reported):
                self.assertFalse(guard.audit_counts(collected, reported)[0])

    def test_each_prohibited_outcome_is_rejected_even_with_all_passed(self):
        for name in guard.PROHIBITED_OUTCOMES:
            with self.subTest(name=name):
                clean, detail = guard.audit_counts(2, stats(passed=2, **{name: 1}))
                self.assertFalse(clean)
                self.assertIn(name, detail)

    def test_hook_turns_silent_skip_into_failing_exit(self):
        reporter = FakeReporter(stats(passed=2, skipped=1))
        current = session(2, reporter)
        fake_pytest = SimpleNamespace(ExitCode=SimpleNamespace(OK=0, TESTS_FAILED=1))
        with mock.patch.dict(sys.modules, {"pytest": fake_pytest}):
            guard.pytest_sessionfinish(current, 0)
        self.assertEqual(current.exitstatus, 1)
        self.assertIn("skipped", reporter.messages[0])

    def test_hook_preserves_preexisting_nonzero_exit(self):
        reporter = FakeReporter(stats(passed=2))
        current = session(2, reporter)
        current.exitstatus = 4
        fake_pytest = SimpleNamespace(ExitCode=SimpleNamespace(OK=0, TESTS_FAILED=1))
        with mock.patch.dict(sys.modules, {"pytest": fake_pytest}):
            guard.pytest_sessionfinish(current, 4)
        self.assertEqual(current.exitstatus, 4)

    def test_hook_rejects_missing_reporter(self):
        current = session(2, None)
        fake_pytest = SimpleNamespace(ExitCode=SimpleNamespace(OK=0, TESTS_FAILED=1))
        with mock.patch.dict(sys.modules, {"pytest": fake_pytest}):
            guard.pytest_sessionfinish(current, 0)
        self.assertEqual(current.exitstatus, 1)


class CIWorkflowIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_path = Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"
        # BaseLoader preserves GitHub's literal "on" key; YAML 1.1 SafeLoader
        # would otherwise parse it as boolean True.
        cls.workflow = yaml.load(workflow_path.read_text(encoding="utf8"),
                                 Loader=yaml.BaseLoader)

    def test_trainingday_pull_request_and_push_are_covered(self):
        triggers = self.workflow["on"]
        self.assertIn("trainingday/**", triggers["pull_request"]["branches"])
        self.assertIn("trainingday/**", triggers["push"]["branches"])

    def test_both_revisions_are_attested_and_strictly_tested(self):
        jobs = self.workflow["jobs"]
        # R153's exact-head/synthetic-merge full diagnostic is additive to
        # R154's source-manifest, R14, R16 and native Windows lock lanes.
        self.assertEqual(set(jobs), {
            "pytest-source", "pytest-synthetic-merge",
            "portable-source-contracts", "trainingday-r14-source",
            "trainingday-r16-source", "windows-stage0-source",
            "rights-crypto-source",
        })
        crypto_job = jobs["rights-crypto-source"]
        self.assertEqual(crypto_job["runs-on"], "ubuntu-24.04")
        crypto_commands = "\n".join(str(step.get("run", "")) for step in crypto_job["steps"])
        self.assertIn("cryptography==46.0.3", crypto_commands)
        self.assertIn("result.skipped", crypto_commands)
        # Every rights evidence profile must execute in the pinned crypto job.
        self.assertIn("tests.test_owner_controlled_rights_v1", crypto_commands)
        self.assertIn("result.testsRun < 25", crypto_commands)
        for name, expected in (("pytest-source", "EXPECTED_SOURCE_SHA"),
                               ("pytest-synthetic-merge", "EXPECTED_MERGE_SHA")):
            with self.subTest(job=name):
                job = jobs[name]
                if name == "pytest-source":
                    self.assertNotIn("if", job)
                self.assertEqual(job["runs-on"], "ubuntu-24.04")
                steps = job["steps"]
                commands = "\n".join(str(step.get("run", "")) for step in steps)
                self.assertIn("-p ci_full_suite_guard", commands)
                self.assertIn("git rev-parse --verify HEAD", commands)
                self.assertIn("CI requires an explicitly qualified runner context", commands)
                self.assertTrue(any(expected in step.get("env", {}) for step in steps))
        self.assertIn("pull_request", jobs["pytest-synthetic-merge"]["if"])


if __name__ == "__main__":
    unittest.main()
