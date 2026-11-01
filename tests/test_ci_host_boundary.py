"""Portable-only manifest integrity and negative tests; no physical fixtures."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import ci_host_boundary as boundary
import yaml


ROOT = Path(__file__).resolve().parents[1]


class HostBoundaryTests(unittest.TestCase):
    def test_manifest_contains_exactly_eleven_real_host_tests(self):
        self.assertEqual(len(boundary.HOST_TESTS), 11)

    def test_all_manifest_methods_remain_in_existing_source(self):
        for nodeid in boundary.HOST_TESTS:
            with self.subTest(nodeid=nodeid):
                file, class_name, method_name = nodeid.split("::")
                source = ROOT / file
                tree = ast.parse(source.read_text(encoding="utf8"))
                methods = {
                    node.name for cls in tree.body
                    if isinstance(cls, ast.ClassDef) and cls.name == class_name
                    for node in cls.body if isinstance(node, ast.FunctionDef)
                }
                self.assertIn(method_name, methods)

    def test_split_deferred_is_exact_and_never_marks_host_passed(self):
        inventory = [SimpleNamespace(nodeid=nodeid) for nodeid in boundary.HOST_TESTS]
        portable = SimpleNamespace(nodeid="tests/test_portable.py::test_true")
        retained, deferred = boundary.split_host_tests(inventory + [portable])
        self.assertEqual(retained, [portable])
        self.assertEqual(set(deferred), boundary.HOST_TESTS)

    def test_missing_or_duplicate_manifest_entry_fails_closed(self):
        inventory = [SimpleNamespace(nodeid=nodeid) for nodeid in boundary.HOST_TESTS]
        with self.assertRaisesRegex(ValueError, "missing from collection"):
            boundary.split_host_tests(inventory[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate test IDs"):
            boundary.split_host_tests(inventory + [inventory[0]])

    def test_portable_accounting_rejects_any_additional_exclusion(self):
        deferred = tuple(sorted(boundary.HOST_TESTS))
        allowed = {"passed": [1, 2, 3], "deselected": [
            SimpleNamespace(nodeid=n) for n in deferred
        ]}
        self.assertTrue(boundary.portable_accounting(3, allowed, deferred, minimum_collected=3)[0])
        for bad in ("failed", "error", "skipped", "xfailed", "xpassed"):
            with self.subTest(outcome=bad):
                self.assertFalse(boundary.portable_accounting(
                    3, {**allowed, bad: [object()]}, deferred, minimum_collected=3)[0])
        self.assertFalse(boundary.portable_accounting(3,
            {**allowed, "deselected": allowed["deselected"] +
             [SimpleNamespace(nodeid="tests/other.py::test_hidden")]}, deferred,
            minimum_collected=3)[0])
        self.assertFalse(boundary.portable_accounting(0, {}, deferred, minimum_collected=3)[0])
        self.assertFalse(boundary.portable_accounting(3, {**allowed, "passed": [1, 2]}, deferred,
            minimum_collected=3)[0])
        self.assertFalse(boundary.portable_accounting(3, allowed, None, minimum_collected=3)[0])
        self.assertFalse(boundary.portable_accounting(3, allowed, deferred)[0])

    def test_source_receipt_observes_git_head_and_tree(self):
        value = boundary.observed_local_source()
        for key in ("git_head", "git_tree"):
            self.assertRegex(value[key], r"^[0-9a-f]{40}$")

    def test_portable_invocation_rejects_any_collection_narrowing(self):
        required = ("-q", "-p", "ci_host_boundary", "--portable-source-only")
        self.assertTrue(boundary.valid_portable_invocation(required))
        for extra in ("tests/subdir", "tests/test_small.py", "--ignore-glob=tests/test_*",
                      "--deselect=tests/test_one.py", "-k", "--maxfail=1", "-o",
                      "python_files=test_selected.py", "--override-ini=python_files=test_one.py",
                      "--lf", "--pyargs", "-n", "2", "--confcutdir=tests",
                      "--rootdir=tests", "--collect-only", "-p", "no:ci_host_boundary"):
            with self.subTest(extra=extra):
                self.assertFalse(boundary.valid_portable_invocation(required + (extra,)))
        self.assertFalse(boundary.valid_portable_invocation(required[:-1]))
        self.assertFalse(boundary.valid_portable_invocation(("--portable-source-only", "-q")))

    def test_workflow_keeps_separate_required_red_host_status(self):
        workflow = ROOT / ".github/workflows/portable-source-ci.yml"
        data = yaml.load(workflow.read_text(encoding="utf8"), Loader=yaml.BaseLoader)
        self.assertEqual(data["on"]["pull_request"], {})  # Includes PRs targeting main.
        jobs = data["jobs"]
        self.assertEqual(set(jobs), {"portable-source-only", "host-integration-required"})
        portable_cmds = "\n".join(step.get("run", "") for step in jobs["portable-source-only"]["steps"])
        self.assertIn("--portable-source-only", portable_cmds)
        self.assertIn("git rev-parse HEAD", portable_cmds)
        host = jobs["host-integration-required"]
        self.assertNotIn("continue-on-error", host)
        self.assertNotIn("if", host)
        self.assertIn("exit 1", "\n".join(step.get("run", "") for step in host["steps"]))


if __name__ == "__main__":
    unittest.main()
