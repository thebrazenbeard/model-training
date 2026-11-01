"""Portable-only manifest integrity and negative tests; no physical fixtures."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ci_host_boundary as boundary
import yaml
from tests.test_owner_historical_marker_contract import EXPECTED as EXPECTED_OWNER_MARKERS


ROOT = Path(__file__).resolve().parents[1]


class HostBoundaryTests(unittest.TestCase):
    def test_manifest_contains_exactly_twelve_real_host_and_two_windows_tests(self):
        self.assertEqual(len(boundary.HOST_TESTS), 12)
        self.assertEqual(len(boundary.WINDOWS_PLATFORM_TESTS), 2)
        self.assertIn(
            "tests/test_v10r3r11_vera_voice.py::R11SourceTests::test_generated_spec_binds_real_completed_parent",
            boundary.HOST_TESTS,
        )
        self.assertEqual(boundary.HOST_TESTS & boundary.WINDOWS_PLATFORM_TESTS, set())
        marked = {
            f"tests/{filename}::{class_name}::{method}"
            for filename, methods in EXPECTED_OWNER_MARKERS.items()
            for class_name in ({
                "test_v10r3r10_vera_voice.py": "R10VoiceCurriculumTests",
                "test_v10r3r11_vera_voice.py": "R11SourceTests",
                "test_v10r3r11_pagefile_retry1.py": "RetrySpecTests",
                "test_r12_fp32_precision.py": "R12PrecisionTests",
                "test_r12_source_portability.py": "SourceTransportTests",
                "test_r13_corpus_contract.py": "R13CorpusContractTests",
            }[filename],)
            for method in methods
        }
        self.assertLessEqual(marked, boundary.HOST_TESTS)

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

        for nodeid in boundary.WINDOWS_PLATFORM_TESTS:
            with self.subTest(nodeid=nodeid):
                file, function_name = nodeid.split("::")
                tree = ast.parse((ROOT / file).read_text(encoding="utf8"))
                functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                self.assertIn(function_name, functions)

    def test_split_deferred_is_exact_and_never_marks_host_passed(self):
        inventory = [SimpleNamespace(nodeid=nodeid) for nodeid in
                     boundary.HOST_TESTS | boundary.WINDOWS_PLATFORM_TESTS]
        portable = SimpleNamespace(nodeid="tests/test_portable.py::test_true")
        retained, deferred = boundary.split_host_tests(inventory + [portable])
        self.assertEqual(retained, [portable])
        self.assertEqual(set(deferred), boundary.HOST_TESTS | boundary.WINDOWS_PLATFORM_TESTS)

    def test_missing_or_duplicate_manifest_entry_fails_closed(self):
        inventory = [SimpleNamespace(nodeid=nodeid) for nodeid in
                     boundary.HOST_TESTS | boundary.WINDOWS_PLATFORM_TESTS]
        with self.assertRaisesRegex(ValueError, "missing from collection"):
            boundary.split_host_tests(inventory[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate test IDs"):
            boundary.split_host_tests(inventory + [inventory[0]])

    def test_portable_accounting_rejects_any_additional_exclusion(self):
        deferred = tuple(sorted(boundary.HOST_TESTS | boundary.WINDOWS_PLATFORM_TESTS))
        allowed = {"passed": [1, 2, 3], "deselected": [
            SimpleNamespace(nodeid=n) for n in deferred
        ]}
        clean, evidence = boundary.portable_accounting(3, allowed, deferred, minimum_collected=3)
        self.assertTrue(clean)
        self.assertEqual(evidence["host_deferred"], 12)
        self.assertEqual(evidence["windows_platform_deferred"], 2)
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
        self.assertFalse(boundary.portable_accounting(
            3, allowed, tuple(sorted(boundary.HOST_TESTS)), minimum_collected=3)[0])
        self.assertFalse(boundary.portable_accounting(3, allowed, deferred)[0])

    def test_unknown_historical_marker_fails_closed_before_execution(self):
        inventory = [SimpleNamespace(nodeid=nodeid) for nodeid in
                     boundary.HOST_TESTS | boundary.WINDOWS_PLATFORM_TESTS]
        unlisted = SimpleNamespace(
            nodeid="tests/test_future.py::test_unlisted_historical",
            get_closest_marker=lambda name: object() if name == "owner_historical" else None,
        )
        with self.assertRaisesRegex(ValueError, "unlisted owner_historical"):
            boundary.split_host_tests(inventory + [unlisted])

    def test_receipt_separates_host_and_windows_without_granting_training(self):
        lines = []
        reporter = SimpleNamespace(
            write_sep=lambda *args: None,
            write_line=lines.append,
        )
        config = SimpleNamespace(
            getoption=lambda name: name == "--portable-source-only",
            _vera_portable_deferred=tuple(sorted(boundary.DECLARED_DEFERRED_TESTS)),
        )
        with patch.object(boundary, "observed_local_source", return_value={
            "git_head": "a" * 40,
            "git_tree": "b" * 40,
        }):
            boundary.pytest_terminal_summary(reporter, 0, config)
        raw = next(line for line in lines if line.startswith("CI_HOST_BOUNDARY_RECEIPT_JSON="))
        receipt = json.loads(raw.split("=", 1)[1])
        self.assertEqual(set(receipt["host_tests_deferred"]), boundary.HOST_TESTS)
        self.assertEqual(set(receipt["windows_platform_tests_deferred"]),
                         boundary.WINDOWS_PLATFORM_TESTS)
        self.assertEqual(receipt["host_tests_executed"], 0)
        self.assertEqual(receipt["windows_platform_tests_executed"], 0)
        self.assertFalse(receipt["authorizes_training"])
        self.assertFalse(receipt["authorizes_model_qualification"])

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
        self.assertIn("12 host + 2 Windows tests", jobs["portable-source-only"]["name"])
        windows = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf8")
        self.assertIn("windows-stage0-source:", windows)
        self.assertIn("runs-on: windows-latest", windows)
        for nodeid in boundary.WINDOWS_PLATFORM_TESTS:
            self.assertIn(nodeid, windows)
        host = jobs["host-integration-required"]
        self.assertNotIn("continue-on-error", host)
        self.assertNotIn("if", host)
        self.assertIn("exit 1", "\n".join(step.get("run", "") for step in host["steps"]))


if __name__ == "__main__":
    unittest.main()
