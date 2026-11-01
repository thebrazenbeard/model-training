"""Explicit, audited boundary between portable and real-host integration tests.

Use only for a labeled *portable-source* lane::

    python -m pytest -q -p ci_host_boundary --portable-source-only

The ordinary full test command is unchanged and still collects every host
test. This plugin cannot qualify real host state, GPU leases, NAS routing,
training, or private held-out evaluation; it never converts a host absence
into a physical pass.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


# PR #179 hosted source run: 1454 passed + 18 failed + 2 skipped = 1474
# collected. The original 11-host-test baseline was 1463. Retain this
# conservative floor even after adding one host and two Windows-only entries;
# no static count proves complete inventory across later source revisions.
MIN_PORTABLE_COLLECTED = 1463


# Historical D:\\VERA and private Lappy artifacts are intentionally NOT
# manufactured in Linux CI. Every named assertion remains in its test file.
HOST_TESTS = frozenset({
    "tests/test_r12_fp32_precision.py::R12PrecisionTests::test_r12_source_is_new_and_no_heldout_prompt_overlap",
    "tests/test_r12_source_portability.py::SourceTransportTests::test_existing_r12_train_corpus_still_exact",
    "tests/test_r13_corpus_contract.py::R13CorpusContractTests::test_new_128_targeted_rows_balance_identity_and_judgment",
    "tests/test_r13_corpus_contract.py::R13CorpusContractTests::test_training_window_alternates_and_never_changes_outer_rows",
    "tests/test_r13_corpus_contract.py::R13CorpusContractTests::test_exact_heldout_prompt_excluded",
    "tests/test_r13_corpus_contract.py::R13CorpusContractTests::test_frozen_source_rebuilds_bit_identical_spec_and_corpus",
    "tests/test_v10r3r11_vera_voice.py::R11SourceTests::test_disk_spec_exactly_binds_outputs",
    "tests/test_v10r3r11_vera_voice.py::R11SourceTests::test_generated_spec_binds_real_completed_parent",
    "tests/test_v10r3r10_vera_voice.py::R10VoiceCurriculumTests::test_generated_spec_targets_qwen_successor_no_external_spend",
    "tests/test_v10r3r11_pagefile_retry1.py::RetrySpecTests::test_old_attempt_and_r10_parent_not_retired",
    "tests/test_v10r3r10_launch_gate.py::QwenOnlyLaunchTests::test_real_source_and_parent_spec_bound",
    "tests/test_r11_voice_heldout_probe_contract.py::R11HeldoutProbeContractTests::test_heldout_prompt_count_and_prohibition",
})

# These are not private owner-evidence tests. The genuine msvcrt process-lock
# behavior is exercised by Tests/windows-stage0-source on Windows; Linux cannot
# execute it, so the portable lane must report these exact platform deferrals
# separately, rather than accepting two silently skipped test outcomes.
WINDOWS_PLATFORM_TESTS = frozenset({
    "tests/test_stage0_runtime_qualification.py::test_stage0_gpu_lock_rejects_concurrent_holder_and_releases",
    "tests/test_stage0_runtime_qualification.py::test_stage0_gpu_lock_blocks_other_process_then_releases",
})
DECLARED_DEFERRED_TESTS = HOST_TESTS | WINDOWS_PLATFORM_TESTS


def valid_portable_invocation(argv: tuple[str, ...]) -> bool:
    """Allow only the CI-owned command, never a narrowed test inventory.

    The lane is not a general-purpose developer test selector: even innocent
    collection overrides would make a green source-only receipt ambiguous.
    """
    return argv == ("-q", "-p", "ci_host_boundary", "--portable-source-only")


def split_host_tests(items: list) -> tuple[list, tuple[str, ...]]:
    """Return portable items and exact deferred IDs, rejecting manifest drift."""
    identifiers = [item.nodeid.replace("\\", "/") for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate test IDs in collected suite")
    missing = DECLARED_DEFERRED_TESTS.difference(identifiers)
    if missing:
        raise ValueError("declared deferred tests missing from collection: " + ", ".join(sorted(missing)))
    # A newly marked owner-historical test must not silently execute in the
    # portable lane, nor be hidden by blanket -m/--deselect logic. An owner
    # must explicitly add its exact node ID to the audited host inventory.
    unlisted = []
    for item, identifier in zip(items, identifiers):
        marker = getattr(item, "get_closest_marker", None)
        if callable(marker) and marker("owner_historical") is not None and identifier not in HOST_TESTS:
            unlisted.append(identifier)
    if unlisted:
        raise ValueError("unlisted owner_historical tests: " + ", ".join(sorted(unlisted)))
    deferred = tuple(sorted(identifier for identifier in identifiers if identifier in DECLARED_DEFERRED_TESTS))
    return [item for item in items if item.nodeid.replace("\\", "/") not in DECLARED_DEFERRED_TESTS], deferred


def portable_accounting(collected: int, stats: dict, deferred: tuple[str, ...] | None,
                        *, minimum_collected: int = MIN_PORTABLE_COLLECTED) -> tuple[bool, dict]:
    """Only the exact 12 host and 2 Windows deferrals may be deselected."""
    counts = {
        category: len(stats.get(category, ()))
        for category in ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "deselected")
    }
    bad = {key: value for key, value in counts.items()
           if key not in ("passed", "deselected") and value}
    excluded = {getattr(item, "nodeid", None) for item in stats.get("deselected", ())}
    clean = (collected >= minimum_collected and counts["passed"] == collected and not bad
             and deferred is not None and set(deferred) == DECLARED_DEFERRED_TESTS
             and counts["deselected"] == len(DECLARED_DEFERRED_TESTS)
             and excluded == DECLARED_DEFERRED_TESTS)
    return clean, {"collected": collected, "minimum_collected": minimum_collected,
                   "counts": counts,
                   "host_deferred": len(set(deferred or ()) & HOST_TESTS),
                   "windows_platform_deferred": len(set(deferred or ()) & WINDOWS_PLATFORM_TESTS),
                   "prohibited": bad}


def observed_local_source() -> dict[str, str]:
    """Git observations only: the workflow must independently bind event SHA."""
    root = Path(__file__).resolve().parent
    values = {}
    for name, revision in (("git_head", "HEAD"), ("git_tree", "HEAD^{tree}")):
        result = subprocess.run(["git", "rev-parse", "--verify", revision],
                                cwd=root, capture_output=True, text=True,
                                timeout=5, check=True)
        value = result.stdout.strip()
        if len(value) != 40 or not all(c in "0123456789abcdef" for c in value):
            raise ValueError("source git identity invalid")
        values[name] = value
    return values


def pytest_addoption(parser):
    parser.addoption("--portable-source-only", action="store_true", default=False,
                     help="explicit portable lane; host tests remain unqualified")


def pytest_collection_modifyitems(session, config, items):
    if not config.getoption("--portable-source-only"):
        return
    import pytest

    # Refuse every altered CLI, including directory targets, -o overrides,
    # --ignore-glob, and plugins that shard or redefine test discovery.
    argv = tuple(config.invocation_params.args)
    if (not valid_portable_invocation(argv) or config.getoption("keyword")
            or config.getoption("markexpr")):
        raise pytest.UsageError("portable lane cannot target or filter a subset")
    try:
        retained, deferred = split_host_tests(items)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    removed = [item for item in items if item.nodeid.replace("\\", "/") in DECLARED_DEFERRED_TESTS]
    # Deliberately tell pytest which 12 host and 2 Windows cases are deferred.
    # The portable guard checks exact IDs and still forbids *all* test skips;
    # an unrecognized selection is a failed run.
    config.hook.pytest_deselected(items=removed)
    items[:] = retained
    config._vera_portable_deferred = deferred


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--portable-source-only"):
        return
    deferred = getattr(config, "_vera_portable_deferred", None)
    if deferred is None:
        return
    terminalreporter.write_sep("=", "PORTABLE_SOURCE_ONLY — HOST_INTEGRATION_UNQUALIFIED")
    host_deferred = tuple(identifier for identifier in deferred if identifier in HOST_TESTS)
    windows_deferred = tuple(identifier for identifier in deferred if identifier in WINDOWS_PLATFORM_TESTS)
    terminalreporter.write_line(
        f"Exactly {len(host_deferred)} owner-host and {len(windows_deferred)} native Windows "
        "tests explicitly deferred. No GPU/NAS/restart/evaluation qualification implied."
    )
    for nodeid in deferred:
        category = "HOST_DEFERRED" if nodeid in HOST_TESTS else "WINDOWS_PLATFORM_DEFERRED"
        terminalreporter.write_line(category + ": " + nodeid)
    try:
        source = observed_local_source()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        source = {"git_source_observation_error": type(exc).__name__}
    terminalreporter.write_line(
        "CI_HOST_BOUNDARY_RECEIPT_JSON=" + json.dumps({
            "schema": "VERA_PORTABLE_HOST_BOUNDARY_RECEIPT_V1",
            "status": "PORTABLE_SOURCE_ONLY_HOST_INTEGRATION_NOT_QUALIFIED",
            "evidence_class": "SOURCE_ONLY_NOT_AUTHENTICATED_HOST_EVIDENCE",
            "source_observation": source,
            "host_tests_deferred": list(host_deferred),
            "host_tests_executed": 0,
            "windows_platform_tests_deferred": list(windows_deferred),
            "windows_platform_tests_executed": 0,
            "authorizes_training": False,
            "authorizes_model_qualification": False,
        }, sort_keys=True, separators=(",", ":"))
    )


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--portable-source-only"):
        return
    import pytest

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    clean, evidence = portable_accounting(
        session.testscollected,
        reporter.stats if reporter is not None else {},
        getattr(session.config, "_vera_portable_deferred", None),
    )
    try:
        observed_local_source()
    except (OSError, ValueError, subprocess.SubprocessError):
        clean = False
    if reporter is None or not clean or exitstatus != pytest.ExitCode.OK:
        if reporter is not None:
            reporter.write_sep("=", f"PORTABLE_SOURCE_ACCOUNTING_FAILED: {evidence}", red=True)
        if exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
