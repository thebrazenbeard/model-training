"""Fail closed when a CI pytest run silently omits or miscounts tests.

Opt in with ``python -m pytest -p ci_full_suite_guard``. This plugin does not
alter developer test commands. A zero-skip run is deliberately *not* a promise
that external host-bound integration evidence exists.
"""

from __future__ import annotations

from collections.abc import Mapping


COUNTED_OUTCOMES = (
    "passed", "failed", "error", "skipped", "xfailed", "xpassed", "deselected"
)
PROHIBITED_OUTCOMES = COUNTED_OUTCOMES[1:]


def audit_counts(collected: int, stats: Mapping[str, object]) -> tuple[bool, str]:
    """Validate the terminal reporter's accounting, including deselections."""
    counts = {outcome: len(stats.get(outcome, ())) for outcome in COUNTED_OUTCOMES}
    prohibited = {name: counts[name] for name in PROHIBITED_OUTCOMES if counts[name]}
    clean = collected > 0 and counts["passed"] == collected and not prohibited
    detail = (
        f"collected={collected}, passed={counts['passed']}, "
        f"prohibited={prohibited}; no empty run, failure, error, skip, "
        "xfail, xpass, or deselection may count as full-suite green"
    )
    return clean, detail


def pytest_sessionfinish(session, exitstatus):
    """Reject incomplete runs without hiding an earlier nonzero pytest exit."""
    import pytest

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    clean, detail = audit_counts(
        session.testscollected,
        reporter.stats if reporter is not None else {},
    )
    if reporter is None or not clean or exitstatus != pytest.ExitCode.OK:
        message = f"CI full-suite accounting FAILED: {detail}"
        if reporter is None:
            message += "; terminal reporter absent"
        if reporter is not None:
            reporter.write_sep("=", message, red=True)
        else:
            print(message)
        if exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
