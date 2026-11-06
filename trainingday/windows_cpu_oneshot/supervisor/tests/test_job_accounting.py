"""Offline source-exact accounting poll characterization.

`windows_job.py` intentionally cannot import on Linux. Load only its pure
post-exit polling function by AST; no Win32 API is fabricated or qualified.
The native handle-release order has a separate source-order assertion and
still needs a real Windows harmless-child integration test.
"""

import ast
from pathlib import Path
import time
from typing import Callable
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "windows_job.py").read_text()


def load_source_exact_poll():
    module = ast.parse(SOURCE)
    candidate = next(x for x in module.body if isinstance(x, ast.FunctionDef)
                     and x.name == "bounded_accounting_drain")
    namespace = {"time": time, "Callable": Callable, "Win32Hold": RuntimeError}
    exec(compile(ast.Module(body=[candidate], type_ignores=[]),
                 "windows_job.py:bounded_accounting_drain", "exec"), namespace)
    return namespace["bounded_accounting_drain"]


class FakeClock:
    def __init__(self):
        self.current = 0.0

    def now(self):
        return self.current

    def pause(self, seconds):
        self.current += seconds


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.poll = load_source_exact_poll()

    def test_zero_already_uses_single_query(self):
        t = FakeClock()
        polls, elapsed = self.poll(lambda: 0, clock=t.now, pause=t.pause)
        self.assertEqual(polls, 1)
        self.assertEqual(elapsed, 0)

    def test_delayed_job_accounting_drains_within_bound(self):
        t = FakeClock()
        values = iter((1, 1, 0))
        polls, elapsed = self.poll(lambda: next(values), clock=t.now, pause=t.pause)
        self.assertEqual(polls, 3)
        self.assertAlmostEqual(elapsed, 0.1)

    def test_stuck_nonzero_remains_hold_without_replay(self):
        t = FakeClock()
        with self.assertRaisesRegex(RuntimeError, "did not drain"):
            self.poll(lambda: 1, timeout_seconds=0.1, interval_seconds=0.05,
                      clock=t.now, pause=t.pause)
        self.assertAlmostEqual(t.current, 0.1)

    def test_invalid_count_or_budget_fails_closed(self):
        t = FakeClock()
        for bad in (-1, None, True, "1"):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    self.poll(lambda: bad, clock=t.now, pause=t.pause)
        with self.assertRaises(RuntimeError):
            self.poll(lambda: 0, timeout_seconds=0)

    def test_process_handle_closed_only_after_latched_exit_and_before_poll(self):
        # Structural check complements runtime tests but does not prove ABI.
        exit_at = SOURCE.index("GetExitCodeProcess(child.hProcess")
        release_at = SOURCE.index("CloseHandle(child.hProcess)")
        drain_at = SOURCE.index("bounded_accounting_drain(read_active)")
        self.assertLess(exit_at, release_at)
        self.assertLess(release_at, drain_at)


if __name__ == "__main__":
    unittest.main()
