"""Non-CUDA source-only cgroup-v2 identity and post-stop HOLD tests."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trainingday.linux_cgroup_scope_readback import (
    CgroupReadbackError,
    RetainedProcess,
    _proc_stat_start_ticks,
    _unified_path,
    inspect_read_only,
    scope_empty_readback,
)


SCOPE = "/user.slice/vera/trainingday-synthetic-test-only-0123456789"


class LinuxCgroupScopeReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.proc_root = root / "proc"
        self.cg_root = root / "sys/fs/cgroup"
        self.pid = 98765
        self.start_ticks = 123456789
        proc = self.proc_root / str(self.pid)
        proc.mkdir(parents=True)
        self.process_stat = proc / "stat"
        self.process_cgroup = proc / "cgroup"
        self.location = self.cg_root / SCOPE.lstrip("/")
        self.location.mkdir(parents=True)
        self.events = self.location / "cgroup.events"
        self.procs = self.location / "cgroup.procs"
        self.events.write_text("populated 1\nfrozen 0\n", encoding="utf-8")
        self.procs.write_text(f"{self.pid}\n", encoding="utf-8")
        self.process_cgroup.write_text(f"0::{SCOPE}\n", encoding="utf-8")
        self.process_stat.write_text(self.make_stat(self.start_ticks), encoding="utf-8")

    def make_stat(self, ticks):
        fields = ["0"] * 22
        fields[0] = "S"
        fields[19] = str(ticks)
        return f"{self.pid} (a weird ) proc ( name) " + " ".join(fields) + "\n"

    def probe(self, *, ticks=None, scope=SCOPE):
        return inspect_read_only(
            RetainedProcess(self.pid, self.start_ticks if ticks is None else ticks),
            scope,
            proc_root=self.proc_root,
            cgroup_root=self.cg_root,
        )

    def test_kernel_membership_readback_still_never_authorizes_training(self):
        status = self.probe()
        self.assertEqual(status["observed_pid"], self.pid)
        self.assertEqual(status["observed_start_ticks"], self.start_ticks)
        self.assertTrue(status["populated_at_observation"])
        self.assertEqual(status["direct_process_count"], 1)
        self.assertFalse(status["exclusive_cgroup_delegation_verified"])
        self.assertFalse(status["all_prior_descendants_proven_dead"])
        self.assertFalse(status["windows_wsl2_crash_qualified"])
        self.assertFalse(status["physical_training_authorized"])

    def test_stale_or_reused_pid_is_rejected(self):
        with self.assertRaisesRegex(CgroupReadbackError, "stale or reused"):
            self.probe(ticks=self.start_ticks + 1)

    def test_forged_scope_cannot_be_substituted(self):
        with self.assertRaisesRegex(CgroupReadbackError, "cgroup mismatch"):
            self.probe(scope="/user.slice/vera/trainingday-synthetic-test-only-OTHER")

    def test_missing_kernel_cgroup_directory_is_unknown_not_empty(self):
        with self.assertRaisesRegex(CgroupReadbackError, "unavailable"):
            scope_empty_readback(
                "/user.slice/vera/trainingday-synthetic-test-only-missing",
                cgroup_root=self.cg_root,
            )

    def test_poststop_empty_observation_is_not_crash_qualification(self):
        self.procs.write_text("", encoding="utf-8")
        self.events.write_text("populated 0\n", encoding="utf-8")
        status = scope_empty_readback(SCOPE, cgroup_root=self.cg_root)
        self.assertTrue(status["scope_empty_at_observation"])
        self.assertFalse(status["all_prior_descendants_proven_dead"])
        self.assertFalse(status["windows_wsl2_crash_qualified"])
        self.assertFalse(status["physical_training_authorized"])

    def test_populated_nested_descendants_are_not_missed_by_empty_procs(self):
        self.procs.write_text("", encoding="utf-8")
        self.events.write_text("populated 1\n", encoding="utf-8")
        status = scope_empty_readback(SCOPE, cgroup_root=self.cg_root)
        self.assertFalse(status["scope_empty_at_observation"])

    def test_contradictory_empty_events_with_live_process_rejected(self):
        self.events.write_text("populated 0\n", encoding="utf-8")
        with self.assertRaisesRegex(CgroupReadbackError, "contradictory"):
            self.probe()
        with self.assertRaisesRegex(CgroupReadbackError, "contradictory"):
            scope_empty_readback(SCOPE, cgroup_root=self.cg_root)

    def test_duplicate_or_malformed_kernel_observation_rejected(self):
        self.events.write_text("populated 0\npopulated 1\n", encoding="utf-8")
        with self.assertRaisesRegex(CgroupReadbackError, "ambiguous"):
            scope_empty_readback(SCOPE, cgroup_root=self.cg_root)
        self.events.write_text("populated nan\n", encoding="utf-8")
        with self.assertRaisesRegex(CgroupReadbackError, "missing"):
            scope_empty_readback(SCOPE, cgroup_root=self.cg_root)

    def test_symlink_substitution_refused_even_when_target_has_good_files(self):
        parent = self.cg_root / "user.slice" / "vera"
        alias = self.cg_root / "alias"
        alias.symlink_to(parent, target_is_directory=True)
        with self.assertRaisesRegex(CgroupReadbackError, "symlink"):
            scope_empty_readback("/alias/trainingday-synthetic-test-only-0123456789", cgroup_root=self.cg_root)

    def test_broad_scope_and_path_traversal_rejected(self):
        for scope in ("/", "/user.slice", "/tmp/../trainingday-synthetic-test-only-0123456789", "/trainingday-x", "../trainingday-synthetic-test-only-0123456789"):
            with self.subTest(scope=scope):
                with self.assertRaises(CgroupReadbackError):
                    scope_empty_readback(scope, cgroup_root=self.cg_root)

    @unittest.skipUnless(os.name == "posix" and Path("/proc/self/cgroup").exists(), "Linux /proc only")
    def test_real_kernel_readonly_unified_cgroup_identity(self):
        # Observational test only: no change to the current shared cgroup.
        own_stat = Path("/proc/self/stat").read_text(encoding="utf-8")
        self.assertGreaterEqual(_proc_stat_start_ticks(own_stat), 0)
        actual = _unified_path(Path("/proc/self/cgroup").read_text(encoding="utf-8"))
        self.assertTrue(actual.startswith("/"))


if __name__ == "__main__":
    unittest.main()
