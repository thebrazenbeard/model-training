"""Real Linux negative witness; never starts CUDA or user workloads."""

import os
import signal
import subprocess
import sys
import unittest
from dataclasses import replace

from trainingday.linux_child_custody_probe import (
    CustodyProbeError,
    observe_test_child,
    read_proc_identity,
    terminate_owned_test_group,
)


@unittest.skipUnless(sys.platform == "linux", "Linux /proc test only")
class LinuxChildCustodyProbeTests(unittest.TestCase):
    def start_sleep(self):
        return subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_owned_harmless_child_group_can_be_stopped(self):
        child = self.start_sleep()
        try:
            identity = observe_test_child(child)
            self.assertEqual(identity.pgid, child.pid)
            receipt = terminate_owned_test_group(child, identity)
            self.assertTrue(receipt["test_group_leader_exited"])
            self.assertFalse(receipt["all_linux_descendants_proven_dead"])
            self.assertFalse(receipt["windows_wsl_broker_crash_qualified"])
            self.assertFalse(receipt["physical_training_authorized"])
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)

    def test_kernel_creation_identity_forgery_is_rejected_before_signal(self):
        child = self.start_sleep()
        try:
            identity = observe_test_child(child)
            with self.assertRaises(CustodyProbeError):
                terminate_owned_test_group(child, replace(identity, start_ticks=identity.start_ticks + 1))
            self.assertIsNone(child.poll())
            self.assertTrue(terminate_owned_test_group(child, identity)["test_group_leader_exited"])
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)

    def test_not_group_leader_rejected(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            with self.assertRaises(CustodyProbeError):
                observe_test_child(child)
        finally:
            child.terminate()
            child.wait(timeout=2)

    def test_killing_supervisor_does_not_prove_linux_child_death(self):
        """Red-team a Windows-broker disappearance analogue, then clean up."""
        script = (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
            "start_new_session=True);"
            "print(p.pid,flush=True);time.sleep(30)"
        )
        supervisor = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        child_pid = None
        try:
            line = supervisor.stdout.readline().strip()
            child_pid = int(line)
            identity = read_proc_identity(child_pid)
            self.assertEqual(identity.pgid, child_pid)
            os.kill(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=2)
            # The unrelated session is still running: parent loss alone is
            # NOT kill-on-owner-loss custody. No positive grant may rely on it.
            current = read_proc_identity(child_pid)
            self.assertEqual(current.start_ticks, identity.start_ticks)
            self.assertNotEqual(current.state, "Z")
        finally:
            if child_pid is not None:
                try:
                    current = read_proc_identity(child_pid)
                    if current.start_ticks == identity.start_ticks and current.pgid == child_pid:
                        os.killpg(child_pid, signal.SIGKILL)
                except (CustodyProbeError, ProcessLookupError):
                    pass
            if supervisor.poll() is None:
                supervisor.kill()
                supervisor.wait(timeout=2)
            if supervisor.stdout is not None:
                supervisor.stdout.close()
