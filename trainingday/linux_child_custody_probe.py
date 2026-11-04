"""Test-only Linux child-group custody evidence for the WSL2 Tier 1 proposal.

This module does not provide a host launch capability.  A Windows JobObject
owning wsl.exe cannot by itself attest that Linux children or CUDA kernels
die when the Windows process disappears.  Only harmless native Linux test
children may be passed to the functions below.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass


SCHEMA = "TRAININGDAY_LINUX_CHILD_CUSTODY_SOURCE_PROBE_V1"


class CustodyProbeError(RuntimeError):
    """Linux child ownership cannot be established from current evidence."""


@dataclass(frozen=True)
class LinuxProcessIdentity:
    pid: int
    start_ticks: int
    pgid: int
    session_id: int
    state: str


def read_proc_identity(pid: int) -> LinuxProcessIdentity:
    """Read the Linux kernel process identity, not a caller-supplied PID claim."""
    if sys.platform != "linux" or type(pid) is not int or pid <= 1:
        raise CustodyProbeError("Linux kernel PID required")
    try:
        # comm may contain spaces and parentheses; parse after its last ')'.
        with open(f"/proc/{pid}/stat", encoding="utf-8") as proc_stat:
            stat = proc_stat.read()
        fields = stat[stat.rfind(")") + 2 :].split()
        return LinuxProcessIdentity(
            pid=pid,
            state=fields[0],
            pgid=int(fields[2]),
            session_id=int(fields[3]),
            start_ticks=int(fields[19]),
        )
    except (OSError, ValueError, IndexError) as exc:
        raise CustodyProbeError("Linux process identity unavailable") from exc


def observe_test_child(child: subprocess.Popen) -> LinuxProcessIdentity:
    """Capture a retained test Popen's dedicated session/group identity."""
    if not isinstance(child, subprocess.Popen) or child.poll() is not None:
        raise CustodyProbeError("test child is not a live owned Popen")
    identity = read_proc_identity(child.pid)
    if identity.pgid != child.pid or identity.session_id != child.pid:
        raise CustodyProbeError("test child is not its own session/group leader")
    return identity


def terminate_owned_test_group(
    child: subprocess.Popen,
    identity: LinuxProcessIdentity,
    *,
    graceful_seconds: float = 0.5,
) -> dict:
    """Best-effort targeted cleanup of a still-live *test-owned* group.

    Identity must match the retained Popen plus kernel PID/start time.  This
    cannot establish kill-on-broker-crash in WSL, full descendant custody,
    or an enforceable GPU lease.  No positive training authority is returned.
    """
    if not isinstance(child, subprocess.Popen) or type(identity) is not LinuxProcessIdentity:
        raise CustodyProbeError("retained test child and kernel identity required")
    if type(graceful_seconds) not in (int, float) or not 0 < graceful_seconds <= 5:
        raise CustodyProbeError("invalid bounded grace interval")
    if child.pid != identity.pid or child.poll() is not None:
        raise CustodyProbeError("test child no longer owns a live group leader")
    current = read_proc_identity(child.pid)
    if (current.start_ticks != identity.start_ticks or current.pgid != child.pid
            or current.session_id != child.pid):
        raise CustodyProbeError("kernel child identity mismatch; refusing group signal")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=graceful_seconds)
    except subprocess.TimeoutExpired:
        # The leader is still retained. Check its start time immediately
        # before sending a stronger signal to the same test-owned group.
        current = read_proc_identity(child.pid)
        if current.start_ticks != identity.start_ticks or current.pgid != child.pid:
            raise CustodyProbeError("kernel child identity changed before escalation")
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=2)
    return {
        "schema": SCHEMA,
        "test_group_leader_exited": child.poll() is not None,
        "all_linux_descendants_proven_dead": False,
        "windows_wsl_broker_crash_qualified": False,
        "physical_training_authorized": False,
        "scope": "HARMLESS_LINUX_TEST_PROCESS_GROUP_ONLY",
    }
