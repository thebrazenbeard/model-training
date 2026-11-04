"""Read-only cgroup-v2 custody observations; never grants TrainingDay launch.

The owner supplies the *expected* private scope and a retained Linux process
identity. The kernel readback can contradict those claims. It cannot prove
that a prior descendant escaped the scope, that WSL2 crash containment works,
or that an external Windows broker is trusted.
"""

from __future__ import annotations

import errno
import os
import sys
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "TRAININGDAY_LINUX_CGROUP_V2_READBACK_V1"


class CgroupReadbackError(ValueError):
    """The Linux custody observation was absent, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class RetainedProcess:
    pid: int
    start_ticks: int


def _proc_stat_start_ticks(data: str, *, expected_pid: int | None = None) -> int:
    # /proc/<pid>/stat's comm may include spaces or parentheses.
    try:
        embedded_pid = int(data.split(" ", 1)[0])
    except ValueError as exc:
        raise CgroupReadbackError("invalid kernel PID in /proc stat") from exc
    if expected_pid is not None and embedded_pid != expected_pid:
        raise CgroupReadbackError("mismatched kernel PID in /proc stat")
    close = data.rfind(")")
    if close < 2 or data[close + 1 : close + 2] != " ":
        raise CgroupReadbackError("invalid /proc stat framing")
    fields = data[close + 2 :].split()
    if len(fields) < 20:
        raise CgroupReadbackError("incomplete /proc stat")
    try:
        ticks = int(fields[19])  # Linux stat field 22, post-comm index 19
    except ValueError as exc:
        raise CgroupReadbackError("invalid kernel start ticks") from exc
    if ticks < 0:
        raise CgroupReadbackError("negative kernel start ticks")
    return ticks


def _unified_path(data: str) -> str:
    paths = [line[3:] for line in data.splitlines() if line.startswith("0::")]
    if len(paths) != 1 or not paths[0].startswith("/"):
        raise CgroupReadbackError("unified cgroup-v2 membership unavailable")
    return paths[0]


def _dedicated_scope(scope: str) -> tuple[str, ...]:
    # This check prevents arbitrary traversal, not owner-trust attestation.
    if not isinstance(scope, str) or not scope.startswith("/"):
        raise CgroupReadbackError("absolute scope required")
    parts = scope.split("/")[1:]
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise CgroupReadbackError("ambiguous cgroup scope")
    if not parts[-1].startswith("trainingday-") or len(parts[-1]) < 20:
        raise CgroupReadbackError("scope is not a named TrainingDay candidate")
    return tuple(parts)


def _parse_events(data: str) -> int:
    values: dict[str, str] = {}
    for line in data.splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] in values:
            raise CgroupReadbackError("ambiguous cgroup.events")
        values[fields[0]] = fields[1]
    if values.get("populated") not in ("0", "1"):
        raise CgroupReadbackError("missing cgroup.events populated")
    return int(values["populated"])


def _parse_pids(data: str) -> tuple[int, ...]:
    try:
        pids = tuple(int(line) for line in data.splitlines() if line.strip())
    except ValueError as exc:
        raise CgroupReadbackError("invalid kernel cgroup PID list") from exc
    if any(pid <= 1 for pid in pids) or len(set(pids)) != len(pids):
        raise CgroupReadbackError("ambiguous kernel cgroup PID list")
    return pids


def _read_scope_files(cgroup_root: Path, parts: tuple[str, ...]) -> tuple[str, str]:
    """Open scope and leaf files through stable no-symlink directory handles."""
    if sys.platform != "linux":
        raise CgroupReadbackError("Linux cgroup-v2 kernel required")
    # lstat/is_symlink alone would race against replacement between check/open.
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    leaf_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory_fd = None
    try:
        directory_fd = os.open(cgroup_root, dir_flags)
        for part in parts:
            next_fd = os.open(part, dir_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd

        def read_leaf(name: str) -> str:
            leaf_fd = os.open(name, leaf_flags, dir_fd=directory_fd)
            with os.fdopen(leaf_fd, "r", encoding="utf-8") as leaf:
                data = leaf.read(65537)
            if len(data) > 65536:
                raise CgroupReadbackError("oversize kernel cgroup readback")
            return data

        return read_leaf("cgroup.events"), read_leaf("cgroup.procs")
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise CgroupReadbackError("symlink or non-directory in cgroup scope") from exc
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def inspect_read_only(
    retained: RetainedProcess,
    expected_scope: str,
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict:
    """Compare owner claims with Linux readback; always emit physical HOLD.

    A missing/stale process or unavailable scope is an error, never evidence
    that all descendants exited. Direct cgroup.procs omits nested children;
    cgroup.events populated covers the descendants at observation time only.
    """
    if not isinstance(retained, RetainedProcess) or type(retained.pid) is not int:
        raise CgroupReadbackError("retained kernel process identity required")
    if retained.pid <= 1 or type(retained.start_ticks) is not int or retained.start_ticks < 0:
        raise CgroupReadbackError("invalid retained PID/start ticks")
    parts = _dedicated_scope(expected_scope)
    proc_root = Path(proc_root)
    cgroup_root = Path(cgroup_root)
    try:
        observed_ticks = _proc_stat_start_ticks(
            (proc_root / str(retained.pid) / "stat").read_text(encoding="utf-8"),
            expected_pid=retained.pid,
        )
        observed_scope = _unified_path(
            (proc_root / str(retained.pid) / "cgroup").read_text(encoding="utf-8")
        )
        if observed_ticks != retained.start_ticks:
            raise CgroupReadbackError("stale or reused Linux PID")
        if observed_scope != expected_scope:
            raise CgroupReadbackError("observed Linux cgroup mismatch")
        events_text, procs_text = _read_scope_files(cgroup_root, parts)
        populated = _parse_events(events_text)
        direct_pids = _parse_pids(procs_text)
        if populated == 0 and direct_pids:
            raise CgroupReadbackError("contradictory cgroup population readback")
        if retained.pid not in direct_pids:
            raise CgroupReadbackError("owner process not directly in claimed scope")
        # The two separate /proc and cgroup files are not an atomic snapshot.
        # A second kernel observation rejects detected PID exit/reuse/movement.
        again_ticks = _proc_stat_start_ticks(
            (proc_root / str(retained.pid) / "stat").read_text(encoding="utf-8"),
            expected_pid=retained.pid,
        )
        again_scope = _unified_path(
            (proc_root / str(retained.pid) / "cgroup").read_text(encoding="utf-8")
        )
        if again_ticks != retained.start_ticks:
            raise CgroupReadbackError("stale or reused Linux PID during readback")
        if again_scope != expected_scope:
            raise CgroupReadbackError("observed Linux cgroup changed during readback")
    except OSError as exc:
        raise CgroupReadbackError("kernel cgroup readback unavailable") from exc
    return {
        "schema": SCHEMA,
        "status": "SOURCE_OBSERVATION_ONLY_TRAINING_HOLD",
        "observed_pid": retained.pid,
        "observed_start_ticks": observed_ticks,
        "observed_scope": observed_scope,
        "readback_origin": "DEFAULT_LINUX_PATHS" if proc_root == Path("/proc") and cgroup_root == Path("/sys/fs/cgroup") else "CALLER_SUPPLIED_TEST_PATHS",
        "readback_atomic": False,
        "populated_at_observation": populated == 1,
        "direct_process_count": len(direct_pids),
        "exclusive_cgroup_delegation_verified": False,
        "all_prior_descendants_proven_dead": False,
        "windows_wsl2_crash_qualified": False,
        "physical_training_authorized": False,
    }


def scope_empty_readback(
    expected_scope: str, *, cgroup_root: Path = Path("/sys/fs/cgroup")
) -> dict:
    """Post-stop snapshot only; missing scope is UNKNOWN, not success."""
    parts = _dedicated_scope(expected_scope)
    try:
        events_text, procs_text = _read_scope_files(Path(cgroup_root), parts)
        populated = _parse_events(events_text)
        direct_pids = _parse_pids(procs_text)
        if populated == 0 and direct_pids:
            raise CgroupReadbackError("contradictory cgroup population readback")
    except OSError as exc:
        raise CgroupReadbackError("post-stop cgroup readback unavailable") from exc
    return {
        "schema": SCHEMA,
        "status": "SOURCE_OBSERVATION_ONLY_TRAINING_HOLD",
        "scope": expected_scope,
        "readback_origin": "DEFAULT_LINUX_PATHS" if Path(cgroup_root) == Path("/sys/fs/cgroup") else "CALLER_SUPPLIED_TEST_PATHS",
        "readback_atomic": False,
        "scope_empty_at_observation": populated == 0 and len(direct_pids) == 0,
        "all_prior_descendants_proven_dead": False,
        "windows_wsl2_crash_qualified": False,
        "physical_training_authorized": False,
    }
