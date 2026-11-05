"""Explicit-owner, Windows-only one-shot, CPU-only Q4 baseline process custody.

Usage after independent review/install: py -3.13 supervisor.py --self-test
Then, after approval and exact pinning: py -3.13 supervisor.py --run

This is intentionally NOT imported by TRAININGDAY training launch authority.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid
import policy

from policy import (
    GIB, HOST, Hold, MIN_RUNTIME_FREE_BYTES, RUNNER_ROOT,
    SELFTEST_STATE_ROOT, STATE_ROOT, parse_plan, required_prelaunch_free,
    state_root_for_mode, blocking_processes_for_mode, ollama_inventory_empty,
    competing_model_workers,
)

if os.name != "nt":
    raise SystemExit("HOLD: Windows-only process supervisor")

from windows_job import (  # noqa: E402 - Windows-only import
    PinnedReadFile, Win32Hold, assert_not_reparse,
    available_physical_bytes, host_cpu_percent, run_job,
    running_executable_names, running_named_processes,
    observed_process_snapshot, observed_windows_identity,
)


STATE = Path(str(STATE_ROOT))
SELFTEST_STATE = Path(str(SELFTEST_STATE_ROOT))
PLAN_PATH = STATE / "approved_q4_baseline_plan.json"


@contextmanager
def single_slot():
    # Both self-test and model run use the *same* host-owned mutex file. It
    # lives outside the atomic trial leaf, which self-test cannot create.
    # Windows file-region locks are kernel-enforced, released if we die.
    with open(SELFTEST_STATE / "oneshot.lock", "a+b") as slot:
        slot.seek(0, os.SEEK_END)
        if slot.tell() == 0:
            slot.write(b"0")
            slot.flush()
        slot.seek(0)
        try:
            msvcrt.locking(slot.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise Hold("another one-shot trial owns the VN1 slot") from exc
        try:
            yield
        finally:
            slot.seek(0)
            msvcrt.locking(slot.fileno(), msvcrt.LK_UNLCK, 1)


def _receipt(path: Path, body: dict) -> None:
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(tmp, "x", encoding="utf-8", newline="\n") as out:
            json.dump(body, out, sort_keys=True, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class _RejectLoopbackRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Hold("Ollama occupancy endpoint redirect forbidden")


def _ollama_local_inventory_empty() -> None:
    """GET-only, fixed loopback API; never starts/stops/loads Ollama."""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectLoopbackRedirect()
    )
    request = urllib.request.Request("http://127.0.0.1:11434/api/ps", method="GET")
    try:
        with opener.open(request, timeout=0.6) as response:
            if response.status != 200:
                raise Hold("Ollama API /api/ps did not return HTTP 200")
            payload = response.read(65537)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise Hold("unable to read current local Ollama model occupancy") from exc
    ollama_inventory_empty(payload)


class OllamaOccupancyGuard:
    """Conservative cooperative occupancy screen, not a host lease.

    API emptiness can be stale/incorrect; supplement with OS-observed PID,
    creation time, RSS, known workers, before-resume and 1-second rechecks.
    """

    def __init__(self):
        self.daemon_identity: tuple[int, int] | tuple[()] | None = None
        self.check_count = 0
        self.daemon_seen = False

    def check(self, owner_pid: int | None = None) -> None:
        proc = running_named_processes()
        # Kernel-owned suspended Job child may be excluded by exact PID only.
        # All other worker PIDs remain competing processes.
        actual_workers = competing_model_workers(proc, owner_pid)
        if actual_workers:
            raise Hold(f"active inference/trainer executable: {', '.join(actual_workers)}")
        pids = proc.get("ollama.exe", set())
        if len(pids) > 1:
            raise Hold("multiple Ollama daemons/runners are not distinguishable")
        if pids:
            pid = next(iter(pids))
            created, rss = observed_process_snapshot(pid)
            if rss > 128 * 1024 ** 2:
                raise Hold("resident Ollama process exceeds idle RSS ceiling")
            identity: tuple[int, int] | tuple[()] = (pid, created)
            if self.daemon_identity is not None and self.daemon_identity != identity:
                raise Hold("Ollama process identity changed during bounded run")
            self.daemon_identity = identity
            self.daemon_seen = True
            _ollama_local_inventory_empty()
        else:
            if self.daemon_identity not in (None, ()):
                raise Hold("Ollama daemon disappeared during model run")
            self.daemon_identity = ()
        self.check_count += 1


def _host_preflight(required_free: int, mode: str,
                    occupancy_guard: OllamaOccupancyGuard | None = None) -> dict:
    _require_owner_identity()
    if os.name != "nt" or sys.maxsize <= 2 ** 32:
        raise Hold("64-bit Windows Python required")
    free = available_physical_bytes()
    if free < required_free:
        raise Hold(f"insufficient available RAM: {free} < {required_free} bytes")
    if mode == "--run":
        if occupancy_guard is None:
            raise Hold("model run requires independent current Ollama occupancy guard")
        occupancy_guard.check()
    present = sorted(running_executable_names() & blocking_processes_for_mode(mode))
    if mode == "--run":
        # The single observed idle daemon is admitted only by the guard,
        # never by a general process-name exception.
        present = [name for name in present if name != "ollama.exe"]
    if present:
        raise Hold(f"existing model/training workload detected: {', '.join(present)}")
    cpu_busy = host_cpu_percent()
    if cpu_busy > 60:
        raise Hold(f"host occupied: {cpu_busy:.1f}% CPU over one second")
    if occupancy_guard is not None:
        occupancy_guard.check()
    return {"free_physical_bytes": free, "cpu_busy_percent": round(cpu_busy, 2),
            "conflicting_processes": present}


def _require_owner_identity() -> None:
    hostname, principal = observed_windows_identity()
    if hostname.upper() != HOST or principal.casefold() != r"vera-node-1\vera":
        raise Hold("observed Windows host or security principal did not match approved owner")


def _state_ready(mode: str) -> None:
    # Allowed scoped one-shot installation directory; no preexisting files
    # may be silently overwritten or chosen as an executable/script.
    state = Path(str(state_root_for_mode(mode)))
    state.mkdir(parents=False, exist_ok=True)
    assert_not_reparse(state)


@contextmanager
def _verified_runtime_bundle():
    expected = policy.REVIEWED_RUNTIME_FILES_SHA256
    if not expected or expected.get("llama-cli.exe") != policy.REVIEWED_RUNNER_SHA256:
        raise Hold("unapproved native executable/DLL bundle")
    # Reject injected native DLLs or unexpected executables even if the exact
    # main llama-cli.exe hash matches an independently approved value.
    bin_dir = Path(str(RUNNER_ROOT))
    assert_not_reparse(bin_dir)
    actual = {
        p.name.casefold(): p for p in bin_dir.iterdir()
        if p.suffix.casefold() in (".exe", ".dll", ".pyd")
    }
    if set(actual) != set(expected):
        raise Hold("unapproved or missing EXE/DLL/PYD in runner staging directory")
    with ExitStack() as held_files:
        for name in sorted(expected):
            if not isinstance(expected[name], str) or len(expected[name]) != 64:
                raise Hold("unreviewed runtime bundle digest")
            held = held_files.enter_context(PinnedReadFile(actual[name]))
            if held.sha256() != expected[name]:
                raise Hold(f"runtime bundle bytes diverged: {name}")
        yield


def self_test() -> dict:
    """Fixed harmless child validates Windows Job kill and signed-off bounds.

    Does not access, hash, load, or infer with any model.
    """
    trial_id = uuid.uuid4().hex
    receipt_path = SELFTEST_STATE / f"selftest-{trial_id}.json"
    with single_slot():
        host = _host_preflight(5 * GIB, "--self-test")
        python_exe = Path(sys.executable)
        # Hard-coded harmless child; caller cannot provide a command or path.
        if str(python_exe).casefold() != (
            r"c:\users\vera\appdata\local\programs\python\python313\python.exe"
        ).casefold():
            raise Hold("selftest requires the exact approved Python 3.13 path")
        with PinnedReadFile(python_exe) as held:
            executable_sha = held.sha256()
            result = run_job(
                executable=python_exe,
                argv=[str(python_exe), "-I", "-c", "import time; time.sleep(30)"],
                cwd=SELFTEST_STATE,
                stdout_path=SELFTEST_STATE / f"selftest-{trial_id}.out",
                stderr_path=SELFTEST_STATE / f"selftest-{trial_id}.err",
                memory_cap=512 * 1024 ** 2, cpu_percent=20, wall_seconds=6,
                runtime_free_floor=MIN_RUNTIME_FREE_BYTES, forced_test_stop=True,
            )
        status = "PASS" if result.reason == "SELFTEST_FORCED_STOP" and result.exit_code == 90 else "FAIL"
        body = {
            "schema": "VERA_VN1_CPU_JOB_SELFTEST_V1", "trial_id": trial_id,
            "state": status, "host_preflight": host, "python_sha256": executable_sha,
            "child_pid": result.pid, "exit_code": result.exit_code,
            "reason": result.reason, "peak_job_commit_bytes": result.peak_job_commit_bytes,
            "job_active_processes_after": result.job_active_processes_after,
            "child_handle_signaled": result.process_handle_signaled,
            "peak_working_set_bytes": result.peak_working_set_bytes,
        }
        _receipt(receipt_path, body)
        if status != "PASS":
            raise Hold("JobObject child cleanup selftest did not pass")
        return body


def run_public_q4_baseline() -> dict:
    with single_slot():
        occupancy = OllamaOccupancyGuard()
        assert_not_reparse(PLAN_PATH)
        with open(PLAN_PATH, "rb") as plan_file:
            raw = plan_file.read(4097)
        if len(raw) > 4096:
            raise Hold("plan exceeds 4096-byte maximum")
        plan = parse_plan(raw)
        plan_sha = hashlib.sha256(raw).hexdigest()
        host = _host_preflight(required_prelaunch_free(plan), "--run", occupancy)
        # Source-only policy checks are necessary but not sufficient; held OS
        # file handles block mutation/rename until the child is reaped.
        with _verified_runtime_bundle(), PinnedReadFile(Path(str(plan.model))) as model:
            if model.sha256() != plan.model_sha256:
                raise Hold("Q4 artifact bytes do not match published SHA-256")
            if available_physical_bytes() < required_prelaunch_free(plan):
                raise Hold("available RAM fell below floor during hash verification")
            occupancy.check()

            # Consume the plan atomically before any CreateProcessW call;
            # crash/ambiguous outcome permanently prevents automatic replay.
            attempt = STATE / f"consumed-{plan.plan_id}.json"
            try:
                with open(attempt, "x", encoding="utf-8") as marker:
                    json.dump({"schema": "VERA_VN1_PLAN_CONSUMED_V1",
                               "plan_id": plan.plan_id, "plan_sha256": plan_sha,
                               "nonce": uuid.uuid4().hex, "at_utc": time.time()}, marker)
                    marker.flush()
                    os.fsync(marker.fileno())
            except FileExistsError as exc:
                raise Hold("plan already consumed; no replay") from exc

            started = time.time()
            receipt_path = STATE / f"receipt-{plan.plan_id}.json"
            try:
                result = run_job(
                    executable=Path(str(plan.executable)), argv=plan.args(),
                    cwd=Path(str(RUNNER_ROOT)),
                    stdout_path=STATE / f"llama-{plan.plan_id}.out",
                    stderr_path=STATE / f"llama-{plan.plan_id}.err",
                    memory_cap=plan.job_memory_bytes,
                    cpu_percent=plan.cpu_rate_percent,
                    wall_seconds=plan.wall_seconds,
                    runtime_free_floor=MIN_RUNTIME_FREE_BYTES,
                    occupancy_guard=occupancy.check,
                )
                body = {
                    "schema": "VERA_VN1_Q4_BASELINE_RECEIPT_V1",
                    "subject": "UNVERIFIED_THIRDPARTY_Q4_BASE_ONLY",
                    "official_qwen_derivation_verified": False,
                    "vera_persona_or_r16_loaded": False,
                    "plan_sha256": plan_sha, "plan_id": plan.plan_id,
                    "host_preflight": host, "start_utc": started,
                    "elapsed_wall_seconds": round(time.time() - started, 3),
                    "runner_sha256": plan.executable_sha256,
                    "model_sha256": plan.model_sha256,
                    "pid": result.pid, "exit_code": result.exit_code,
                    "termination_reason": result.reason,
                    "peak_job_commit_bytes": result.peak_job_commit_bytes,
                    "last_free_ram_bytes": result.last_free_ram_bytes,
                    "job_active_processes_after": result.job_active_processes_after,
                    "child_handle_signaled": result.process_handle_signaled,
                    "peak_working_set_bytes": result.peak_working_set_bytes,
                    "ollama_observed_idle_daemon": occupancy.daemon_seen,
                    "occupancy_guard_checks": occupancy.check_count,
                }
            except Exception as exc:
                body = {
                    "schema": "VERA_VN1_Q4_BASELINE_RECEIPT_V1",
                    "subject": "UNVERIFIED_THIRDPARTY_Q4_BASE_ONLY",
                    "plan_sha256": plan_sha, "plan_id": plan.plan_id,
                    "state": "UNKNOWN_NO_REPLAY", "failure_type": type(exc).__name__,
                    "start_utc": started, "elapsed_wall_seconds": round(time.time() - started, 3),
                }
                _receipt(receipt_path, body)
                raise
            _receipt(receipt_path, body)
            return body


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("--self-test", "--run"):
        print("HOLD: exact mode --self-test or --run required", file=sys.stderr)
        return 2
    try:
        _require_owner_identity()
        _state_ready(sys.argv[1])
        body = self_test() if sys.argv[1] == "--self-test" else run_public_q4_baseline()
        print(json.dumps(body, sort_keys=True))
        return 0 if body.get("state") == "PASS" or body.get("exit_code") == 0 else 1
    except (Hold, Win32Hold, OSError, ValueError) as exc:
        print("HOLD: " + str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
