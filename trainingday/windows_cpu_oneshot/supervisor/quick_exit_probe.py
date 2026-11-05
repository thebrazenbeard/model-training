"""Harmless Windows JobObject quick-exit regression; NEVER loads a model.

Exercises the native accounting race seen when a child exits in milliseconds.
Runs one fixed Python -I -c pass child in the same bounded Job custody path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

if os.name != "nt":
    raise SystemExit("HOLD: Windows-only regression probe")

import supervisor
from policy import GIB, Hold, MIN_RUNTIME_FREE_BYTES
from windows_job import PinnedReadFile, Win32Hold, run_job


APPROVED_PYTHON = Path(r"C:\Users\Vera\AppData\Local\Programs\Python\Python313\python.exe")
APPROVED_PYTHON_SHA256 = "85b71d8c6ec1905935f74be0c9869aae198d00e98f39df699ec66f9c5a84cecd"


def main() -> int:
    if len(sys.argv) != 1:
        raise Hold("no caller-supplied arguments accepted")
    supervisor._require_owner_identity()
    supervisor._state_ready("--self-test")
    nonce = uuid.uuid4().hex
    root = supervisor.SELFTEST_STATE
    with supervisor.single_slot():
        host = supervisor._host_preflight(5 * GIB, "--self-test")
        if Path(sys.executable) != APPROVED_PYTHON:
            raise Hold("unapproved Python executable")
        with PinnedReadFile(APPROVED_PYTHON) as python_exe:
            if python_exe.sha256() != APPROVED_PYTHON_SHA256:
                raise Hold("Python executable digest changed")
            result = run_job(
                executable=APPROVED_PYTHON,
                argv=[str(APPROVED_PYTHON), "-I", "-c", "pass"],
                cwd=root,
                stdout_path=root / f"quick-{nonce}.out",
                stderr_path=root / f"quick-{nonce}.err",
                memory_cap=512 * 1024 ** 2,
                cpu_percent=20,
                wall_seconds=8,
                runtime_free_floor=MIN_RUNTIME_FREE_BYTES,
            )
        body = {
            "schema": "VERA_VN1_JOB_QUICK_EXIT_PROBE_V1",
            "nonce": nonce,
            "host_preflight": host,
            "child_pid": result.pid,
            "exit_code": result.exit_code,
            "reason": result.reason,
            "child_handle_signaled": result.process_handle_signaled,
            "job_active_processes_after": result.job_active_processes_after,
            "accounting_poll_count": result.accounting_poll_count,
            "accounting_wait_seconds": result.accounting_wait_seconds,
            "model_loaded": False,
            "state": "PASS" if (
                result.exit_code == 0 and result.reason == "CHILD_EXITED"
                and result.process_handle_signaled
                and result.job_active_processes_after == 0
            ) else "FAIL",
        }
        supervisor._receipt(root / f"quick-{nonce}.json", body)
        print(json.dumps(body, sort_keys=True))
        return 0 if body["state"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Hold, Win32Hold, OSError, ValueError) as exc:
        print("HOLD: " + str(exc), file=sys.stderr)
        raise SystemExit(3)
