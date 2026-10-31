"""Source-only, fail-closed process budget for an externally authorized R15 pilot.

This optional wrapper monitors only a child it starts. It does not contain
descendant processes, so its enforcement status remains INCOMPLETE until a
verified host process-tree boundary (e.g., Windows Job Object) exists. It is
NOT a host lease, physical-training qualification, independent GPU attestation,
or private-bank permission. Its sampled temperature cannot bound peaks between
observations. Do not use it to authorize physical training.

Example (after independent host lease and corpus/base verification):
    python -m trainingday.r38_budget_guard --receipt-dir <fresh-dir> \
      --max-seconds 900 --max-gpu-temp-c 82 -- \
      --manifest <manifest> --base-binding <binding> --work-root <root> \
      --output-root <root> --run-id <fresh-id> --mode correct --steps 1 \
      --cases 8 --expected-initial-digest <zero-arm-sha256>

The guard never automatically retries a terminated child; preserve its output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class BudgetHold(RuntimeError):
    """The requested guard operation did not satisfy safety preconditions."""


def assess_budget(elapsed_s: float, timeout_s: float,
                  gpu_temp_c: float | None, max_temp_c: float) -> str:
    """Return the sampled disposition, failing closed on missing telemetry."""
    if (not isinstance(elapsed_s, (int, float)) or not math.isfinite(elapsed_s)
            or elapsed_s < 0 or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0
            or not isinstance(max_temp_c, (int, float))
            or not math.isfinite(max_temp_c) or not 30 <= max_temp_c <= 90):
        raise BudgetHold("invalid wallclock/thermal guard thresholds")
    if elapsed_s >= timeout_s:
        return "WALLCLOCK_EXCEEDED"
    if (not isinstance(gpu_temp_c, (int, float))
            or not math.isfinite(gpu_temp_c) or gpu_temp_c < 0 or gpu_temp_c > 120):
        raise BudgetHold("GPU thermal telemetry unavailable or invalid")
    return "THERMAL_LIMIT_REACHED" if gpu_temp_c >= max_temp_c else "PASS"


def nvidia_temperature(gpu_index: int = 0) -> int | None:
    """Observe one GPU temperature; missing/ambiguous readings never pass."""
    if not isinstance(gpu_index, int) or not 0 <= gpu_index <= 16:
        return None
    try:
        report = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        rows = report.stdout.strip().splitlines()
        if report.returncode != 0 or len(rows) != 1 or not rows[0].strip().isdigit():
            return None
        return int(rows[0].strip())
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, body: dict) -> None:
    # Exclusive creation ensures a result cannot replace an earlier receipt.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _stop_owned_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def guarded_child(command: list[str], receipt_dir: Path, *, timeout_s: float,
                  max_temp_c: float, poll_s: float = 2.0,
                  temperature: Callable[[], float | None] = nvidia_temperature) -> dict:
    """Launch only one owned process, then monitor until exit or hard budget.

    This is deliberately separate from `train_neural.py` so it cannot change
    model/trainer semantics or silently convert incomplete runs to receipts.
    """
    if (not isinstance(command, list) or not command or
            any(not isinstance(arg, str) or not arg for arg in command)):
        raise BudgetHold("child command malformed")
    if (not isinstance(poll_s, (int, float)) or not math.isfinite(poll_s)
            or not 0.005 <= poll_s <= 30):
        raise BudgetHold("invalid watchdog polling interval")
    receipt_dir = Path(receipt_dir)
    if (not receipt_dir.is_absolute() or not receipt_dir.parent.is_dir()
            or receipt_dir.exists()
            or any(path.is_symlink() for path in (receipt_dir, *receipt_dir.parents))):
        raise BudgetHold("receipt directory missing parent, symlinked, or already occupied")
    # Read one sample BEFORE spawning anything, validating every threshold.
    try:
        initial_temp = temperature()
        initial = assess_budget(0, timeout_s, initial_temp, max_temp_c)
    except Exception:
        initial = "SENSOR_UNAVAILABLE"
    if initial != "PASS":
        return {"schema": "TRAININGDAY_R38_GUARD_V1", "status": "HOLD_NO_LAUNCH",
                "abort_reason": initial, "attempt_consumed": False}

    receipt_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    launched = {"schema": "TRAININGDAY_R38_GUARD_V1", "status": "PROCESS_LAUNCHED",
                "started_at_utc": _utc(), "attempt_consumed": True,
                "automatic_retry": False, "max_seconds": timeout_s,
                "max_gpu_temp_c": max_temp_c, "poll_seconds": poll_s,
                "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                "initial_gpu_temp_c": initial_temp}
    _write_new(receipt_dir / "launch.json", launched)
    child = None
    abort_reason = None
    exit_code = None
    interruption = None
    with (receipt_dir / "stdout.log").open("xb") as stdout, \
            (receipt_dir / "stderr.log").open("xb") as stderr:
        try:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    temp = temperature()
                    reason = assess_budget(elapsed, timeout_s, temp, max_temp_c)
                except Exception:
                    temp, reason = None, "SENSOR_UNAVAILABLE"
                with (receipt_dir / "watchdog.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"elapsed_seconds": round(elapsed, 3),
                                          "gpu_temp_c": temp, "decision": reason,
                                          "child_pid": child.pid}) + "\n")
                if reason != "PASS":
                    abort_reason = reason
                    break
                # A timed wait wakes at the child's actual exit instead of
                # sleeping past the deadline and mistaking late exit for PASS.
                remaining = timeout_s - (time.monotonic() - started)
                try:
                    child.wait(timeout=max(0.001, min(poll_s, remaining)))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as exc:
            # Preserve an incident even when a signal or process-spawn error
            # interrupted the watchdog. Stop only the child this guard owns.
            abort_reason = abort_reason or "MONITOR_INTERRUPTED"
            interruption = exc
        finally:
            if child is not None:
                if abort_reason:
                    _stop_owned_child(child)
                exit_code = child.wait()
    # The child may exit between sampled observations; never classify an
    # already-exited process as on-budget without comparing its exit time.
    if abort_reason is None and time.monotonic() - started >= timeout_s:
        abort_reason = "WALLCLOCK_EXCEEDED"
    outcome = {"schema": "TRAININGDAY_R38_GUARD_V1",
               "status": ("INCOMPLETE_NO_CHILD" if child is None else
                          "INCOMPLETE_ABORTED" if abort_reason else
                          "INCOMPLETE_PROCESS_TREE_UNVERIFIED" if exit_code == 0
                          else "INCOMPLETE_CHILD_NONZERO"),
               "attempt_consumed": True, "automatic_retry": False,
               "ended_at_utc": _utc(), "elapsed_seconds": round(time.monotonic()-started, 3),
               "child_returncode": exit_code, "abort_reason": abort_reason,
               "physical_training_verified": False, "adapter_reloaded": False,
               # Direct-child termination does not prove descendant cleanup.
               # A real Windows Job Object / verified process-tree contract
               # is required before this is an enforceable host-run guard.
               "guard_enforcement_qualified": False,
               "process_tree_containment": "UNVERIFIED_DIRECT_CHILD_ONLY"}
    _write_new(receipt_dir / "result.json", outcome)
    if interruption is not None:
        raise interruption
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--max-gpu-temp-c", type=float, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="R15 train_neural arguments after --; never an arbitrary executable")
    args = parser.parse_args()
    if args.gpu_index != 0:
        parser.error("R15 train_neural hardcodes CUDA device 0; monitor GPU index must be 0")
    forwarded = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if not forwarded or "--preflight-only" in forwarded:
        parser.error("R15 training arguments required (preflight-only is not a run)")
    cmd = [sys.executable, "-m", "trainingday.train_neural", *forwarded]
    result = guarded_child(
        cmd, args.receipt_dir,
        timeout_s=args.max_seconds, max_temp_c=args.max_gpu_temp_c,
        poll_s=args.poll_seconds, temperature=lambda: nvidia_temperature(args.gpu_index),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    # All source-only outcomes are incomplete until process-tree containment
    # is independently qualified; exit-zero cannot mean launch approval.
    raise SystemExit(3)


if __name__ == "__main__":
    main()
