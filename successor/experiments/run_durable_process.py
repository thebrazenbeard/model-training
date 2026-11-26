from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class DurableProcessHold(RuntimeError):
    """Fail-closed refusal for the durable process harness."""


_PROGRESS = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_SPEC_REL = (
    "successor/experiments/"
    "V10R3R2_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261004_V1.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _lf_normalized_sha256(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DurableProcessHold(f"spec_path unreadable:{path}") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _git_head(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise DurableProcessHold(
            f"repo_head unavailable:{result.returncode}:{result.stderr[-300:]}"
        )
    head = result.stdout.strip().lower()
    if _HEX40.fullmatch(head) is None:
        raise DurableProcessHold(f"repo_head invalid:{head}")
    return head


def _child_spec(cwd: Path, command: Sequence[str]) -> tuple[Path, str]:
    positions = [index for index, value in enumerate(command) if value == "--spec"]
    if len(positions) != 1:
        raise DurableProcessHold(
            f"spec_path child argv requires exactly one --spec:{len(positions)}"
        )
    index = positions[0]
    if index + 1 >= len(command):
        raise DurableProcessHold("spec_path child argv missing --spec value")

    raw = Path(command[index + 1])
    resolved = raw.resolve() if raw.is_absolute() else (cwd / raw).resolve()
    try:
        relative = resolved.relative_to(cwd.resolve()).as_posix()
    except ValueError as exc:
        raise DurableProcessHold(
            f"spec_path escapes cwd:{resolved}"
        ) from exc
    if relative != _REQUIRED_SPEC_REL:
        raise DurableProcessHold(
            f"spec_path not protocol-frozen R2 subject:{relative}"
        )
    if not resolved.is_file():
        raise DurableProcessHold(f"spec_path does not exist:{relative}")
    return resolved, relative


def _validate_subject_binding(
    cwd: Path,
    command: Sequence[str],
    metadata: dict,
) -> dict[str, str]:
    repo_head = metadata["repo_head"].lower()
    spec_sha = metadata["spec_sha256"].lower()
    runtime_sha = metadata["runtime_binding_sha256"].lower()

    if _HEX40.fullmatch(repo_head) is None:
        raise DurableProcessHold(f"repo_head malformed:{repo_head}")
    if _HEX64.fullmatch(spec_sha) is None:
        raise DurableProcessHold(f"spec_sha256 malformed:{spec_sha}")
    if _HEX64.fullmatch(runtime_sha) is None:
        raise DurableProcessHold(
            f"runtime_binding_sha256 malformed:{runtime_sha}"
        )

    actual_head = _git_head(cwd)
    if repo_head != actual_head:
        raise DurableProcessHold(
            f"repo_head mismatch:{repo_head}!={actual_head}"
        )

    spec_path, spec_relative = _child_spec(cwd, command)
    if metadata["spec_path"] != spec_relative:
        raise DurableProcessHold(
            "spec_path mismatch:"
            f"{metadata['spec_path']}!={spec_relative}"
        )

    actual_spec_sha = _lf_normalized_sha256(spec_path)
    if spec_sha != actual_spec_sha:
        raise DurableProcessHold(
            f"spec_sha256 mismatch:{spec_sha}!={actual_spec_sha}"
        )

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DurableProcessHold(f"spec_path json invalid:{spec_relative}") from exc
    runtime_binding = (
        spec.get("source_subject", {}).get("runtime_binding_sha256")
        if isinstance(spec, dict)
        else None
    )
    if not isinstance(runtime_binding, str):
        raise DurableProcessHold(
            f"runtime_binding_sha256 missing from spec:{spec_relative}"
        )
    actual_runtime_sha = runtime_binding.lower()
    if _HEX64.fullmatch(actual_runtime_sha) is None:
        raise DurableProcessHold(
            f"runtime_binding_sha256 invalid in spec:{actual_runtime_sha}"
        )
    if runtime_sha != actual_runtime_sha:
        raise DurableProcessHold(
            "runtime_binding_sha256 mismatch:"
            f"{runtime_sha}!={actual_runtime_sha}"
        )

    return {
        "repo_head": actual_head,
        "spec_path": spec_relative,
        "spec_sha256": actual_spec_sha,
        "runtime_binding_sha256": actual_runtime_sha,
    }


def _gpu_probe() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}:{exc}"}

    if result.returncode != 0:
        return {
            "status": "ERROR",
            "exit_code": result.returncode,
            "stderr": result.stderr[-500:],
        }

    line = result.stdout.strip().splitlines()
    if not line:
        return {"status": "EMPTY"}
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) != 3:
        return {"status": "UNPARSEABLE", "raw": line[0]}
    try:
        return {
            "status": "OK",
            "utilization_percent": int(parts[0]),
            "memory_used_mib": int(parts[1]),
            "memory_total_mib": int(parts[2]),
        }
    except ValueError:
        return {"status": "UNPARSEABLE", "raw": line[0]}


def _latest_optimizer_step(
    paths: Sequence[Path],
    expected_optimizer_steps: int | None,
) -> int | None:
    if expected_optimizer_steps is None:
        return None
    observed: list[int] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            text = path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
        observed.extend(
            int(step)
            for step, total in _PROGRESS.findall(text)
            if int(total) == expected_optimizer_steps
            and 0 <= int(step) <= expected_optimizer_steps
        )
    return max(observed) if observed else None


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--watch-seconds", type=float, default=30.0)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--expected-optimizer-steps", type=int)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise DurableProcessHold("child command is required")
    if args.watch_seconds <= 0:
        raise DurableProcessHold("watch seconds must be positive")
    if (
        args.expected_optimizer_steps is not None
        and args.expected_optimizer_steps <= 0
    ):
        raise DurableProcessHold("expected optimizer steps must be positive")
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise DurableProcessHold("metadata json is invalid") from exc
    if not isinstance(metadata, dict):
        raise DurableProcessHold("metadata json must be an object")
    args.metadata = metadata
    return args


def run_durable_process(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    log_dir = args.log_dir
    if log_dir.exists():
        raise DurableProcessHold(f"log namespace already exists:{log_dir}")
    required_metadata = (
        "repo_head",
        "spec_path",
        "spec_sha256",
        "runtime_binding_sha256",
    )
    missing = [
        key
        for key in required_metadata
        if not isinstance(args.metadata.get(key), str)
        or not args.metadata[key].strip()
    ]
    if missing:
        raise DurableProcessHold(
            "required metadata missing or invalid:" + ",".join(missing)
        )

    cwd = args.cwd.resolve() if args.cwd is not None else Path.cwd().resolve()
    _validate_subject_binding(cwd, args.command, args.metadata)
    log_dir.mkdir(parents=True, exist_ok=False)

    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    watchdog_path = log_dir / "watchdog.jsonl"
    launch_path = log_dir / "launch.json"
    final_path = log_dir / "final.json"

    started_wall = _utc_now()
    started_perf = time.perf_counter()

    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        child = subprocess.Popen(
            list(args.command),
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
        _write_json(
            launch_path,
            {
                "schema": "DURABLE_PROCESS_LAUNCH_V2",
                "status": "STARTED",
                "started_at_utc": started_wall,
                "pid": child.pid,
                "cwd": str(cwd),
                "command": list(args.command),
                "metadata": args.metadata,
                "expected_optimizer_steps": args.expected_optimizer_steps,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "watchdog_path": str(watchdog_path),
            },
        )

        with watchdog_path.open("a", encoding="utf-8", newline="\n") as watchdog:
            while True:
                returncode = child.poll()
                row = {
                    "schema": "DURABLE_PROCESS_WATCHDOG_V2",
                    "observed_at_utc": _utc_now(),
                    "pid": child.pid,
                    "returncode": returncode,
                    "latest_optimizer_step": _latest_optimizer_step(
                        (stdout_path, stderr_path),
                        args.expected_optimizer_steps,
                    ),
                    "stdout_bytes": stdout_path.stat().st_size,
                    "stderr_bytes": stderr_path.stat().st_size,
                    "gpu": _gpu_probe(),
                }
                watchdog.write(json.dumps(row, sort_keys=True) + "\n")
                watchdog.flush()
                os.fsync(watchdog.fileno())
                if returncode is not None:
                    break
                time.sleep(args.watch_seconds)

    elapsed = time.perf_counter() - started_perf
    returncode = int(child.returncode)
    final = {
        "schema": "DURABLE_PROCESS_FINAL_V2",
        "status": "EXITED",
        "started_at_utc": started_wall,
        "ended_at_utc": _utc_now(),
        "elapsed_seconds": round(elapsed, 6),
        "pid": child.pid,
        "exit_code": returncode,
        "signal": -returncode if returncode < 0 else None,
        "latest_optimizer_step": _latest_optimizer_step(
            (stdout_path, stderr_path),
            args.expected_optimizer_steps,
        ),
        "metadata": args.metadata,
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "watchdog_path": str(watchdog_path),
    }
    _write_json(final_path, final)
    return returncode


def main() -> int:
    return run_durable_process()


if __name__ == "__main__":
    raise SystemExit(main())
