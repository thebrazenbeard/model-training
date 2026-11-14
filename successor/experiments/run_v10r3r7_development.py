from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PRELAUNCH_MIN_PHYSICAL_GIB = 8.0
PRELAUNCH_MIN_COMMIT_GIB = 25.0
INRUN_MIN_COMMIT_GIB = 8.0
GPU_TEMP_ABORT_C = 88


class R7Hold(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def verify_corpus_file(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise R7Hold(f"training corpus missing:{path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha256:
        raise R7Hold(
            f"training corpus sha256 mismatch:{actual_sha}!={expected_sha256}"
        )
    rows = _row_count(path)
    if rows != int(expected_rows):
        raise R7Hold(f"training corpus row count mismatch:{rows}!={expected_rows}")
    return {
        "status": "EXACT_CORPUS_VERIFIED",
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "rows": rows,
        "bytes": path.stat().st_size,
    }


def require_fresh_output_namespace(path: Path | str) -> None:
    path = Path(path)
    if path.exists():
        raise R7Hold(f"output namespace already exists:{path}")


def evaluate_resource_sample(
    *,
    available_physical_gib: float,
    commit_headroom_gib: float,
    gpu_utilization_percent: int,
    gpu_memory_used_mib: int,
    compute_apps: list[str],
    gpu_temperature_c: int,
    prelaunch: bool,
) -> dict:
    values = (
        float(available_physical_gib),
        float(commit_headroom_gib),
        float(gpu_utilization_percent),
        float(gpu_memory_used_mib),
        float(gpu_temperature_c),
    )
    if not all(math.isfinite(value) for value in values):
        return {"status": "HOLD", "reason": "non-finite resource observation"}

    if int(gpu_temperature_c) >= GPU_TEMP_ABORT_C:
        return {
            "status": "HOLD",
            "reason": f"GPU temperature at/above {GPU_TEMP_ABORT_C}C",
        }

    if prelaunch:
        if float(available_physical_gib) < PRELAUNCH_MIN_PHYSICAL_GIB:
            return {
                "status": "HOLD",
                "reason": (
                    "available physical memory below prelaunch minimum:"
                    f"{available_physical_gib}<{PRELAUNCH_MIN_PHYSICAL_GIB}"
                ),
            }
        if float(commit_headroom_gib) < PRELAUNCH_MIN_COMMIT_GIB:
            return {
                "status": "HOLD",
                "reason": (
                    "commit headroom below prelaunch minimum:"
                    f"{commit_headroom_gib}<{PRELAUNCH_MIN_COMMIT_GIB}"
                ),
            }
        if int(gpu_utilization_percent) != 0:
            return {
                "status": "HOLD",
                "reason": f"GPU utilization is not idle:{gpu_utilization_percent}%",
            }
        if compute_apps:
            return {
                "status": "HOLD",
                "reason": f"competing compute apps present:{compute_apps}",
            }
    elif float(commit_headroom_gib) < INRUN_MIN_COMMIT_GIB:
        return {
            "status": "HOLD",
            "reason": (
                "commit headroom below in-run safety floor:"
                f"{commit_headroom_gib}<{INRUN_MIN_COMMIT_GIB}"
            ),
        }

    return {
        "status": "PASS",
        "available_physical_gib": float(available_physical_gib),
        "commit_headroom_gib": float(commit_headroom_gib),
        "gpu_utilization_percent": int(gpu_utilization_percent),
        "gpu_memory_used_mib": int(gpu_memory_used_mib),
        "compute_apps": list(compute_apps),
        "gpu_temperature_c": int(gpu_temperature_c),
        "wddm_memory_is_informational": True,
        "prelaunch": bool(prelaunch),
    }


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def observe_resources() -> dict:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise R7Hold("GlobalMemoryStatusEx failed")

    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(gpu) != 1:
        raise R7Hold("unexpected nvidia-smi GPU observation")
    used, util, temp = [part.strip() for part in gpu[0].split(",")]

    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    compute_apps = [line.strip() for line in apps.stdout.splitlines() if line.strip()]

    return {
        "available_physical_gib": status.ullAvailPhys / (1024**3),
        "commit_headroom_gib": status.ullAvailPageFile / (1024**3),
        "gpu_utilization_percent": int(util),
        "gpu_memory_used_mib": int(used),
        "compute_apps": compute_apps,
        "gpu_temperature_c": int(temp),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise R7Hold(f"git HEAD unavailable:{result.stderr[-300:]}")
    return result.stdout.strip()


def _tail(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R7Hold(f"JSON unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise R7Hold(f"JSON object required:{path}")
    return value


def build_training_command(
    *, repo_root: Path, train_jsonl: Path, spec_path: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "successor.experiments.train_v10r2_dev",
        "--repo-root",
        str(repo_root),
        "--train-jsonl",
        str(train_jsonl),
        "--spec",
        str(spec_path),
    ]


def _validate_spec(
    root: Path,
    spec_path: Path,
    *,
    expected_spec_sha256: str,
    expected_train_sha256: str,
    expected_train_rows: int,
) -> dict:
    if not spec_path.is_file():
        raise R7Hold(f"execution spec missing:{spec_path}")
    actual_sha = _sha256_file(spec_path)
    if actual_sha != expected_spec_sha256:
        raise R7Hold(
            f"execution spec sha256 mismatch:{actual_sha}!={expected_spec_sha256}"
        )
    spec = _load_json(spec_path)
    if spec.get("schema") != "V10R2_QWEN35_DEV_TRAINING_SPEC_V1":
        raise R7Hold("unexpected execution spec schema")
    if spec.get("status") != "AUTHORIZED_LOCAL_DEVELOPMENT_TRAINING":
        raise R7Hold("execution spec is not authorized local development training")
    source = spec.get("source_subject", {})
    if source.get("train_sha256") != expected_train_sha256:
        raise R7Hold("execution spec train sha mismatch")
    if source.get("train_rows") != int(expected_train_rows):
        raise R7Hold("execution spec train rows mismatch")
    trainer = spec.get("trainer", {})
    if trainer.get("max_optimizer_steps") != 20:
        raise R7Hold("R7 execution spec must request exactly 20 optimizer steps")
    if trainer.get("optimizer") != "adamw_bnb_8bit":
        raise R7Hold("R7 execution spec optimizer mismatch")
    output = spec.get("output", {})
    namespace = output.get("namespace")
    allowed_namespace_tags = ("v10r3r7", "v10r3r8")
    if (
        not isinstance(namespace, str)
        or not any(tag in namespace.casefold() for tag in allowed_namespace_tags)
    ):
        raise R7Hold("R7/R8 fresh output namespace missing")
    require_fresh_output_namespace(namespace)
    return {"sha256": actual_sha, "spec": spec, "output_namespace": namespace}


def execute(
    *,
    repo_root: Path,
    train_jsonl: Path,
    spec_path: Path,
    expected_head: str,
    expected_spec_sha256: str,
    expected_train_sha256: str,
    expected_train_rows: int,
    log_dir: Path,
    receipt_dir: Path,
    watchdog_seconds: float = 10.0,
    run_id: str = "V10R3R7",
) -> int:
    if run_id not in {"V10R3R7", "V10R3R8"}:
        raise R7Hold(f"unsupported run id:{run_id}")
    if _git_head(repo_root) != expected_head:
        raise R7Hold(f"repo HEAD mismatch:{_git_head(repo_root)}!={expected_head}")

    spec_info = _validate_spec(
        repo_root,
        spec_path,
        expected_spec_sha256=expected_spec_sha256,
        expected_train_sha256=expected_train_sha256,
        expected_train_rows=expected_train_rows,
    )
    corpus = verify_corpus_file(
        train_jsonl,
        expected_sha256=expected_train_sha256,
        expected_rows=expected_train_rows,
    )

    launch_path = receipt_dir / f"{run_id}_DEVELOPMENT_LAUNCH_20261007_V1.json"
    watchdog_path = receipt_dir / f"{run_id}_DEVELOPMENT_WATCHDOG_20261007_V1.jsonl"
    result_path = receipt_dir / f"{run_id}_DEVELOPMENT_RESULT_20261007_V1.json"
    incident_path = receipt_dir / f"{run_id}_DEVELOPMENT_INCIDENT_20261007_V1.json"
    for path in (launch_path, watchdog_path, result_path, incident_path):
        if path.exists():
            raise R7Hold(f"refusing to reuse {run_id} execution artifact:{path}")
    if log_dir.exists():
        raise R7Hold(f"refusing to reuse {run_id} log namespace:{log_dir}")

    observed = observe_resources()
    gate = evaluate_resource_sample(**observed, prelaunch=True)
    if gate["status"] != "PASS":
        print(
            json.dumps(
                {
                    "schema": f"{run_id}_PRELAUNCH_HOLD_V1",
                    "status": "HOLD_NO_ATTEMPT_CONSUMED",
                    "reason": gate["reason"],
                    "observation": observed,
                },
                sort_keys=True,
            )
        )
        return 4

    log_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    launch = {
        "schema": f"{run_id}_DEVELOPMENT_LAUNCH_V1",
        "status": "LAUNCHED",
        "started_at_utc": _utc_now(),
        "attempt_consumed": True,
        "automatic_retry": False,
        "repo_head": expected_head,
        "spec_path": str(spec_path.relative_to(repo_root)).replace("\\", "/"),
        "spec_sha256": expected_spec_sha256,
        "corpus": corpus,
        "prelaunch_observation": observed,
        "prelaunch_gate": gate,
        "in_run_min_commit_headroom_gib": INRUN_MIN_COMMIT_GIB,
        "gpu_temperature_abort_c": GPU_TEMP_ABORT_C,
        "output_namespace": spec_info["output_namespace"],
        "log_dir": str(log_dir),
    }
    _write_json(launch_path, launch)

    command = build_training_command(
        repo_root=repo_root,
        train_jsonl=train_jsonl,
        spec_path=spec_path,
    )
    started = time.monotonic()
    abort_reason = None

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=stdout,
            stderr=stderr,
        )
        while child.poll() is None:
            sample = observe_resources()
            decision = evaluate_resource_sample(**sample, prelaunch=False)
            _append_jsonl(
                watchdog_path,
                {
                    "observed_at_utc": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "child_pid": child.pid,
                    "resource": sample,
                    "decision": decision,
                },
            )
            if decision["status"] != "PASS":
                abort_reason = decision["reason"]
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
                break
            time.sleep(max(1.0, float(watchdog_seconds)))

        returncode = child.wait()

    elapsed = time.monotonic() - started
    if abort_reason is not None or returncode != 0:
        incident = {
            "schema": f"{run_id}_DEVELOPMENT_INCIDENT_V1",
            "status": "FAILED_INCOMPLETE_ONE_ATTEMPT_EXHAUSTED",
            "ended_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "attempt_consumed": True,
            "automatic_retry": False,
            "repo_head": expected_head,
            "spec_sha256": expected_spec_sha256,
            "child_returncode": returncode,
            "abort_reason": abort_reason,
            "stdout_tail": _tail(stdout_path),
            "stderr_tail": _tail(stderr_path),
        }
        _write_json(incident_path, incident)
        print(json.dumps(incident, indent=2, sort_keys=True), file=sys.stderr)
        return 3

    completion_path = Path(spec_info["output_namespace"]) / "DEV_TRAINING_COMPLETE.json"
    completion = _load_json(completion_path)
    if completion.get("status") != "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE":
        raise R7Hold("development completion receipt status invalid")
    if completion.get("weight_digest_changed") is not True:
        raise R7Hold("development completion receipt does not prove weight change")
    if completion.get("cumulative_optimizer_steps") != 20:
        raise R7Hold("development completion optimizer step count mismatch")

    result = {
        "schema": f"{run_id}_DEVELOPMENT_RESULT_V1",
        "status": "DEVELOPMENT_TRAINING_PASS",
        "ended_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "attempt_consumed": True,
        "automatic_retry": False,
        "repo_head": expected_head,
        "spec_sha256": expected_spec_sha256,
        "corpus_sha256": expected_train_sha256,
        "output_namespace": spec_info["output_namespace"],
        "completion_receipt_path": str(completion_path),
        "completion_receipt_sha256": _sha256_file(completion_path),
        "weight_digest_before": completion.get("weight_digest_before"),
        "weight_digest_after": completion.get("weight_digest_after"),
        "weight_digest_changed": True,
        "optimizer_steps": completion.get("cumulative_optimizer_steps"),
        "qualification_status": completion.get("qualification_status"),
        "deployment_status": completion.get("deployment_status"),
        "claim_ceiling": completion.get("claim_ceiling"),
    }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--watchdog-seconds", type=float, default=10.0)
    parser.add_argument(
        "--run-id",
        choices=("V10R3R7", "V10R3R8"),
        default="V10R3R7",
    )
    args = parser.parse_args(argv)

    try:
        return execute(
            repo_root=args.repo_root.resolve(),
            train_jsonl=args.train_jsonl.resolve(),
            spec_path=args.spec.resolve(),
            expected_head=args.expected_head,
            expected_spec_sha256=args.expected_spec_sha256,
            expected_train_sha256=args.expected_train_sha256,
            expected_train_rows=args.expected_train_rows,
            log_dir=args.log_dir.resolve(),
            receipt_dir=args.receipt_dir.resolve(),
            watchdog_seconds=args.watchdog_seconds,
            run_id=args.run_id,
        )
    except R7Hold as exc:
        print(
            json.dumps(
                {
                    "schema": f"{args.run_id}_DEVELOPMENT_RUNNER_HOLD_V1",
                    "status": "HOLD_NO_ATTEMPT_CONSUMED",
                    "reason": str(exc),
                },
                sort_keys=True,
            )
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
