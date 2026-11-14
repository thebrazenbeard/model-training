from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
from pathlib import Path
from typing import Any


GIB = 1024 ** 3


class HostResourceHold(RuntimeError):
    """Fail-closed refusal when the host is unsafe for a fresh GPU trainer."""


class MEMORYSTATUSEX(ctypes.Structure):
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


def read_windows_memory_status() -> dict[str, float]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise HostResourceHold("GlobalMemoryStatusEx failed")
    commit_limit = float(status.ullTotalPageFile)
    commit_headroom = float(status.ullAvailPageFile)
    committed = max(0.0, commit_limit - commit_headroom)
    committed_percent = 0.0 if commit_limit <= 0 else committed / commit_limit * 100.0
    return {
        "available_physical_gib": round(status.ullAvailPhys / GIB, 3),
        "total_physical_gib": round(status.ullTotalPhys / GIB, 3),
        "commit_headroom_gib": round(commit_headroom / GIB, 3),
        "commit_limit_gib": round(commit_limit / GIB, 3),
        "committed_percent": round(committed_percent, 3),
    }


def read_gpu_status() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise HostResourceHold(
            f"nvidia-smi failed:{result.returncode}:{result.stderr[-300:]}"
        )
    line = result.stdout.strip().splitlines()
    if not line:
        raise HostResourceHold("nvidia-smi returned no gpu rows")
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) != 3:
        raise HostResourceHold(f"nvidia-smi output unparseable:{line[0]}")
    try:
        return {
            "gpu_utilization_percent": int(parts[0]),
            "gpu_memory_used_mib": int(parts[1]),
            "gpu_memory_total_mib": int(parts[2]),
        }
    except ValueError as exc:
        raise HostResourceHold(f"nvidia-smi values unparseable:{line[0]}") from exc


def read_competing_trainers() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise HostResourceHold(
            f"gpu compute process scan failed:{result.returncode}:"
            f"{result.stderr[-300:]}"
        )
    payload = result.stdout.strip()
    if not payload:
        return []

    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise HostResourceHold(
                f"gpu compute process output unparseable:{line}"
            )
        try:
            pid = int(parts[0])
        except ValueError as exc:
            raise HostResourceHold(
                f"gpu compute process pid unparseable:{line}"
            ) from exc
        memory_field = parts[2].strip()
        if memory_field.upper() in {"[N/A]", "N/A", "NA"}:
            used_memory_mib = None
        else:
            try:
                used_memory_mib = int(memory_field)
            except ValueError as exc:
                raise HostResourceHold(
                    f"gpu compute process memory unparseable:{line}"
                ) from exc
        rows.append(
            {
                "pid": pid,
                "process_name": parts[1],
                "used_memory_mib": used_memory_mib,
            }
        )
    return rows


def read_live_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics.update(read_windows_memory_status())
    metrics.update(read_gpu_status())
    metrics["competing_trainers"] = read_competing_trainers()
    return metrics


def validate_host_resources(
    metrics: dict[str, Any],
    *,
    min_available_physical_gib: float,
    min_commit_headroom_gib: float,
) -> dict[str, Any]:
    available = float(metrics["available_physical_gib"])
    commit_headroom = float(metrics["commit_headroom_gib"])
    gpu_used = int(metrics["gpu_memory_used_mib"])
    competing = list(metrics.get("competing_trainers") or [])

    if available < min_available_physical_gib:
        raise HostResourceHold(
            "available physical memory below minimum:"
            f"{available}<{min_available_physical_gib}"
        )
    if commit_headroom < min_commit_headroom_gib:
        raise HostResourceHold(
            "commit headroom below minimum:"
            f"{commit_headroom}<{min_commit_headroom_gib}"
        )
    if gpu_used != 0:
        raise HostResourceHold(f"gpu memory is not free:{gpu_used} MiB")
    if competing:
        raise HostResourceHold(
            "competing trainer detected:" + json.dumps(competing, sort_keys=True)
        )

    return {
        **metrics,
        "status": "PASS",
        "min_available_physical_gib": min_available_physical_gib,
        "min_commit_headroom_gib": min_commit_headroom_gib,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-available-physical-gib", type=float, required=True)
    parser.add_argument("--min-commit-headroom-gib", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        metrics = read_live_metrics()
        result = validate_host_resources(
            metrics,
            min_available_physical_gib=args.min_available_physical_gib,
            min_commit_headroom_gib=args.min_commit_headroom_gib,
        )
    except HostResourceHold as exc:
        metrics = locals().get("metrics", {})
        result = {
            **metrics,
            "status": "HOLD",
            "reason": str(exc),
            "min_available_physical_gib": args.min_available_physical_gib,
            "min_commit_headroom_gib": args.min_commit_headroom_gib,
        }
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8", newline="\n")
        print(payload, end="")
        return 2

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
