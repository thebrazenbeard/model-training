"""Read-only local NVIDIA GPU telemetry check for Lappy's Ubuntu WSL2.

This probe does not import CUDA/PyTorch, allocate a GPU context, install
packages, access training data, authorize training, or start any training job.
"""

import json
from pathlib import Path
import re
import subprocess


SCHEMA = "VERA_TRAININGDAY_LAPPY_WSL2_PREFLIGHT_V1"
_GPU_ROW = re.compile(r"\A([^\r\n,]{1,200}),\s*([1-9][0-9]{0,5})\s+MiB,\s*([0-9]+(?:\.[0-9]+){1,3})\Z")
_WSL_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


def is_wsl2(proc_version):
    """Check kernel identity, not merely the presence of a WSL environment var."""
    return (isinstance(proc_version, str)
            and "microsoft" in proc_version.lower()
            and "wsl2" in proc_version.lower())


def parse_gpu_row(stdout):
    """Parse one NVIDIA CSV row; reject ambiguous or malformed results."""
    if not isinstance(stdout, str):
        raise ValueError("invalid GPU telemetry")
    match = _GPU_ROW.fullmatch(stdout.strip())
    if match is None:
        raise ValueError("invalid GPU telemetry")
    gpu_name, memory_mib, driver_version = match.groups()
    memory = int(memory_mib)
    if memory > 262144:
        raise ValueError("invalid GPU memory")
    return {
        "gpu_name": gpu_name.strip(),
        "vram_mib": memory,
        "driver_version": driver_version,
    }


def inspect_wsl_nvidia_gpu(proc_version):
    """Return non-authorizing NVIDIA telemetry without touching a CUDA context."""
    result = {
        "schema": SCHEMA,
        "status": "NOT_WSL2_TRAINING_HOLD",
        "wsl2_detected": is_wsl2(proc_version),
        "nvidia_gpu_visible": False,
        "data_rights_granted": False,
        "gpu_lease_granted": False,
        "training_authorized": False,
    }
    if not result["wsl2_detected"]:
        return result
    result["status"] = "NVIDIA_GPU_NOT_VERIFIED_TRAINING_HOLD"
    if not _WSL_SMI.exists():
        return result
    try:
        completed = subprocess.run(
            [str(_WSL_SMI), "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if completed.returncode != 0:
            return result
        info = parse_gpu_row(completed.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return result
    result.update(info)
    result["nvidia_gpu_visible"] = True
    result["status"] = "NVIDIA_GPU_VISIBLE_TRAINING_HOLD"
    return result


def main():
    try:
        version = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        version = ""
    result = inspect_wsl_nvidia_gpu(version)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    # Exit zero proves ONLY NVIDIA telemetry, never usable PyTorch CUDA.
    return 0 if result["nvidia_gpu_visible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
