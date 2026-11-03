"""Bounded CUDA and Linux RAM telemetry for a future authorized lowmem run.

This is *measurement plumbing*, not a host lease, weight admission, or fit
qualification. Importing this module does not initialize torch or touch CUDA.
The caller must own the device and pass an actual CUDA runtime after the
independent physical launch authority check has succeeded.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class MemoryTelemetryError(ValueError):
    """A missing or incoherent physical memory observation."""


PHASES = ("before_load", "after_load", "after_step")


def parse_linux_kib(text: str, label: str) -> int:
    """Read an exact positive Linux /proc field; tolerate other fields."""
    if type(text) is not str or type(label) is not str:
        raise MemoryTelemetryError("invalid /proc input")
    values = []
    for line in text.splitlines():
        name, separator, rest = line.partition(":")
        if separator and name == label:
            fields = rest.strip().split()
            if (len(fields) != 2 or fields[1] != "kB"
                    or not fields[0].isascii() or not fields[0].isdecimal()):
                raise MemoryTelemetryError(f"malformed {label} field")
            value = int(fields[0])
            if value <= 0:
                raise MemoryTelemetryError(f"empty {label} field")
            values.append(value * 1024)
    if len(values) != 1:
        raise MemoryTelemetryError(f"missing or duplicate {label} field")
    return values[0]


def read_linux_host_memory() -> dict[str, int]:
    """Read bounded live WSL2 /proc data; no privileged or network access."""
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
        status = Path("/proc/self/status").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise MemoryTelemetryError("Linux host memory telemetry unavailable") from exc
    if len(meminfo) > 65536 or len(status) > 65536:
        raise MemoryTelemetryError("oversized Linux /proc telemetry")
    return {
        "host_mem_available_bytes": parse_linux_kib(meminfo, "MemAvailable"),
        "process_resident_bytes": parse_linux_kib(status, "VmRSS"),
    }


def capture_memory_phase(torch, *, phase: str) -> dict:
    """Capture device-wide free memory plus process-local PyTorch counters.

    `torch` is dependency-injected so the source guard is CPU-testable. The
    physical caller must synchronize before calling this method; the function
    synchronizes again to avoid asynchronous kernel memory deltas.
    """
    if phase not in PHASES:
        raise MemoryTelemetryError("unrecognized lowmem phase")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    result = {
        "phase": phase,
        "device_free_bytes": free,
        "device_total_bytes": total,
        "process_cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "process_cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "process_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result.update(read_linux_host_memory())
    validate_memory_phase(result)
    return result


def validate_memory_phase(record: Mapping[str, object]) -> None:
    if type(record) is not dict or record.get("phase") not in PHASES:
        raise MemoryTelemetryError("unrecognized memory phase record")
    keys = ("device_free_bytes", "device_total_bytes",
            "process_cuda_allocated_bytes", "process_cuda_reserved_bytes",
            "process_cuda_peak_allocated_bytes", "process_cuda_peak_reserved_bytes",
            "host_mem_available_bytes", "process_resident_bytes")
    if any(type(record.get(key)) is not int or record[key] < 0 for key in keys):
        raise MemoryTelemetryError("missing or negative numeric memory counter")
    total = record["device_total_bytes"]
    if (total == 0 or record["device_free_bytes"] > total
            or record["host_mem_available_bytes"] == 0
            or record["process_resident_bytes"] == 0
            or record["process_cuda_allocated_bytes"] > record["process_cuda_reserved_bytes"]
            or record["process_cuda_reserved_bytes"] > total
            or record["process_cuda_peak_allocated_bytes"]
            > record["process_cuda_peak_reserved_bytes"]
            or record["process_cuda_peak_reserved_bytes"] > total):
        raise MemoryTelemetryError("incoherent device/host memory counters")


def summarize_memory_timeline(records: list[dict]) -> dict:
    """Reject gaps and incompatible GPU totals; never announce model fit.

    Free device memory is *device-wide at each sample*; PyTorch counters are
    process-local. They are not added together as though disjoint footprints.
    No sample bounds unsampled transient peaks or another process's activity.
    """
    if type(records) is not list or len(records) != 3:
        raise MemoryTelemetryError("all three phases are required")
    for record in records:
        validate_memory_phase(record)
    if ([record["phase"] for record in records] != list(PHASES)
            or len({record["device_total_bytes"] for record in records}) != 1):
        raise MemoryTelemetryError("phase ordering or GPU identity changed")
    total = records[0]["device_total_bytes"]
    return {
        "status": "MEMORY_TELEMETRY_FIELDS_VALIDATED_FIT_UNQUALIFIED",
        "device_total_bytes": total,
        "minimum_sampled_device_free_bytes": min(
            record["device_free_bytes"] for record in records),
        "minimum_sampled_host_mem_available_bytes": min(
            record["host_mem_available_bytes"] for record in records),
        "maximum_sampled_process_resident_bytes": max(
            record["process_resident_bytes"] for record in records),
        "maximum_process_cuda_peak_reserved_bytes": max(
            record["process_cuda_peak_reserved_bytes"] for record in records),
        "measurement_scope": "DEVICE_WIDE_SNAPSHOTS_AND_PROCESS_LOCAL_CUDA_PEAKS",
        "authority_or_weight_custody_proven": False,
        "four_gib_training_fit": "UNKNOWN",
        "training_authorized": False,
        "phases": records,
    }
