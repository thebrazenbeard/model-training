"""Read-only, VN1-first Qwen3.5-4B inference preparation (never a launch grant).

No subcommand downloads weights, converts GGUF, starts inference, changes a
service, or grants host authority. A separate authenticated host owner must
approve and execute the emitted argv, and capture its output for `record`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import time

from trainingday.official_public_base import (
    METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS, OFFICIAL_ID,
    OFFICIAL_REVISION, PHYSICAL_SHARD_BYTES, SHARDS,
)


SCHEMA = "TRAININGDAY_OFFICIAL_INFERENCE_ADMISSION_W20_V1"
TELEMETRY_SCHEMA = "TRAININGDAY_TEXT_ONLY_INFERENCE_TELEMETRY_W20_V1"
STAGE_DISK_RESERVE = 24 * 1024**3
INFERENCE_MIN_AVAILABLE_RAM = 6 * 1024**3
STAGE_MIN_AVAILABLE_RAM = 1024**3
CPU_LOAD_CEILING = 0.80
MAX_LOG_BYTES = 2 * 1024**2
PROBE_PROMPT = "Respond in one sentence: What is two plus two?"
MODEL_FILENAMES = tuple(name for name, _, _ in SHARDS) + tuple(
    METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS
)
ENGINE_NAMES = {"llama.cpp": "llama-cli", "KoboldCPP": "koboldcpp"}
BUSY_NAMES = {"llama-cli", "llama-server", "koboldcpp", "koboldcpp.exe",
              "llama-cli.exe", "llama-server.exe", "main", "main.exe"}


class AdmissionError(ValueError):
    pass


def safe_absolute(path: str | Path, *, needs_file: bool = False) -> Path:
    """Reject relative/symlinked target components; do not resolve aliases."""
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise AdmissionError("absolute path without traversal required")
    for part in (value, *value.parents):
        if part.is_symlink():
            raise AdmissionError("symlink path component rejected")
    if needs_file and (not value.is_file() or value.is_symlink()):
        raise AdmissionError("regular existing file required")
    return value


def measure_file(path: Path, *, expected_bytes: int | None = None,
                 expected_sha256: str | None = None) -> dict:
    """Stream SHA256 from a no-follow FD and detect concurrent modification."""
    path = safe_absolute(path, needs_file=True)
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdmissionError("model entry is not regular")
        sha = hashlib.sha256()
        count = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            count += len(block)
            sha.update(block)
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or count != before.st_size):
            raise AdmissionError("file changed during verification")
        digest = sha.hexdigest()
        size_match = expected_bytes is None or count == expected_bytes
        hash_match = expected_sha256 is None or digest == expected_sha256
        return {"size_bytes": count, "sha256_measured": digest,
                "size_match": size_match, "sha256_match": hash_match,
                "verified": size_match and hash_match}
    except OSError as exc:
        raise AdmissionError("file hash read failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def verify_snapshot(snapshot: Path) -> dict:
    """Full two-shard byte check; metadata recorded as untrusted local digests."""
    snapshot = safe_absolute(snapshot)
    if not snapshot.is_dir():
        raise AdmissionError("snapshot directory does not exist")
    inspected = {}
    for name, size, sha in SHARDS:
        inspected[name] = measure_file(snapshot / name, expected_bytes=size,
                                      expected_sha256=sha)
    shard_verified = all(inspected[name]["verified"] for name, _, _ in SHARDS)
    for name in METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
        inspected[name] = measure_file(snapshot / name)
    return {"schema": SCHEMA, "evidence_class": "LOCAL_HASH_WITH_UPSTREAM_LFS_PINS",
            "repository": OFFICIAL_ID, "revision": OFFICIAL_REVISION,
            "snapshot": str(snapshot), "files": inspected,
            "shards_verified": shard_verified,
            "metadata_full_byte_pins_admitted": False,
            "official_checkpoint_admitted": False,
            "inference_executed": False,
            "note": "Metadata hashes are measurements, not independently trusted upstream pins."}


def _available_ram_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    elif os.name == "nt":
        try:
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in (
                                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                "ullAvailExtendedVirtual")]]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys
        except (AttributeError, OSError, ValueError):
            return None
    return None


def _cpu_fraction() -> float | None:
    if hasattr(os, "getloadavg"):
        try:
            return min(1.0, os.getloadavg()[0] / (os.cpu_count() or 1))
        except OSError:
            pass
    if os.name == "nt":
        try:
            import ctypes
            class FileTime(ctypes.Structure):
                _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]
            def read():
                idle, kernel, user = FileTime(), FileTime(), FileTime()
                if not ctypes.windll.kernel32.GetSystemTimes(
                        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                    return None
                return [v.low | (v.high << 32) for v in (idle, kernel, user)]
            start = read()
            time.sleep(0.25)
            end = read()
            if start and end:
                idle, kernel, user = (b - a for a, b in zip(start, end))
                if kernel + user > 0:
                    return max(0.0, min(1.0, 1.0 - idle / (kernel + user)))
        except (OSError, AttributeError):
            pass
    return None


def _running_model_processes() -> list[dict] | None:
    """Conservative process-name inventory: unknown stays a HOLD."""
    try:
        if sys.platform.startswith("linux"):
            found = []
            for item in Path("/proc").iterdir():
                if item.name.isdecimal():
                    try:
                        name = (item / "comm").read_text().strip().lower()
                    except (OSError, UnicodeError):
                        continue
                    if name in BUSY_NAMES:
                        found.append({"pid": int(item.name), "name": name})
            return sorted(found, key=lambda x: x["pid"])
        if os.name == "nt":
            result = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                                    capture_output=True, text=True, timeout=8,
                                    check=False)
            if result.returncode:
                return None
            found = []
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) >= 2 and row[0].lower() in BUSY_NAMES:
                    found.append({"pid": row[1], "name": row[0].lower()})
            return found
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return None
    return None


def _disk_free_bytes(path: Path) -> int | None:
    try:
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        if candidate.is_dir():
            return shutil.disk_usage(candidate).free
    except OSError:
        pass
    return None


def preflight(host: str, target: Path, *, stage: bool) -> dict:
    """Unprivileged local observations never prove named-host identity/authority."""
    target = safe_absolute(target)
    hostname = socket.gethostname()
    declared = {"VN1": "VERA-NODE-1", "VN2": "VERA-NODE-2"}[host]
    observed = {hostname.casefold(), os.environ.get("COMPUTERNAME", "").casefold()}
    identity_match = declared.casefold() in observed
    disk = _disk_free_bytes(target)
    ram = _available_ram_bytes()
    cpu = _cpu_fraction()
    busy = _running_model_processes()
    needed_disk = (PHYSICAL_SHARD_BYTES + STAGE_DISK_RESERVE) if stage else STAGE_DISK_RESERVE
    needed_ram = STAGE_MIN_AVAILABLE_RAM if stage else INFERENCE_MIN_AVAILABLE_RAM
    guard = {
        "host_identity_matches_declared": identity_match,
        "disk_headroom": disk is not None and disk >= needed_disk,
        "ram_headroom": ram is not None and ram >= needed_ram,
        "cpu_load": cpu is not None and cpu <= CPU_LOAD_CEILING,
        "no_conflicting_model_process": busy is not None and len(busy) == 0,
    }
    return {"declared_host": host, "observed_hostname": hostname,
            "observed_not_attested": True, "target": str(target),
            "free_disk_bytes": disk, "needed_disk_bytes": needed_disk,
            "available_ram_bytes": ram, "needed_ram_bytes": needed_ram,
            "cpu_fraction": cpu, "cpu_fraction_limit": CPU_LOAD_CEILING,
            "busy_model_processes": busy, "guards": guard,
            "resource_preflight_pass": all(guard.values()),
            "host_authorized": False, "may_start_workload": False}


def download_plan(snapshot: Path, host: str) -> dict:
    snapshot = safe_absolute(snapshot)
    # A non-empty target is a collision; no file is overwritten by this CLI.
    collision = snapshot.is_file() or (snapshot.is_dir() and any(snapshot.iterdir()))
    # Explicit positional filenames follow the current official HF CLI
    # contract; no wildcard, recursive whole-repository download, or force.
    argv = (["hf", "download", OFFICIAL_ID, *MODEL_FILENAMES,
             "--revision", OFFICIAL_REVISION, "--local-dir", str(snapshot)])
    observed = preflight(host, snapshot, stage=True)
    return {"schema": SCHEMA, "mode": "DRY_RUN_ONLY", "host": host,
            "repository": OFFICIAL_ID, "revision": OFFICIAL_REVISION,
            "snapshot": str(snapshot), "required_filenames": MODEL_FILENAMES,
            "shard_total_bytes": PHYSICAL_SHARD_BYTES,
            "collision": collision,
            "preflight": observed,
            "operator_command_argv_not_executed": None if collision else argv,
            "candidate_resource_ready": not collision and observed["resource_preflight_pass"],
            "next": "Host-owned authorization and distinct execution; then run verify.",
            "downloaded": False, "authorized": False}


def binary_diagnostics(binary: Path, *, probe_version: bool = False) -> dict:
    binary = safe_absolute(binary, needs_file=True)
    hash_result = measure_file(binary)
    report = {"binary": str(binary), **hash_result,
              "version_argv": [str(binary), "--version"],
              "version_probe_executed": False, "version_output": None,
              "version_probe_exit_code": None}
    if probe_version:
        try:
            result = subprocess.run([str(binary), "--version"],
                                    capture_output=True, text=True, timeout=5,
                                    check=False)
            report.update(version_probe_executed=True,
                          version_probe_exit_code=result.returncode,
                          version_output=(result.stdout + "\n" + result.stderr)[:4096])
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["version_probe_error"] = type(exc).__name__
    return report


def benchmark_plan(host: str, gguf: Path, binary: Path, engine: str,
                   *, probe_version: bool = False) -> dict:
    gguf = safe_absolute(gguf, needs_file=True)
    binary = safe_absolute(binary, needs_file=True)
    if gguf.suffix.lower() != ".gguf":
        raise AdmissionError("GGUF suffix required")
    with gguf.open("rb") as source:
        if source.read(4) != b"GGUF":
            raise AdmissionError("GGUF magic missing")
    measured_gguf = measure_file(gguf)
    if engine == "llama.cpp":
        argv = [str(binary), "-m", str(gguf), "-p", PROBE_PROMPT,
                "-n", "48", "-c", "512", "--temp", "0",
                "-ngl", "0", "-t", "4", "-tb", "4", "--no-mmproj",
                "--no-display-prompt", "--no-warmup", "--single-turn",
                "--show-timings"]
    else:
        # KoboldCPP versions differ in API and --cli support. Refuse to
        # invent a text generation CLI: review version-specific adapter.
        argv = None
    observed = preflight(host, gguf.parent, stage=False)
    return {"schema": SCHEMA, "mode": "DRY_RUN_ONLY",
            "evidence_class": "UNATTESTED_LOCAL_GGUF_AND_BINARY_INSPECTION",
            "engine": engine, "gguf": str(gguf), "gguf_measured": measured_gguf,
            "gguf_provenance_from_official_checkpoint_proven": False,
            "binary": binary_diagnostics(binary, probe_version=probe_version),
            "preflight": observed,
            "text_only": True, "sample_prompt": PROBE_PROMPT,
            "operator_command_argv_not_executed": argv,
            "version_compatibility_confirmed": False,
            "host_authorized": False, "inference_executed": False,
            "telemetry_schema": TELEMETRY_SCHEMA,
            "note": ("KoboldCPP requires a reviewed, version-specific API adapter."
                     if argv is None else "Run only under separately authenticated host owner.")}


_EVAL_PATTERN = re.compile(
    r"(?<!prompt )eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s+tokens?"
    r"\s*\([^)]*?([0-9.]+)\s+tokens? per second\)", re.IGNORECASE)
_PROMPT_EVAL_PATTERN = re.compile(
    r"prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s+tokens?"
    r"\s*\([^)]*?([0-9.]+)\s+tokens? per second\)", re.IGNORECASE)


def parse_operator_logs(stdout: str, stderr: str, *, returncode: int,
                        wall_ms: int, gguf_sha256: str) -> dict:
    """Parse operator-supplied logs; no subprocess or authenticity assertion."""
    combined = stdout + "\n" + stderr
    eval_match = _EVAL_PATTERN.search(combined)
    prompt_match = _PROMPT_EVAL_PATTERN.search(combined)
    generated = stdout.strip()
    return {"schema": TELEMETRY_SCHEMA,
            "evidence_class": "UNATTESTED_OPERATOR_SUPPLIED_LOGS",
            "official_model_identity_proven": False,
            "host_runtime_attestation_present": False,
            "text_only": True, "sample_prompt": PROBE_PROMPT,
            "gguf_sha256_operator_declared": gguf_sha256,
            "exit_code_operator_declared": returncode,
            "wall_ms_operator_declared": wall_ms,
            "generated_text_excerpt": generated[:512],
            "nonempty_stdout": bool(generated),
            "decode_claim_in_logs": returncode == 0 and bool(generated),
            "decode_verified_by_trusted_witness": False,
            "generated_tokens_timing": int(eval_match.group(1)) if eval_match else None,
            "generation_tokens_per_second": float(eval_match.group(2)) if eval_match else None,
            "prompt_tokens_timing": int(prompt_match.group(1)) if prompt_match else None,
            "prompt_tokens_per_second": float(prompt_match.group(2)) if prompt_match else None,
            "peak_ram_bytes": None, "peak_vram_bytes": None,
            "memory_telemetry_note": "No monitored OS peak-memory witness supplied."}


def _read_bounded_log(path: Path) -> str:
    path = safe_absolute(path, needs_file=True)
    if path.stat().st_size > MAX_LOG_BYTES:
        raise AdmissionError("log exceeds 2 MiB bound")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AdmissionError("log read failed") from exc


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="VN1-first dry-run download plan")
    plan.add_argument("--host", choices=("VN1", "VN2"), default="VN1")
    plan.add_argument("--snapshot", required=True)
    verify = commands.add_parser("verify", help="offline full shard SHA256+size")
    verify.add_argument("--snapshot", required=True)
    benchmark = commands.add_parser("benchmark-plan", help="dry-run text-only GGUF probe")
    benchmark.add_argument("--host", choices=("VN1", "VN2"), default="VN1")
    benchmark.add_argument("--gguf", required=True)
    benchmark.add_argument("--binary", required=True)
    benchmark.add_argument("--engine", choices=tuple(ENGINE_NAMES), default="llama.cpp")
    benchmark.add_argument("--probe-version", action="store_true",
                           help="opt in to 5-second --version subprocess")
    version = commands.add_parser("version", help="hash executable and optionally call --version")
    version.add_argument("--binary", required=True)
    version.add_argument("--probe-version", action="store_true")
    record = commands.add_parser("record", help="parse already captured operator logs")
    record.add_argument("--stdout", required=True)
    record.add_argument("--stderr", required=True)
    record.add_argument("--exit-code", type=int, required=True)
    record.add_argument("--wall-ms", type=int, required=True)
    record.add_argument("--gguf-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = download_plan(Path(args.snapshot), args.host)
        elif args.command == "verify":
            result = verify_snapshot(Path(args.snapshot))
        elif args.command == "benchmark-plan":
            result = benchmark_plan(args.host, Path(args.gguf), Path(args.binary),
                                    args.engine, probe_version=args.probe_version)
        elif args.command == "version":
            result = binary_diagnostics(Path(args.binary),
                                        probe_version=args.probe_version)
        else:
            if args.wall_ms < 0 or not re.fullmatch("[0-9a-f]{64}", args.gguf_sha256):
                raise AdmissionError("invalid timing or sha256")
            result = parse_operator_logs(
                _read_bounded_log(Path(args.stdout)),
                _read_bounded_log(Path(args.stderr)),
                returncode=args.exit_code, wall_ms=args.wall_ms,
                gguf_sha256=args.gguf_sha256)
        print(json.dumps(result, sort_keys=True, indent=2))
        if args.command == "verify" and not result["shards_verified"]:
            return 2
        return 0
    except (AdmissionError, OSError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "HOLD",
                          "reason": str(exc), "inference_executed": False}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
