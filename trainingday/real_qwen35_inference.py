"""Fail-closed offline GGUF preflight for Qwen3.5-4B (no model launch).

An untrusted manifest and a matching SHA do not authenticate a server binary,
conversion ancestry, listener ownership, process custody, or GPU lease.
Therefore the `probe` action is deliberately disabled until those boundaries
are independently implemented and qualified by a trusted host-owned launcher.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


SCHEMA = "TRAININGDAY_QWEN35_REAL_GGUF_PROBE_V1"
BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OFFICIAL_SHARD_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
PROMPT = "What is 17 + 23? Reply with only the number."


class InferenceHold(ValueError):
    """A required local source, binding, or runtime check is missing."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceHold(message)


def _absolute_regular(value: object, *, executable: bool = False) -> Path:
    _require(type(value) is str and bool(value), "missing file path")
    p = Path(value)
    _require(p.is_absolute(), "relative file path rejected")
    # Reject directory links and the final component; do not follow links to
    # mutable files while attesting a model. This is not a host security ACL.
    _require(not any(part.is_symlink() for part in (p, *p.parents)), "symlink path rejected")
    _require(p.is_file(), "file does not exist")
    if executable:
        _require(p.stat().st_size > 0, "empty server executable")
    return p


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified_file(record: object, *, gguf: bool = False) -> tuple[Path, str]:
    _require(type(record) is dict and set(record) == {"path", "sha256"},
             "invalid artifact binding")
    path = _absolute_regular(record["path"])
    digest = record["sha256"]
    _require(type(digest) is str and len(digest) == 64 and
             all(ch in "0123456789abcdef" for ch in digest), "invalid SHA256")
    _require(_hash_file(path) == digest, "artifact bytes do not match binding")
    if gguf:
        _require(path.suffix.lower() == ".gguf", "GGUF extension required")
        with path.open("rb") as stream:
            _require(stream.read(4) == b"GGUF", "GGUF magic mismatch")
    return path, digest


def verify_manifest(manifest: dict, mode: str) -> dict:
    """Byte-check GGUF, optional R16 adapter, server, and source declarations.

    A caller-created manifest cannot prove that a derived GGUF came from
    official shards; this remains NOT_INDEPENDENTLY_ATTESTED until conversion
    receipts plus custody are verified outside this tool.
    """
    _require(mode in {"base", "trained"}, "unknown run mode")
    _require(type(manifest) is dict and manifest.get("schema") == SCHEMA,
             "unknown manifest schema")
    _require(manifest.get("base_id") == BASE_ID and
             manifest.get("base_revision") == BASE_REVISION,
             "not the pinned official Qwen3.5-4B source identity")
    _require(manifest.get("official_shard_sha256") == OFFICIAL_SHARD_SHA256,
             "official shard digest declaration mismatch")
    base, base_sha = _verified_file(manifest.get("base_gguf"), gguf=True)
    server, server_sha = _verified_file(manifest.get("llama_server"))
    adapter = manifest.get("r16_correct8_adapter_gguf")
    adapter_path = None
    adapter_sha = None
    if mode == "trained":
        _require(type(adapter) is dict and
                 adapter.get("source_run") == "r16-v2-correct8-20261030-b",
                 "trained mode requires the correct-label R16 adapter")
        adapter_path, adapter_sha = _verified_file(
            {"path": adapter.get("path"), "sha256": adapter.get("sha256")}, gguf=True)
        _require(adapter_path != base, "adapter path aliases base GGUF")
    return {
        "schema": SCHEMA,
        "status": "LOCAL_BYTES_VERIFIED_PROVENANCE_NOT_INDEPENDENTLY_ATTESTED",
        "mode": mode,
        "source": f"{BASE_ID}@{BASE_REVISION}",
        "base_gguf_sha256": base_sha,
        "base_gguf_bytes": base.stat().st_size,
        "adapter_gguf_sha256": adapter_sha,
        "llama_server_sha256": server_sha,
        "paths": {"base": str(base), "adapter": str(adapter_path) if adapter_path else None,
                  "server": str(server)},
        "official_shards_freshly_checked": False,
        "conversion_lineage_independently_verified": False,
        "server_binary_identity_independently_verified": False,
        "host_file_immutability_verified": False,
        "listener_process_binding_verified": False,
        "runtime_probe_qualified": False,
        "runtime_started": False,
    }


def available_ram_bytes() -> int:
    """Read current OS available physical RAM without a dependency or swap fiction."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        _require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))),
                 "Windows physical available RAM unavailable")
        return int(status.ullAvailPhys)
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                _require(len(parts) == 3 and parts[2] == "kB" and parts[1].isdigit(),
                         "Linux available RAM report malformed")
                return int(parts[1]) * 1024
    raise InferenceHold("available physical RAM cannot be established on this host")


def require_probe_resources(preflight: dict, *, host_idle_acknowledged: bool) -> int:
    """Refuse to compete with host work or model size before spawning a process.

    The human idle assertion is not a host-owned lease or process custody proof.
    """
    _require(host_idle_acknowledged is True,
             "explicit owner confirmation that host workloads are idle required")
    model_bytes = preflight["base_gguf_bytes"]
    _require(type(model_bytes) is int and model_bytes >= 5,
             "model size missing from verified preflight")
    minimum = max(5 * 1024**3, int(model_bytes * 1.3) + 1024**3)
    available = available_ram_bytes()
    _require(available >= minimum,
             f"insufficient available physical RAM: {available} < {minimum}; swap not counted")
    return available


def probe(*_args, **_kwargs) -> dict:
    """Hard HOLD: no self-attested server execution or HTTP receipt accepted."""
    raise InferenceHold(
        "runtime probe disabled: independent binary identity, immutable file "
        "custody, bound listener ownership and process-tree cleanup not qualified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "probe"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["base", "trained"], required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "probe":
            probe()
        _require(args.manifest.is_file() and not args.manifest.is_symlink(),
                 "manifest path unavailable")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = verify_manifest(manifest, args.mode)
    except (InferenceHold, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REAL QWEN INFERENCE HOLD: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
