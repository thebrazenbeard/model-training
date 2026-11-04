"""Fail-closed local GGUF conversation probe for official Qwen3.5-4B.

This does not train, convert, authenticate a caller-provided provenance claim,
or grant a physical GPU lease. It only verifies named local GGUF bytes and
starts an explicitly selected, pinned llama-server *on loopback* for one prompt.
Run `preflight` first. `probe` must be invoked separately for base and R16.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
        "runtime_started": False,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, body: dict | None = None, *, timeout: int = 5) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    with urlopen(Request(url, data=data,
                         headers={"Content-Type": "application/json"}),
                 timeout=timeout) as result:
        doc = json.load(result)
    _require(type(doc) is dict, "runtime returned non-object JSON")
    return doc


def build_command(preflight: dict, *, port: int) -> list[str]:
    _require(1 <= port <= 65535, "invalid loopback port")
    p = preflight["paths"]
    cmd = [p["server"], "--model", p["base"], "--host", "127.0.0.1",
           "--port", str(port), "--ctx-size", "512", "--parallel", "1",
           "--threads", "2", "--threads-batch", "2", "--n-gpu-layers", "0"]
    if p["adapter"] is not None:
        cmd += ["--lora", p["adapter"]]
    return cmd


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


def probe(preflight: dict, *, host_idle_acknowledged: bool = False,
          timeout: int = 150) -> dict:
    """One bounded fresh-process CPU conversation; never touch existing services."""
    available_before = require_probe_resources(
        preflight, host_idle_acknowledged=host_idle_acknowledged)
    port = _free_port()
    command = build_command(preflight, port=port)
    # Keep noisy loader logs private to this owned temporary process.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   text=True)
        started = time.monotonic()
        try:
            while True:
                _require(process.poll() is None, "llama-server quit before ready")
                try:
                    h = _http_json(f"http://127.0.0.1:{port}/health", timeout=2)
                    if h.get("status") == "ok":
                        break
                except (HTTPError, URLError, OSError, TimeoutError, ValueError):
                    pass
                _require(time.monotonic() - started < timeout,
                         "llama-server readiness exceeded timeout")
                time.sleep(0.5)
            ready_sec = time.monotonic() - started
            response = _http_json(f"http://127.0.0.1:{port}/v1/chat/completions", {
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0, "max_tokens": 16, "stream": False,
                "seed": 20261104,
            }, timeout=timeout)
            elapsed = time.monotonic() - started
            choices = response.get("choices")
            _require(type(choices) is list and len(choices) == 1 and
                     type(choices[0]) is dict, "missing single completion choice")
            message = choices[0].get("message")
            _require(type(message) is dict and type(message.get("content")) is str,
                     "missing generated message")
            _require(process.poll() is None, "server exited during generation")
            return {**preflight,
                    "status": "ACTUAL_LOCAL_LLAMA_SERVER_GENERATION",
                    "runtime_started": True,
                    "runtime": "llama.cpp /v1/chat/completions, fresh process, CPU",
                    "readiness_seconds": round(ready_sec, 3),
                    "prelaunch_available_ram_bytes": available_before,
                    "host_idle_status": "OPERATOR_ACKNOWLEDGED_NOT_INDEPENDENTLY_VERIFIED",
                    "elapsed_seconds": round(elapsed, 3),
                    "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                    "generated_text": message["content"],
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": response.get("usage"),
                    "peak_ram_bytes": None,
                    "peak_vram_bytes": None,
                    "response_quality_verified": False}
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "probe"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["base", "trained"], required=True)
    parser.add_argument("--acknowledge-host-idle", action="store_true",
                        help="required for probe; operator confirms the host is not busy")
    args = parser.parse_args(argv)
    try:
        _require(args.manifest.is_file() and not args.manifest.is_symlink(),
                 "manifest path unavailable")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        bound = verify_manifest(manifest, args.mode)
        report = (probe(bound, host_idle_acknowledged=args.acknowledge_host_idle)
                  if args.action == "probe" else bound)
    except (InferenceHold, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REAL QWEN INFERENCE HOLD: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
