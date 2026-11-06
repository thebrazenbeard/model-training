"""One VN1 loader probe with no GGUF arguments; file access not traced."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

if os.name != "nt":
    raise SystemExit("HOLD: Windows-only probe")

import policy
import supervisor
from policy import GIB, Hold, MIN_RUNTIME_FREE_BYTES
from windows_job import PinnedReadFile, Win32Hold, assert_not_reparse, run_job


ROOT = Path(r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2")
BIN = ROOT / "bin"
STATE = ROOT / "state" / "loader-probe-v1"
EXE = BIN / "llama-cli.exe"
PYTHON = Path(r"C:\Users\Vera\AppData\Local\Programs\Python\Python313\python.exe")
PYTHON_SHA = "85b71d8c6ec1905935f74be0c9869aae198d00e98f39df699ec66f9c5a84cecd"
NEW_DLLS = {
    "llama-cli-impl.dll": "e60450e7897b025c4ce1eba615dcf66e2d95cea90d022aa161d31ef1d4f13eb1",
    "llama-server-impl.dll": "5cb7e275a9a5c3a697c1a6a95da866e28f8ccd379f5b7ba1a3df367ffbe53569",
    "llama-common.dll": "9f1e8a5714baabd1fb72187698f1ec358793d86c6be308681fc1dd49b757cf54",
    "mtmd.dll": "5713c5478ba954848ae62ed02ecdc49a0524612646f887406483503af95bea71",
}


def sha_file(path: Path) -> str:
    with PinnedReadFile(path) as held:
        return held.sha256()


def main() -> int:
    if len(sys.argv) != 1:
        raise Hold("loader probe takes no arguments")
    supervisor._require_owner_identity()
    if Path(sys.executable) != PYTHON or sha_file(PYTHON) != PYTHON_SHA:
        raise Hold("wrong Python executable or bytes")
    expected = dict(policy.REVIEWED_RUNTIME_FILES_SHA256)
    if len(expected) != 19 or set(expected).intersection(NEW_DLLS):
        raise Hold("original approved 19-file pin map changed")
    expected.update(NEW_DLLS)
    if len(expected) != 23 or expected.get(EXE.name) != policy.REVIEWED_RUNNER_SHA256:
        raise Hold("v2 native bundle contract failed")
    for path in (ROOT, BIN, ROOT / "state", ROOT / "stage-receipt.json"):
        assert_not_reparse(path)
    receipt = json.loads((ROOT / "stage-receipt.json").read_text(encoding="utf-8-sig"))
    inventory = receipt.get("Binaries")
    actual = {p.name: p for p in BIN.iterdir() if p.suffix.lower() in (".exe", ".dll", ".pyd")}
    if (receipt.get("Schema") != "bt2.q4.stage.receipt.v2"
            or receipt.get("Destination") != str(ROOT)
            or not isinstance(inventory, list) or len(inventory) != 23
            or {item["File"]: item["Sha256"] for item in inventory} != expected
            or set(actual) != set(expected)):
        raise Hold("stage receipt or native file set not exact")

    with supervisor.single_slot():
        occupancy = supervisor.OllamaOccupancyGuard()
        preflight = supervisor._host_preflight(5 * GIB, "--run", occupancy)
        with ExitStack() as held_files:
            for name in sorted(expected):
                pinned = held_files.enter_context(PinnedReadFile(actual[name]))
                if pinned.sha256() != expected[name]:
                    raise Hold("native bundle SHA-256 mismatch: " + name)
            occupancy.check()
            # A fresh isolated state leaf is the one-use marker. Never reuse it.
            STATE.mkdir(parents=False, exist_ok=False)
            nonce = uuid.uuid4().hex
            marker = {"schema": "VERA_VN1_LOADER_PROBE_CONSUMED_V1", "nonce": nonce,
                      "at_utc": time.time(), "argv": [str(EXE), "--version"]}
            supervisor._receipt(STATE / "consumed.json", marker)
            out = STATE / "loader.out"
            err = STATE / "loader.err"
            started = time.time()
            try:
                result = run_job(
                    executable=EXE, argv=[str(EXE), "--version"], cwd=BIN,
                    stdout_path=out, stderr_path=err,
                    memory_cap=512 * 1024 ** 2, cpu_percent=20,
                    wall_seconds=8, runtime_free_floor=MIN_RUNTIME_FREE_BYTES,
                    occupancy_guard=occupancy.check,
                )
                # A clean exit alone is weaker than a reported version.
                banner = (out.read_text(encoding="utf-8", errors="replace") + "\n"
                          + err.read_text(encoding="utf-8", errors="replace"))[:8192]
                reported_version = "version" in banner.casefold()
                body = {
                    "schema": "VERA_VN1_LOADER_PROBE_RECEIPT_V1", "nonce": nonce,
                    "state": "PASS" if result.exit_code == 0 and result.reason == "CHILD_EXITED"
                        and result.process_handle_signaled and result.job_active_processes_after == 0
                        and reported_version else "FAIL",
                    "child_pid": result.pid, "exit_code": result.exit_code,
                    "reason": result.reason, "child_handle_signaled": result.process_handle_signaled,
                    "job_active_processes_after": result.job_active_processes_after,
                    "accounting_polls": result.accounting_poll_count,
                    "accounting_wait_seconds": result.accounting_wait_seconds,
                    "stdout_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(err.read_bytes()).hexdigest(),
                    "stdout_bytes": out.stat().st_size, "stderr_bytes": err.stat().st_size,
                    "version_reported": reported_version, "banner_excerpt": banner[:1024],
                    "host_preflight": preflight, "occupancy_guard_checks": occupancy.check_count,
                    "elapsed_wall_seconds": round(time.time() - started, 3),
                    "gguf_argument_passed": False, "gguf_access_verified": False,
                    "model_inference_requested": False,
                }
            except Exception as exc:
                body = {"schema": "VERA_VN1_LOADER_PROBE_RECEIPT_V1", "nonce": nonce,
                        "state": "UNKNOWN_NO_REPLAY", "failure_type": type(exc).__name__,
                        "failure": str(exc), "gguf_argument_passed": False,
                        "gguf_access_verified": False, "model_inference_requested": False}
                supervisor._receipt(STATE / "receipt.json", body)
                raise
            supervisor._receipt(STATE / "receipt.json", body)
            print(json.dumps(body, sort_keys=True))
            return 0 if body["state"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Hold, Win32Hold, OSError, ValueError) as exc:
        print("HOLD: " + str(exc), file=sys.stderr)
        raise SystemExit(3)
