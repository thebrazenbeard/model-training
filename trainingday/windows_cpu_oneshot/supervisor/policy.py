"""Immutable, deliberately narrow plan for one public VN1 CPU-only Q4 smoke.

No remote command, Python expression, arbitrary prompt, or dynamic server port is
accepted.  Parsing this plan is not itself permission to launch a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import PureWindowsPath


SCHEMA = "VERA_VN1_CPU_Q4_ONESHOT_V1"
HOST = "VERA-NODE-1"
MODEL_SHA256 = "b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca"
# Must be filled from a separately reviewed exact pinned Windows runner build.
# A SHA asserted solely by the caller's mutable JSON has no trust value.
# Leave blank (HOLD) until the coordinator has an authenticated source/build.
REVIEWED_RUNNER_SHA256 = ""
# Exact native DLL/EXE bundle manifest. The empty map is deliberately HOLD.
# Populate only after independent build provenance and immutable ACL review.
# Keys are lowercase basenames for every .exe/.dll/.pyd in the staging bin.
REVIEWED_RUNTIME_FILES_SHA256: dict[str, str] = {}
TRIAL_ROOT = PureWindowsPath(r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot")
MODEL_ROOT = TRIAL_ROOT / "models"
RUNNER_ROOT = TRIAL_ROOT / "bin"
STATE_ROOT = TRIAL_ROOT / "state"
# Harmless JobObject validation never creates TRIAL_ROOT or any child beneath
# it. The atomically staging writer must see the trial leaf as nonexistent.
SELFTEST_STATE_ROOT = PureWindowsPath(r"C:\Users\Vera\AppData\Local\Temp\w56-q4-selftest")
PROMPT = (
    "Please introduce yourself in your own words. If you do not know who you "
    "are, say so without inventing memories, achievements, or a personal history."
)
GIB = 1024 ** 3
MAX_CONTEXT = 1024
MAX_OUTPUT = 64
MAX_THREADS = 2
MAX_WALL_SECONDS = 180
MAX_JOB_BYTES = 4 * GIB
MIN_FREE_BYTES = 5 * GIB
MIN_POST_JOB_RESERVE = 1536 * 1024 ** 2
MIN_RUNTIME_FREE_BYTES = 2 * GIB
MAX_HOST_CPU_PERCENT = 25
BLOCKING_PROCESSES = frozenset({
    "llama-cli.exe", "llama-server.exe", "llama-bench.exe", "koboldcpp.exe",
    "ollama.exe", "ollama_llama_server.exe", "torchrun.exe",
})


class Hold(ValueError):
    """A source or policy requirement is not satisfied; nothing may launch."""


@dataclass(frozen=True)
class Plan:
    plan_id: str
    executable: PureWindowsPath
    executable_sha256: str
    model: PureWindowsPath
    model_sha256: str
    context: int
    output_tokens: int
    threads: int
    wall_seconds: int
    job_memory_bytes: int
    cpu_rate_percent: int

    def args(self) -> list[str]:
        """Fixed llama.cpp CLI input; never shell-joined or caller extended."""
        return [
            str(self.executable), "-m", str(self.model),
            "-c", str(self.context), "-n", str(self.output_tokens),
            "-ngl", "0", "-t", str(self.threads),
            "--seed", "0", "--temp", "0",
            "-p", PROMPT,
        ]


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Hold(f"{field}: expected lowercase SHA-256")
    return value


def _integer(value: object, field: str, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise Hold(f"{field}: outside approved bounds")
    return value


def _bounded_file(value: object, root: PureWindowsPath, suffix: str, field: str) -> PureWindowsPath:
    if not isinstance(value, str):
        raise Hold(f"{field}: expected absolute Windows path")
    path = PureWindowsPath(value)
    # Avoid lexical aliases, dot segments, and ADS; exact parent restricts
    # every candidate to one literal approved staging directory.
    if (
        not path.is_absolute() or path.parent != root or
        path.suffix.lower() != suffix or
        ":" in path.name or ".." in value.replace("/", "\\").split("\\")
    ):
        raise Hold(f"{field}: not an exact approved staging file")
    return path


def parse_plan(raw: str | bytes) -> Plan:
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise Hold("invalid JSON plan") from exc
    allowed = {
        "schema", "host", "plan_id", "executable", "executable_sha256",
        "model", "model_sha256", "context", "output_tokens", "threads",
        "wall_seconds", "job_memory_bytes", "cpu_rate_percent",
    }
    if not isinstance(data, dict) or set(data) != allowed:
        raise Hold("exact approved plan fields required")
    if data["schema"] != SCHEMA or data["host"] != HOST:
        raise Hold("wrong schema or host")
    if not isinstance(data["plan_id"], str) or re.fullmatch(r"[0-9a-f]{32}", data["plan_id"]) is None:
        raise Hold("plan_id: lowercase 128-bit nonce required")
    executable = _bounded_file(data["executable"], RUNNER_ROOT, ".exe", "executable")
    if executable.name.lower() != "llama-cli.exe":
        raise Hold("runner must be llama-cli.exe")
    model = _bounded_file(data["model"], MODEL_ROOT, ".gguf", "model")
    if model.name != "Qwen3.5-4B-Q4_K_M.gguf":
        raise Hold("model name not approved for this baseline")
    runner_sha = _sha(data["executable_sha256"], "executable_sha256")
    if (runner_sha != REVIEWED_RUNNER_SHA256 or
            REVIEWED_RUNTIME_FILES_SHA256.get("llama-cli.exe") != runner_sha):
        raise Hold("runner digest is not independently approved in installed policy")
    model_sha = _sha(data["model_sha256"], "model_sha256")
    if model_sha != MODEL_SHA256:
        raise Hold("publisher's published Q4 bytes do not match the approved manifest")
    context = _integer(data["context"], "context", 1, MAX_CONTEXT)
    output_tokens = _integer(data["output_tokens"], "output_tokens", 1, MAX_OUTPUT)
    threads = _integer(data["threads"], "threads", 1, MAX_THREADS)
    wall_seconds = _integer(data["wall_seconds"], "wall_seconds", 1, MAX_WALL_SECONDS)
    job_bytes = _integer(data["job_memory_bytes"], "job_memory_bytes", 3 * GIB, MAX_JOB_BYTES)
    cpu_rate = _integer(data["cpu_rate_percent"], "cpu_rate_percent", 1, MAX_HOST_CPU_PERCENT)
    return Plan(data["plan_id"], executable, runner_sha, model, model_sha,
                context, output_tokens, threads, wall_seconds, job_bytes, cpu_rate)


def required_prelaunch_free(plan: Plan) -> int:
    return max(MIN_FREE_BYTES, plan.job_memory_bytes + MIN_POST_JOB_RESERVE)


def state_root_for_mode(mode: str) -> PureWindowsPath:
    if mode == "--self-test":
        return SELFTEST_STATE_ROOT
    if mode == "--run":
        return STATE_ROOT
    raise Hold("unrecognized one-shot mode")


def blocking_processes_for_mode(mode: str) -> frozenset[str]:
    """Ollama's idle UI/daemon is harmless only for the fixed self-test.

    Process names cannot prove the Ollama daemon has no loaded model. The
    self-test remains capped to a harmless Python child, while real `--run`
    retains the original strict refusal even for resident ollama.exe.
    """
    if mode == "--self-test":
        return BLOCKING_PROCESSES - {"ollama.exe"}
    if mode == "--run":
        return BLOCKING_PROCESSES
    raise Hold("unrecognized one-shot mode")
