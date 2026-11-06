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
# Exact native DLL/EXE bundle from the official ggml-org/llama.cpp b12125
# Windows x64 CPU archive. SHA-256 of the ZIP was independently checked on
# VN1 against GitHub's b12125 release attestation subject:
# 968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8
# Staged EXE/DLL files were independently rehashed on VN1 (2026-11-05).
# The manifest binds exact distributed bytes, NOT formal build-lineage proof.
REVIEWED_RUNNER_SHA256 = "02042615249dd4852f47cc2d4dd5caabba91766b5161dd4642ed6272f788172e"
REVIEWED_RUNTIME_FILES_SHA256: dict[str, str] = {
    "ggml-base.dll": "677e00edd688e5ab2812421a6a8154c0c81385f5aec14664c62275fe0e8ffd26",
    "ggml-cpu-alderlake.dll": "d38b846776fc75e52d01d859a8f002101dfe7ac8284872fff74864a5c80374f5",
    "ggml-cpu-cannonlake.dll": "3a123c84da08eff7f76a3f0f684148426f7caf0a3c736ce6824ccefea2756e89",
    "ggml-cpu-cascadelake.dll": "b5a2b01550ec816435bf53b395b74209e91ab94e91c2bdfe69a9fe2b97bc0a42",
    "ggml-cpu-cooperlake.dll": "fd38ea8115267677ddf502d3a08317c53acaa3eab2698e75993227a41be87bfc",
    "ggml-cpu-haswell.dll": "693331b22e0b663fe4ab9d4844a7646620225958749052ace3cdcf2bff4d89ad",
    "ggml-cpu-icelake.dll": "f288a515c1ffd7bdff439a2bdede533d64247385d237b9c22a019b08005e53d5",
    "ggml-cpu-ivybridge.dll": "2a32a0e136843402eb9c4f76646370657cbc4c7da042ba272bdfe3733314b48a",
    "ggml-cpu-piledriver.dll": "e4f621e8392a4ff68a73f7f6fd78cf137a6cd8c51f2d3f1d6a9b9a9297fa1e34",
    "ggml-cpu-sandybridge.dll": "09b45075a870305182a82ba584429d34ef65f0cc6538440c3393ca47c977223f",
    "ggml-cpu-sapphirerapids.dll": "8ec5839f1b36fba99cb9850703ff7cfd82867877ce4ba016207ad0cd020db0af",
    "ggml-cpu-skylakex.dll": "97ac403dbf9a144f572a961521f4a10a53025c05704cc8fcd8cfa2db1df9b8c3",
    "ggml-cpu-sse42.dll": "bf2d20c091f3ecde50dc642c57d30700dc3ddd83631ebb1ca82f67e4f1da68e9",
    "ggml-cpu-x64.dll": "34a92dd071e07b979715ec6c28282e7684f8f051dc9604c5191b827a1ea4eace",
    "ggml-cpu-zen4.dll": "2648e542e5abd061cca5264f303a925353974fba208ee7f16c131e05285c8fba",
    "ggml.dll": "27c4a9a9b48ebf8d7aa3ec50152b38efecd5afd61d5886023561c1f9823a82f8",
    "libomp.dll": "a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c",
    "llama-cli.exe": "02042615249dd4852f47cc2d4dd5caabba91766b5161dd4642ed6272f788172e",
    "llama.dll": "669c47bee16dcad5f6e6d0755f16c9d0969508c1ac1983f3e2f74280ab44822a",
}
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
            str(self.executable), "--model", str(self.model),
            "--ctx-size", str(self.context),
            "--predict", str(self.output_tokens),
            "--gpu-layers", "0", "--threads", str(self.threads),
            "--threads-batch", str(self.threads),
            "--seed", "0", "--temp", "0", "--fit", "off",
            "--offline", "--no-display-prompt", "--log-disable",
            "--simple-io", "--no-show-timings", "--no-context-shift",
            "--reasoning", "off", "--reasoning-budget", "0",
            "--single-turn", "--prompt", PROMPT,
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


def competing_model_workers(processes: dict[str, set[int]], owner_pid: int | None = None) -> list[str]:
    """Block every known model worker except the one kernel-owned llama-cli PID.

    Never infer ownership from executable name. At preflight owner_pid is None
    and any preexisting llama-cli must block. The Win32 Job launcher supplies
    its exact held PROCESS_INFORMATION PID once the suspended child is assigned.
    """
    if owner_pid is not None and (type(owner_pid) is not int or owner_pid <= 0):
        raise Hold("unverified child PID exemption")
    found: list[str] = []
    for name in sorted(BLOCKING_PROCESSES - {"ollama.exe"}):
        pids = set(processes.get(name, ()))
        if name == "llama-cli.exe" and owner_pid is not None:
            pids.discard(owner_pid)
        if pids:
            found.append(name)
    return found


def ollama_inventory_empty(raw: bytes) -> bool:
    """Interpret bounded local /api/ps response; missing/invalid is HOLD."""
    if not isinstance(raw, bytes) or len(raw) > 65536:
        raise Hold("Ollama inventory response missing or oversized")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise Hold("Ollama inventory response is not JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise Hold("Ollama inventory response has no authoritative models list")
    if data["models"]:
        raise Hold("Ollama has a loaded model")
    return True
