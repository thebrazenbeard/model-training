"""Offline, nonexecuting llama.cpp b12125 Windows CPU CLI argument builder.

Only one approved PUBLIC first-turn prompt may be put on a command line.
The command is a candidate awaiting on-host --help, hash, memory and
JobObject custody verification. It cannot establish multi-turn memory.
"""

from __future__ import annotations

from pathlib import PureWindowsPath

from .baseline import CASES


LLAMA_CPP_RELEASE_TAG = "b12125"
WINDOWS_CPU_ARCHIVE = "llama-b12125-bin-win-cpu-x64.zip"
WINDOWS_CPU_ARCHIVE_SHA256 = (
    "968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8"
)
ATTESTATION_URL = "https://github.com/ggml-org/llama.cpp/attestations/63933469"
CLI_DOC_REF = "https://github.com/ggml-org/llama.cpp/blob/b12125/tools/cli/README.md"

# Every non-default is confirmed in b12125 tools/cli/README.md. Explicitly
# disable the automatic memory-fit optimizer: otherwise nominal context/caps
# can silently change. Do not include any network/model-repository flags.
CPU_BASE_FLAGS = (
    "--gpu-layers", "0",
    "--ctx-size", "1024",
    "--predict", "64",
    "--temp", "0",
    "--seed", "0",
    "--threads", "2",
    "--threads-batch", "2",
    "--fit", "off",
    "--offline",
    "--no-display-prompt",
    "--log-disable",
    "--simple-io",
    "--no-show-timings",
    "--no-context-shift",
    "--reasoning", "off",
    "--reasoning-budget", "0",
)


def _absolute_local(path: str, *, suffix: str) -> str:
    if type(path) is not str or not 4 <= len(path) <= 512:
        raise ValueError("invalid local path")
    item = PureWindowsPath(path)
    if (not item.is_absolute() or not item.drive.endswith(":")
            or item.suffix.lower() != suffix or ".." in item.parts):
        raise ValueError("expected absolute local Windows path")
    return path


def build_first_turn_args(executable: str, model: str, case_id: str) -> list[str]:
    """Candidate one-process, first-turn-only public probe; DOES NOT execute.

    --single-turn exits after this prompt, so the correction test MUST NOT be
    faked by separate invocations. No private/user-supplied prompt accepted.
    Prompt is visible to a local process listing: only public case text is
    permitted. Launcher must separately verify archive/binary/model hashes.
    """
    exe = _absolute_local(executable, suffix=".exe")
    weight = _absolute_local(model, suffix=".gguf")
    if type(case_id) is not str:
        raise ValueError("unknown public case")
    selected = next((case for case in CASES if case["id"] == case_id), None)
    if selected is None:
        raise ValueError("unknown public case")
    return [
        exe, "--model", weight,
        *CPU_BASE_FLAGS,
        "--single-turn", "--prompt", selected["turns"][0],
    ]
