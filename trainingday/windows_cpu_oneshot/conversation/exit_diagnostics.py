"""Offline inference-startup exit triage; never launch, replay or alter a host.

An 82 ms empty stdout/stderr report is not enough to identify a cause. Windows
GetExitCodeProcess, child creation state, JobObject termination and CLI
redirection semantics are independent observations. Every outcome below
remains unqualified as a physical model-inference result.

Status names use Microsoft MS-ERREF Windows NTSTATUS values:
https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitObservation:
    elapsed_ms: int
    exit_code: int | None
    stdout_bytes: int
    stderr_bytes: int
    process_created: bool
    owner_stop_reported: bool


_NTSTATUS = {
    0xC0000135: (
        "DLL_LOAD_MISSING",
        "Windows loader could not find a required DLL; runtime or app-local dependency possible.",
        "Compare archived CPU zip inventory with extracted app-local DLL inventory, and read-only VC++ runtime version; no installs.",
    ),
    0xC000007B: (
        "INVALID_IMAGE_FORMAT",
        "A binary image is invalid or architecture-incompatible; exe/DLL bitness or integrity possible.",
        "Read PE machine types and compare exact attested archive bytes to extracted exe/DLL hash inventory.",
    ),
    0xC0000142: (
        "DLL_INITIALIZATION_FAILED",
        "A DLL failed during startup initialization; dependency/CRT/security policy remain possible.",
        "Inspect read-only Windows event/WER summary and DLL dependency inventory, without installing or bypassing policy.",
    ),
    0xC000001D: (
        "ILLEGAL_INSTRUCTION",
        "CPU instruction is unsupported in current execution mode, or a bad instruction path was reached.",
        "Compare read-only CPU ISA capabilities with the exact compiled binary's target; do not label AVX2 from timing alone.",
    ),
    0xC0000005: (
        "ACCESS_VIOLATION",
        "Native access violation; not enough to choose between CRT compatibility, app bug and corrupt/untrusted image.",
        "Inspect read-only WER module/exception and VC++ runtime version; historical upstream CRT fix is not proof here.",
    ),
    0xC0000006: (
        "IN_PAGE_ERROR",
        "Mapped or paged data could not be retrieved; storage, paging, or image issue possible.",
        "Inspect scoped read-only filesystem and relevant system event status before considering another run.",
    ),
    0xC0000017: (
        "VIRTUAL_MEMORY_EXHAUSTED",
        "Insufficient committed virtual memory or paging-file quota.",
        "Inspect read-only process peak commit and system memory/commit state; do not expand pagefile or limits.",
    ),
    0xC000013A: (
        "CONTROL_C_EXIT",
        "Application reported termination by Ctrl+C or an equivalent control path.",
        "Correlate owner supervisor stop reason, signal timing and process-tree readback.",
    ),
}


def _validate(obs: ExitObservation) -> None:
    if type(obs) is not ExitObservation:
        raise ValueError("expected explicit exit observation")
    for key in ("elapsed_ms", "stdout_bytes", "stderr_bytes"):
        value = getattr(obs, key)
        if type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise ValueError(f"invalid {key}")
    for key in ("process_created", "owner_stop_reported"):
        if type(getattr(obs, key)) is not bool:
            raise ValueError(f"invalid {key}")
    if obs.exit_code is not None and (
            type(obs.exit_code) is not int
            or not -2**31 <= obs.exit_code <= 2**32 - 1):
        raise ValueError("invalid Windows process exit code")


def diagnose_exit(obs: ExitObservation) -> dict:
    """Classify an observation; never transform it into model success."""
    _validate(obs)
    hexcode = (
        None if obs.exit_code is None else f"0x{obs.exit_code & 0xFFFFFFFF:08X}"
    )
    normalized = None if obs.exit_code is None else obs.exit_code & 0xFFFFFFFF
    if not obs.process_created:
        category = "PROCESS_NOT_CREATED"
        explanation = "No child creation was observed; no process exit can be inferred."
        next_step = "Read separate CreateProcess result/Win32 error; no silent retry."
    elif obs.owner_stop_reported:
        category = "OWNER_STOP_REPORTED"
        explanation = "Owner claims a stop/JobObject termination; this observation is not an independent kill receipt."
        next_step = "Correlate exact JobObject termination, PID creation time and child-tree readback."
    elif normalized is None:
        category = "EXIT_CODE_MISSING"
        explanation = "No raw child exit code available; cannot distinguish loader, argument parser or transport."
        next_step = "Read original OS process exit status including signed/hex representation, not wrapper code."
    elif normalized in _NTSTATUS:
        category, explanation, next_step = _NTSTATUS[normalized]
    elif normalized == 0 and obs.stdout_bytes == 0 and obs.stderr_bytes == 0:
        category = (
            "EARLY_EMPTY_ZERO_EXIT" if obs.elapsed_ms < 1000
            else "EMPTY_ZERO_EXIT"
        )
        explanation = (
            "Zero exit with no captured bytes is not proof of execution. "
            "The CLI may be inert under piped IO or the capture path may be wrong."
        )
        next_step = (
            "Read-only compare captured child exit code, redirected handle setup, "
            "process start/stop events and exact b12125 CLI mode. "
            "Do not re-run until owner gates authorize."
        )
    elif normalized == 0 and obs.stdout_bytes == 0:
        category = "EXIT_ZERO_STDERR_ONLY"
        explanation = "Successful exit status but no captured generated output."
        next_step = "Inspect bounded stderr category and stdout pipe lifecycle in a private owner receipt."
    elif normalized == 0:
        category = "OUTPUT_PRESENT_UNVERIFIED"
        explanation = (
            "Output bytes and exit zero do not authenticate the loaded model, "
            "prompt processing or response quality."
        )
        next_step = "Require separately attested model/binary/process binding and private human review."
    else:
        category = "UNCLASSIFIED_NONZERO_EXIT"
        explanation = "Unexpected code; cannot infer root cause from duration or empty streams."
        next_step = "Preserve exact hexadecimal exit code and identify its originating process."
    return {
        "classification": category,
        "normalized_exit_hex": hexcode,
        "elapsed_ms": obs.elapsed_ms,
        "captured_stdout_bytes": obs.stdout_bytes,
        "captured_stderr_bytes": obs.stderr_bytes,
        "evidence_state": "UNCONFIRMED_INFERENCE",
        "physical_inference_proved": False,
        "explanation": explanation,
        "next_read_only_discriminator": next_step,
    }
