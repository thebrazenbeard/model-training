"""Bounded CPU-only Windows Job Object request rehearsal.

This module deliberately has NO native Windows bindings, subprocess launcher,
job handle, process watcher, or claim of an installed Windows runtime. It can
check a narrow lexical request schema using a recording fake. Windows device
aliases, symlinks/reparse points, 8.3 names, current host identity, executable
hashes, actual process limits, and Modern Standby must be qualified separately.

The production-facing launcher is *always unavailable*. A fake binding cannot
be promoted to a host authority or evidence of an actual process launch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,94}[A-Za-z0-9]")
_DRIVE_PATH = re.compile(r"([A-Za-z]):\\(.+)", flags=re.ASCII)
_COMPONENT = re.compile(r"[A-Za-z0-9_()+. -]+", flags=re.ASCII)
_MAX_JOB_SECONDS = 86_400
_MAX_FAKE_MEMORY_BYTES = 1 << 40
_DOS_DEVICE_DIGIT = "123456789¹²³"


class WindowsAdapterUnavailable(RuntimeError):
    """No live Windows Job Object launcher is installed or authorized."""


@dataclass(frozen=True, slots=True)
class JobRequest:
    """Untrusted request; construction itself does not authorize any effect."""

    run_id: str
    executable: str
    argv: tuple[str, ...]
    cwd: str
    budget_seconds: int | float
    memory_bytes: int
    operator_authorized: bool


@dataclass(frozen=True, slots=True)
class FakeJobBinding:
    run_id: str
    executable_identity: str
    cwd_identity: str
    budget_seconds: int | float
    memory_bytes: int
    provenance: str = "SOURCE_ONLY_FAKE_JOB_NOT_LIVE"
    authorizes_process_launch: bool = False
    enforces_live_memory_limit: bool = False
    enforces_live_time_limit: bool = False


def _windows_lexical_identity(value: object) -> str:
    """Fail-closed lexical identity, *not* filesystem identity or a real path pin.

    Restrict to an explicitly small ASCII Windows drive-absolute subset. No
    UNC, extended namespace, device namespace, relative path, alternate data
    stream, empty segment, dot traversal, trailing dot/space, or control char.
    """
    if type(value) is not str or not 3 < len(value) <= 4096 or "/" in value:
        raise ValueError("noncanonical Windows path")
    match = _DRIVE_PATH.fullmatch(value)
    if match is None:
        raise ValueError("noncanonical Windows path")
    parts = match.group(2).split("\\")
    if not parts:
        raise ValueError("empty Windows path")
    for part in parts:
        if (not part or part in (".", "..") or part[0] == " "
                or part[-1] in " ." or any(ord(char) < 32 or ord(char) == 127
                                          for char in part)):
            raise ValueError("noncanonical Windows path component")
        # Win32 DOS names remain devices even when followed by extensions.
        # Windows may trim spaces immediately before an extension during DOS
        # device recognition (e.g. ``CON .txt``); normalize *for refusal*.
        stem = part.split(".", 1)[0].rstrip(" ").upper()
        if (stem in ("CON", "PRN", "AUX", "NUL")
                or (len(stem) == 4 and stem[:3] in ("COM", "LPT")
                    and stem[3] in _DOS_DEVICE_DIGIT)):
            raise ValueError("reserved Windows DOS device name")
        if _COMPONENT.fullmatch(part) is None:
            raise ValueError("unsupported Windows path character")
    return match.group(1).lower() + ":\\" + "\\".join(part.lower() for part in parts)


def _validate_windows_request(request: object) -> tuple[str, str]:
    """Validate before ANY fake backend action; never authorizes live effects."""
    if type(request) is not JobRequest:
        raise ValueError("invalid Windows source-only request type")
    if (type(request.run_id) is not str or _RUN_ID.fullmatch(request.run_id) is None
            or ".." in request.run_id):
        raise ValueError("invalid run identifier")
    if type(request.operator_authorized) is not bool or not request.operator_authorized:
        raise ValueError("synthetic source exercise requires explicit operator flag")
    budget = request.budget_seconds
    if (type(budget) not in (int, float)
            or not 0 < budget <= _MAX_JOB_SECONDS
            or (type(budget) is float and not math.isfinite(budget))):
        raise ValueError("invalid synthetic Windows job time budget")
    if (type(request.memory_bytes) is not int or not 0 < request.memory_bytes
            <= _MAX_FAKE_MEMORY_BYTES):
        raise ValueError("invalid synthetic memory limit")
    if type(request.argv) is not tuple or len(request.argv) > 128:
        raise ValueError("invalid source-only argv")
    for arg in request.argv:
        if (type(arg) is not str or len(arg) > 4096
                or any(ord(char) < 32 or ord(char) == 127 for char in arg)):
            raise ValueError("invalid source-only argv item")
    executable_identity = _windows_lexical_identity(request.executable)
    cwd_identity = _windows_lexical_identity(request.cwd)
    return executable_identity, cwd_identity


class FakeWindowsJobBackend:
    """Records in-memory fake bindings; cannot create or control a process."""

    def __init__(self):
        self.calls: list[FakeJobBinding] = []

    def bind(self, request: JobRequest, executable_id: str, cwd_id: str) -> FakeJobBinding:
        # Even direct use of the recording fake must not bypass validation.
        if type(self.calls) is not list:
            raise ValueError("invalid source-only fake recording store")
        expected_executable_id, expected_cwd_id = _validate_windows_request(request)
        if (type(executable_id) is not str or type(cwd_id) is not str
                or executable_id != expected_executable_id
                or cwd_id != expected_cwd_id):
            raise ValueError("synthetic Windows path identity mismatch")
        binding = FakeJobBinding(
            run_id=request.run_id,
            executable_identity=executable_id,
            cwd_identity=cwd_id,
            budget_seconds=request.budget_seconds,
            memory_bytes=request.memory_bytes,
        )
        self.calls.append(binding)
        return binding


# Capture the local no-effects implementation once. A caller controls the fake
# instance and may replace its ``bind`` attribute; even the class attribute can
# be patched in-process. Neither substitution may create a live-looking receipt
# through the public source-only rehearsal function.
_CANONICAL_FAKE_BIND = FakeWindowsJobBackend.bind


def exercise_fake_windows_binding(request: JobRequest, *, backend: FakeWindowsJobBackend
                                  ) -> FakeJobBinding:
    """Only fake/source simulation; no subprocess or native Job Object calls."""
    if type(backend) is not FakeWindowsJobBackend:
        raise ValueError("only an exact in-memory fake backend is supported")
    calls = backend.calls
    if type(calls) is not list:
        raise ValueError("invalid source-only fake recording store")
    executable_id, cwd_id = _validate_windows_request(request)
    preceding_calls = len(calls)
    binding = _CANONICAL_FAKE_BIND(backend, request, executable_id, cwd_id)
    if (type(binding) is not FakeJobBinding
            or backend.calls is not calls
            or len(calls) != preceding_calls + 1
            or calls[-1] is not binding
            or type(binding.run_id) is not str
            or binding.run_id != request.run_id
            or type(binding.executable_identity) is not str
            or binding.executable_identity != executable_id
            or type(binding.cwd_identity) is not str
            or binding.cwd_identity != cwd_id
            or type(binding.budget_seconds) is not type(request.budget_seconds)
            or binding.budget_seconds != request.budget_seconds
            or type(binding.memory_bytes) is not int
            or binding.memory_bytes != request.memory_bytes
            or binding.provenance != "SOURCE_ONLY_FAKE_JOB_NOT_LIVE"
            or binding.authorizes_process_launch is not False
            or binding.enforces_live_memory_limit is not False
            or binding.enforces_live_time_limit is not False):
        raise ValueError("forged source-only Windows fake binding")
    return binding


class SourceOnlyWindowsJobLauncher:
    """Explicit non-adapter: always reject an attempted production launch."""

    def launch(self, _request: JobRequest) -> None:
        raise WindowsAdapterUnavailable(
            "Source-only Windows Job Object contract: no live process launch available")


__all__ = [
    "FakeJobBinding", "FakeWindowsJobBackend", "JobRequest",
    "SourceOnlyWindowsJobLauncher", "WindowsAdapterUnavailable",
    "exercise_fake_windows_binding",
]
