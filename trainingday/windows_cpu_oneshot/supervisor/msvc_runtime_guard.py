"""Read-only, fail-closed VC++ runtime *prerequisite* gate for VN1.

This does not authorize a model launch, mint a nonce, update system software,
or prove which DLLs the Windows loader will eventually resolve. The actual
b12125 build-run toolset/minimum runtime version is not verified, so the
shipped contract deliberately returns HOLD regardless of host observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
from typing import Any


RUNNER_SHA256 = "02042615249dd4852f47cc2d4dd5caabba91766b5161dd4642ed6272f788172e"
RUNTIME_FILES = ("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")
STAGED_BIN = Path(r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2\bin")
VERSION_RE = re.compile(r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", re.ASCII)
SHA_RE = re.compile(r"[a-f0-9]{64}", re.ASCII)
MAX_VERSION_PART = (1 << 32) - 1


class Hold(ValueError):
    """The prerequisite cannot be established; the runner must not start."""


def parse_version(value: object) -> tuple[int, int, int, int]:
    """Strict four-component numeric comparison; no lexical comparisons."""
    if not isinstance(value, str) or len(value) > 64:
        raise Hold("invalid_version")
    match = VERSION_RE.fullmatch(value)
    if match is None:
        raise Hold("invalid_version")
    result = tuple(map(int, match.groups()))
    if any(part > MAX_VERSION_PART for part in result):
        raise Hold("invalid_version")
    return result  # type: ignore[return-value]


def parse_registry_version(value: object) -> tuple[int, int, int, int]:
    """Windows redist uses an observed zero-padded fourth field ('...00')."""
    if isinstance(value, str) and value.endswith(".00"):
        value = value[:-2] + "0"
    return parse_version(value)


@dataclass(frozen=True)
class BuildContract:
    runner_sha256: str
    minimum_runtime_version: str | None
    build_run_evidence_sha256: str | None
    evidence_class: str


# Only upstream release workflow *source* is known, not the exact binary's
# build-run toolset. Never replace these Nones with a guessed 14.40/14.50.
CURRENT_CONTRACT = BuildContract(
    runner_sha256=RUNNER_SHA256,
    minimum_runtime_version=None,
    build_run_evidence_sha256=None,
    evidence_class="SOURCE_ONLY",
)


def build_contract_reasons(contract: BuildContract) -> list[str]:
    """Block host collection until real build-run provenance is reviewed."""
    reasons: list[str] = []
    if contract.runner_sha256 != RUNNER_SHA256:
        reasons.append("artifact_digest_not_approved")
    if (contract.evidence_class != "BUILD_RUN_VERIFIED"
            or not isinstance(contract.build_run_evidence_sha256, str)
            or SHA_RE.fullmatch(contract.build_run_evidence_sha256) is None):
        reasons.append("exact_build_run_provenance_unknown")
    try:
        if contract.minimum_runtime_version is None:
            raise Hold("unknown_required_runtime_minimum")
        parse_version(contract.minimum_runtime_version)
    except Hold:
        reasons.append("unknown_or_invalid_required_runtime_minimum")
    return reasons


def _require_version_parts(value: object) -> tuple[int, int, int, int]:
    if (not isinstance(value, list) or len(value) != 4
            or any(type(part) is not int or not 0 <= part <= MAX_VERSION_PART for part in value)):
        raise Hold("invalid_file_version")
    return tuple(value)  # type: ignore[return-value]


def _is_microsoft_signer(subject: object) -> bool:
    return isinstance(subject, str) and re.search(
        r"(?:^|,\s*)(?:CN|O)=Microsoft Corporation(?:,|$)", subject,
        re.IGNORECASE,
    ) is not None


def evaluate(
    evidence: object, contract: BuildContract = CURRENT_CONTRACT,
) -> dict[str, Any]:
    """Pure evaluator: fixture PASS means prerequisite only, never loader PASS.

    An accepted build contract must be compiled into independently reviewed
    source and tied to the exact runner bytes; caller-supplied JSON is not
    authoritative build provenance.
    """
    reasons = build_contract_reasons(contract)
    try:
        minimum = parse_version(contract.minimum_runtime_version)
    except Hold:
        minimum = None
        reasons.append("unknown_or_invalid_required_runtime_minimum")
    if not isinstance(evidence, dict):
        reasons.append("missing_host_evidence")
        return {"state": "HOLD", "reasons": reasons, "scope": "version_prerequisite_only"}
    if evidence.get("host") != "VERA-NODE-1" or evidence.get("architecture") != "AMD64":
        reasons.append("wrong_host_or_architecture")
    if evidence.get("process_bits") != 64:
        reasons.append("wrong_python_process_architecture")
    if evidence.get("runner_sha256") != contract.runner_sha256:
        reasons.append("runner_digest_mismatch")
    if evidence.get("app_local_runtime_files") != []:
        reasons.append("app_local_runtime_shadow_or_unknown")
    registry = evidence.get("registry")
    if (not isinstance(registry, dict) or type(registry.get("installed")) is not int
            or registry.get("installed") != 1 or registry.get("view") != "Registry64"):
        reasons.append("x64_redist_registry_unverified")
        installed_version = None
    else:
        try:
            installed_version = parse_registry_version(registry.get("version"))
        except Hold:
            installed_version = None
            reasons.append("redist_registry_version_invalid")
    files = evidence.get("runtime_files")
    if not isinstance(files, dict) or set(files) != set(RUNTIME_FILES):
        reasons.append("runtime_file_set_unknown")
        files = {}
    for name in RUNTIME_FILES:
        info = files.get(name)
        if not isinstance(info, dict):
            reasons.append("runtime_file_missing:" + name)
            continue
        if info.get("system32") is not True or info.get("pe_machine") != "AMD64":
            reasons.append("runtime_file_location_or_arch_invalid:" + name)
        if info.get("signature_status") != "Valid" or not _is_microsoft_signer(info.get("signer_subject")):
            reasons.append("runtime_file_signature_unverified:" + name)
        try:
            file_version = _require_version_parts(info.get("version_parts"))
        except Hold:
            reasons.append("runtime_file_version_invalid:" + name)
            continue
        if installed_version is not None and file_version != installed_version:
            reasons.append("registry_runtime_file_version_mismatch:" + name)
        if minimum is not None and file_version < minimum:
            reasons.append("runtime_file_too_old:" + name)
    if installed_version is not None and minimum is not None and installed_version < minimum:
        reasons.append("redist_registry_too_old")
    return {
        "state": "HOLD" if reasons else "ALLOW_VERSION_PREREQUISITE_ONLY",
        "reasons": reasons,
        "scope": "version_prerequisite_only",
    }


def _pe_machine(path: Path) -> str:
    """Parse only PE header architecture; does not execute or load the DLL."""
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            return "UNKNOWN"
        stream.seek(0x3C)
        offset_raw = stream.read(4)
        if len(offset_raw) != 4:
            return "UNKNOWN"
        offset = struct.unpack("<I", offset_raw)[0]
        if not 0x40 <= offset <= 1024 * 1024:
            return "UNKNOWN"
        stream.seek(offset)
        header = stream.read(26)
    if len(header) < 26 or header[:4] != b"PE\x00\x00":
        return "UNKNOWN"
    machine = struct.unpack_from("<H", header, 4)[0]
    opt_magic = struct.unpack_from("<H", header, 24)[0]
    return "AMD64" if machine == 0x8664 and opt_magic == 0x20B else "UNKNOWN"


def collect_windows_evidence() -> dict[str, Any]:
    """Read-only host inspection; no marker, child model process or mutation."""
    if os.name != "nt" or sys.maxsize <= 2**32 or platform.machine().upper() not in ("AMD64", "X86_64"):
        raise Hold("64_bit_windows_python_required")
    import hashlib
    import ctypes
    import socket
    import winreg

    if socket.gethostname().upper() != "VERA-NODE-1":
        raise Hold("wrong_host")
    key_path = r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0,
                        winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
        installed, installed_type = winreg.QueryValueEx(key, "Installed")
        version, version_type = winreg.QueryValueEx(key, "Version")
    if installed_type != winreg.REG_DWORD or version_type != winreg.REG_SZ:
        raise Hold("redist_registry_types_invalid")

    # Resolve the real OS directory through Win32, not a caller-controlled
    # SystemRoot/WINDIR environment variable that could point to a fake tree.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_system_directory = kernel32.GetSystemDirectoryW
    get_system_directory.argtypes = [ctypes.POINTER(ctypes.c_wchar), ctypes.c_uint]
    get_system_directory.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_system_directory(buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise Hold("cannot_resolve_native_system32")
    system32 = Path(buffer.value)
    if system32.name.casefold() != "system32":
        raise Hold("unexpected_native_system_directory")
    powershell = system32 / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    # Paths and command are constants; PowerShell does not run the candidate.
    script = r'''
$ErrorActionPreference = 'Stop'
$names = @('MSVCP140.dll','VCRUNTIME140.dll','VCRUNTIME140_1.dll')
$system = [Environment]::GetFolderPath('System')
$records = @{}
foreach ($name in $names) {
    $path = Join-Path $system $name
    $item = Get-Item -LiteralPath $path -ErrorAction Stop
    $sig = Get-AuthenticodeSignature -LiteralPath $path -ErrorAction Stop
    $subject = if ($null -eq $sig.SignerCertificate) { '' } else { $sig.SignerCertificate.Subject }
    $records[$name] = @{
        full_path = $item.FullName
        version_parts = @($item.VersionInfo.FileMajorPart, $item.VersionInfo.FileMinorPart,
                          $item.VersionInfo.FileBuildPart, $item.VersionInfo.FilePrivatePart)
        signature_status = $sig.Status.ToString()
        signer_subject = $subject
    }
}
ConvertTo-Json -InputObject @{ SystemDirectory = $system; Records = $records } -Depth 5 -Compress
'''
    completed = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=25, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    collected = json.loads(completed.stdout)
    if (not isinstance(collected, dict)
            or str(collected.get("SystemDirectory", "")).casefold() != str(system32).casefold()):
        raise Hold("system_directory_witness_mismatch")
    observed = collected.get("Records")
    if not isinstance(observed, dict) or set(observed) != set(RUNTIME_FILES):
        raise Hold("powershell_runtime_inventory_invalid")
    runtime_files: dict[str, Any] = {}
    for name in RUNTIME_FILES:
        file_path = system32 / name
        info = observed[name]
        if not isinstance(info, dict):
            raise Hold("powershell_runtime_record_invalid")
        runtime_files[name] = {
            "system32": str(file_path).casefold() == str(info.get("full_path", "")).casefold(),
            "pe_machine": _pe_machine(file_path),
            "version_parts": info.get("version_parts"),
            "signature_status": info.get("signature_status"),
            "signer_subject": info.get("signer_subject"),
        }
    local = [name for name in RUNTIME_FILES if (STAGED_BIN / name).exists()]
    digest = hashlib.sha256((STAGED_BIN / "llama-cli.exe").read_bytes()).hexdigest()
    return {
        "host": socket.gethostname().upper(), "architecture": "AMD64", "process_bits": 64,
        "runner_sha256": digest,
        "app_local_runtime_files": local,
        "registry": {"view": "Registry64", "installed": installed, "version": version},
        "runtime_files": runtime_files,
    }


def main(contract: BuildContract = CURRENT_CONTRACT) -> int:
    # Critical: today's SOURCE_ONLY contract is known to fail. Do not even
    # spawn a read-only PowerShell child before this precondition is settled.
    prior_reasons = build_contract_reasons(contract)
    if prior_reasons:
        print(json.dumps({"state": "HOLD", "scope": "version_prerequisite_only",
                          "reasons": prior_reasons}, sort_keys=True))
        return 3
    try:
        evidence = collect_windows_evidence()
    except (Hold, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"state": "HOLD", "scope": "version_prerequisite_only",
                          "reasons": ["host_collection_failed:" + type(exc).__name__]}))
        return 3
    result = evaluate(evidence, contract)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "ALLOW_VERSION_PREREQUISITE_ONLY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
