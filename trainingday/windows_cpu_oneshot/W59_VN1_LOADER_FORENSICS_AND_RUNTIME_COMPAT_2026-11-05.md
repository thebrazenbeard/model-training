# W59 VN1 loader forensics — read-only follow-up (2026-11-05)

Status: **HOLD / probable toolset-runtime incompatibility, not proven root cause**.
This follows the consumed W58 `llama-cli.exe --version` probe in Draft PR #347.
No native model executable was run again, no runtime was installed, no DLL was
swapped, and no services were started/stopped during W59.

## Host readback and missing crash context

- Windows Application Error Event **1000** at `2026-11-05T23:27:09.507Z`:
  `llama-cli.exe` PID 15448, exception `0xC0000005`, faulting module
  `MSVCP140.dll` `14.32.31332.0`, offset `0x132f8`, report ID
  `1db652bf-9ef5-4996-bc3e-fcf9fe8eafad`.
- No matching Event 1001/1002 in the bounded crash window and no matching
  existing WER reports in ProgramData `ReportArchive`, `ReportQueue`, `Temp`,
  or the user-level locations checked. `%LOCALAPPDATA%\CrashDumps` and
  configured `LocalDumps` keys were absent. No stack/dump was available.
- WER service was stopped and set to Manual. That alone does not establish
  reporting is disabled; we did **not** change its state or WER settings.
- VN1: Windows 10 IoT Enterprise LTSC 19044, x64, Intel i7-6770HQ.
  Installed VC++ x64 runtime registry version `14.32.31332.00`.

## Native dependency inspection

There was no preinstalled `dumpbin`, `llvm-readobj`, `llvm-objdump`, `objdump`,
`sigcheck`, or Python `pefile`. A bounded read-only PE import-table parser using
existing PowerShell/.NET capabilities inspected the exact staged v2 files:

| Staged PE file | Selected direct import DLLs |
| --- | --- |
| `llama-cli.exe` (9,728 B) | `llama-cli-impl.dll`, `VCRUNTIME140.dll` |
| `llama-cli-impl.dll` | `llama-server-impl.dll`, `llama-common.dll`, `llama.dll`, **`MSVCP140.dll`**, `VCRUNTIME140.dll` |
| `llama-common.dll` | `llama.dll`, `ggml.dll`, `ggml-base.dll`, **`MSVCP140.dll`**, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` |
| `llama.dll` | `ggml.dll`, `ggml-base.dll`, **`MSVCP140.dll`**, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` |
| `ggml.dll` | `ggml-base.dll`, **`MSVCP140.dll`**, `VCRUNTIME140.dll` |
| `ggml-base.dll` | `libomp.dll`, **`MSVCP140.dll`**, `VCRUNTIME140.dll` |
| `ggml-cpu-x64.dll` | `ggml-base.dll`, `libomp.dll`, **`MSVCP140.dll`**, `VCRUNTIME140.dll` |
| `libomp.dll` | `KERNEL32.dll`, `PSAPI.dll` |

No delay imports were found in the eight PE files examined. This establishes
that the four W56-omitted application DLLs included genuine direct imports,
but those files are **present and pinned** in W57/v2. Import presence does not
prove which function crashed or which DLL caused corrupt state.

## New exact upstream source evidence

1. `ggml-org/llama.cpp` [issue #13498](https://github.com/ggml-org/llama.cpp/issues/13498)
   reports an earlier Windows `llama-cli` build crashing with `0xC0000005`
   in `MSVCP140!mtx_do_lock` during `ggml_init`/CPU backend initialization.
   The reporter says a newer official MSVC Redistributable eliminated it.
   Different release and host; **analogue, not our stack trace**.
2. The [Microsoft STL releases](https://github.com/microsoft/STL/releases)
   document `constexpr std::mutex` constructor changes, potential null
   dereferences when mixed against older MSVC runtimes, and the requirement
   that the redistributable be at least as new as the newest build toolset
   used. `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` is described as a build-time
   compatibility escape hatch, not a system runtime repair.
3. Microsoft [supported Redistributable guidance](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)
   requires the runtime installed on the target computer to be at least as
   recent as the MSVC build tools used for the application.
4. The exact [llama.cpp b12125 release workflow](https://github.com/ggml-org/llama.cpp/blob/b12125/.github/workflows/release.yml)
   (Git blob `acda37fb9cfc91907110b243b207e74f5530704b`) defines its
   Windows CPU job on `windows-2025-vs2026`, invoking Visual Studio **18**
   `vcvarsall.bat` before the LLVM CMake toolchain build and packaging
   `llama-bin-win-cpu-x64.zip`. This is source/CI workflow provenance,
   **not** a captured exact build-run compiler version or formal proof of
   the binary's toolchain.

## Evidence classification and conclusion

- **RUNTIME_PROBE:** W58 produced an access violation and failed version
  report; Job cleanup passed. The old one-use nonce remains consumed.
- **SYSTEM_READBACK:** VN1 runs VC++ runtime `14.32`, which predates the
  toolset family specified by b12125's VS2026 workflow.
- **SOURCE:** Microsoft documents an exact compatibility mechanism and
  upstream llama.cpp documents a near-identical symptom cured by updating
  the redistributable.
- **UNKNOWN:** There is no VN1 stack and no exact build-run toolset version;
  causal attribution remains unproven.

> **Hostile falsifier:** It could still be a separate native initialization,
> memory corruption, or ABI defect that happens to fault in the same runtime
> DLL. Do not conclude MSVCP140 itself is corrupt merely because it faulted.
> Do not create a new run nonce or silently retry the binary.

## Proposed next bounded gate (not performed)

1. Bind an explicit required MSVC runtime version to verified release-build
   metadata; if that metadata is unavailable, HOLD rather than inventing a
   generic `14.40` minimum for this VS2026-based artifact.
2. Add a source-level read-only preflight comparing x64 Redistributable
   registry version and actual System32 `MSVCP140`/`VCRUNTIME140*` versions.
   Mismatch => HOLD before any process creation.
3. Any system-wide Microsoft VC++ update/repair, DLL swap, privately rebuilt
   alternative, or **new** single-use loader launch is a distinct effect that
   needs authorization and a fresh reviewed plan. No merge/deployment now.
