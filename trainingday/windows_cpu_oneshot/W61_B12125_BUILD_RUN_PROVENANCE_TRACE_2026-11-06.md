# W61 — b12125 archive-to-build-run trace (read-only)

Date: 2026-11-06. Scope: the exact upstream Windows x64 CPU ZIP staged on
VERA-NODE-1. This is provenance research, not Windows execution qualification.

## Stronger authenticated provenance link

1. The public `ggml-org/llama.cpp` b12125 release points to SLSA provenance
   attestation [63933469](https://github.com/ggml-org/llama.cpp/attestations/63933469).
   Subject `llama-b12125-bin-win-cpu-x64.zip` SHA-256 is
   `968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8`.
   The attestation signer is GitHub Actions **release-publish.yml**, run
   `41987435984`; the attestation itself is NOT the binary compiler job.
2. Independent GitHub connector read of the publish job log (job
   `127111419525`) shows `actions/download-artifact` fetched from originating
   **Release build run `41973638293`**, using source commit
   `7d516e25ead4653169236b3f90d13f7bd8629724`.
3. Independent GitHub workflow artifact metadata from build run
   `41973638293` confirms exact name `llama-bin-win-cpu-x64.zip`, artifact ID
   `13396480465`, size 24,730,921 bytes, and the **same SHA-256** as the
   attested published `llama-b12125-bin-win-cpu-x64.zip`. The staged VN1 ZIP
   was independently rehashed to this digest in W56/W57.
4. Exact upstream `b12125` `release.yml` source (Git blob
   `acda37fb9cfc91907110b243b207e74f5530704b`) defines CPU x64 Windows
   build on `windows-2025-vs2026`, invokes Visual Studio 18 `vcvarsall.bat`,
   and configures the LLVM CMake toolchain. This is strong source evidence
   for intended environment, **not** a readback of compiler versions actually
   used in the build job.

Public pointers:

- https://github.com/ggml-org/llama.cpp/attestations/63933469
- https://github.com/ggml-org/llama.cpp/actions/runs/41987435984
- https://github.com/ggml-org/llama.cpp/actions/runs/41973638293
- https://github.com/ggml-org/llama.cpp/releases/tag/b12125

## Independent VN1 static import evidence

Read-only native PE parsing of the staged, hash-pinned `llama-cli.exe` and
seven companion DLLs by `/root/vc_host_validation` showed:

- CLI runner `llama-cli.exe`: 9,728 bytes, SHA-256
  `02042615249dd4852f47cc2d4dd5caabba91766b5161dd4642ed6272f788172e`.
- `llama-cli.exe` directly imports `VCRUNTIME140.dll` and UCRT; companion
  `llama-cli-impl.dll` imports `MSVCP140.dll`/`VCRUNTIME140.dll`;
  `llama.dll` and `llama-common.dll` additionally import `VCRUNTIME140_1.dll`.
- Linker header 11.2 across the inspected files is NOT a reliable MSVC
  toolset identifier. Relevant EXE/DLL files have no useful FileVersion or
  ProductVersion resources; eight inspected upstream binaries were
  Authenticode NotSigned. The Microsoft Windows System32 runtime DLLs were
  separately observed as signed Valid at 14.32.31332.0.

These observations prove an import dependency on Microsoft C++ runtime DLLs.
They do not prove the **minimum required** runtime version, effective DLL
resolution, crash cause, or successful loader initialization.

## Unresolved provenance gap — HOLD

The available GitHub `github_fetch_workflow_run_jobs` connector exposes only
the first 30 jobs of build run `41973638293`; the exact `windows-cpu x64`
compiler job is not present on that page. The connector has no pagination
argument. The present web search/public run view did not expose its log.
One **different job** in that same run, `windows (vulkan, x64)` job
`127103613066`, logged image `windows-2025-vs2026` version
`20261030.294.1`, MSVC compiler `19.51.36260.0`, and VS18 `cl.exe` under
`VC/Tools/MSVC/14.51.36231`. This corroborates a newer build-runner
environment but is NOT the target CPU x64 job's toolset readback and must
not establish its minimum. The target uses the LLVM toolchain.
There is therefore **no verified exact build-run compiler/toolset version or
minimum redist version**. Do not replace W60's `CURRENT_CONTRACT` Nones with
an inferred 14.40/14.50. Source candidate remains HOLD.

Microsoft states the installed v14 redistributable should be at least as
recent as the build tools and that the latest supports Windows 10, but this
general compatibility rule does not turn workflow source into proof of the
exact compiled artifact's required version:
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

> **Hostile review:** Artifact custody from attested release to source build
> run survives these checks. Binary build-toolset identification does **not**.
> A version embedded in a linker header, source workflow, or adjacent job log
> could be weaker than an exact CPU x64 build job compiler/version readback.

No host writes, system runtime changes, model execution, or consumed-nonce
changes were made during W61 provenance investigation.
