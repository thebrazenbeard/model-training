# W60 — VN1 MSVC compatibility preflight (SOURCE ONLY)

Date: 2026-11-06. Subject: `thebrazenbeard/model-training` Draft PR #347,
based on initial W59 head `3095f878efc6af68e3ea59ea0eee3e5f79ff79e5`.

## Status: HOLD; no new native execution

`supervisor/msvc_runtime_guard.py` is a standalone, read-only Windows
prerequisite checker. It **does not** mint a one-use nonce, write a trial state,
alter host runtimes, launch `llama-cli.exe`, load a model, or authorize a run.
The historical `loader_probe_v2.py` and both consumed nonces remain unchanged.
The new source is not yet deployed to VN1 or wired into a new runner; a
future approved probe must call the gate **before** state creation, nonce
consumption, or native process launch, then independently revalidate race and
DLL-resolution assumptions.

The exact b12125 artifact's build-run compiler/toolset version and its
corresponding minimum supported VC++ Redistributable version have not been
independently tied to the pinned executable bytes. A source workflow mentioning
Visual Studio 18/LLVM is not such an attestation. The shipped
`CURRENT_CONTRACT` therefore has an unknown minimum and `SOURCE_ONLY`
provenance, and **unconditionally returns HOLD** even if host evidence looks
new enough. No speculative minimum (e.g. 14.40 or 14.50) has been installed.
The command's entrypoint checks this incomplete contract **before** invoking
the PowerShell-based read-only collector, so the current command exits HOLD
without launching even a collector subprocess.

## Host facts and limits

Independent read-only VN1 check by subagent `/root/vc_host_validation`:

- `VERA-NODE-1`, installed x64 runtime registry `v14.32.31332.00`.
- Actual System32 `MSVCP140.dll`, `VCRUNTIME140.dll`, and
  `VCRUNTIME140_1.dll`: file versions `14.32.31332.0` and valid Authenticode
  signatures. This witness did **not** exercise the new Python collector.
- No observed llama/KoboldCPP/trainer workers; Ollama daemon present with
  empty GET `/api/ps` model inventory.
- Previous native `llama-cli --version` one-use probe exited with
  `0xC0000005` in `MSVCP140.dll` (W59). A runtime mismatch is plausible,
  not causally proven; no native stack dump was recovered.

The collector, when separately installed and invoked read-only on VN1, is
designed to examine the 64-bit registry view, three actual System32 DLL file
versions, PE x64 architecture, valid Microsoft Authenticode signer, exact
runner SHA-256, and app-local VC DLL shadows. Collection failures HOLD. The
collector cannot establish actual native loader resolution, protect against
privileged host tampering, or eliminate the time-of-check/time-of-use race.

## Offline tests and evidence class

On the isolated Linux workspace using Python 3.12:

```text
python -m unittest -v test_msvc_runtime_guard.py
20 tests, all PASS (offline synthetic fixtures only)
python -m py_compile msvc_runtime_guard.py test_msvc_runtime_guard.py
PASS
python msvc_runtime_guard.py
HOLD / exit code 3, no collector invoked
```

Tests cover strict numeric versions (14.9 < 14.50), registry zero padding,
unknown provenance, old runtime, missing/corrupt registry or DLLs, signature,
architecture, location, app-local shadow, runner hash, and collector failure.
The default entrypoint test asserts the read-only collector is not called
when exact build provenance is unknown.
The one positive test uses a **synthetic, non-authoritative**
`BUILD_RUN_VERIFIED` fixture solely to exercise boolean logic; it does not
certify the actual b12125 build or any real host. Source tests are neither
Windows integration testing nor native inference qualification.

Independent reviewers `/root/vc_guard_design` and `/root/vc_hostile_review`
reviewed the exact W59 source head and recommended this fail-closed design;
their subsequent review of this new implementation is a separate step.

> **Hostile challenge:** A crash in Microsoft C++ runtime code can still
> originate in application memory corruption. An apparently valid version
> preflight may race a loader DLL change and cannot prove effective resolution.
> Without exact build-run provenance, an invented minimum would create a
> false PASS and must be rejected.

## Exit/acceptance

1. Obtain independently verifiable exact binary build-run provenance and
   minimum redist; review and pin in source, or build a qualified alternative.
2. Independently review the Python/PowerShell read-only collector, run it on
   VN1 under a separately bounded check, and confirm source findings.
3. Only after a fresh scoped authorization, prepare a **new**, independently
   reviewed one-use loader plan that invokes a prelaunch gate before any
   irreversible marker. Never replay W56 or W58 nonces.
4. Treat a version preflight PASS as prerequisite evidence only; a separate
   native loader and model inference result is required for runtime claims.

No VC++ redist update, no new native launch, no service restart, no model
training and no monetization effect occurred in W60 source work.
