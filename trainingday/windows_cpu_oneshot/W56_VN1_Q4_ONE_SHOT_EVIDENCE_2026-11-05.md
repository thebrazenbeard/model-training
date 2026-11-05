# VN1 Q4 one-shot — physical evidence and failed first launch

Evidence date: 2026-11-05 UTC. This is not a TRAININGDAY training qualification.

## Scope and authority

- User authorized one bounded CPU-only Windows JobObject experiment on VN1, with
  no shared service stop/restart, no paid compute, no inference retries after an
  ambiguous consumed plan.
- GitHub review artifact: `thebrazenbeard/model-training` Draft PR #347. Branch
  `work/w56-vn1-cpu-q4-oneshot-20261105`, initial launch revision
  `4d8cd5d1b946291eeccabccfcee04ce80b5d277f`.
- Actual host: `VERA-NODE-1`, Windows principal `VERA-NODE-1\Vera`.
- Local staging leaf: `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot`.
- Deployed supervisor files independently SHA-256 matched the source reviewed
  before launch: `policy.py d7a65c7f3b1a6d6...`, `supervisor.py
  5729e3cefb2ab7cb...`, `windows_job.py dccdc8dfa215b20f...`.

## Artifact integrity — real VN1 readback

| Artifact | On-disk bytes | SHA-256 | Evidence scope |
|---|---:|---|---|
| Third-party twoody Qwen3.5-4B-Q4_K_M.gguf | 2,783,446,880 | `b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca` | Publisher bytes, NOT verified official-Qwen derivation |
| Official llama.cpp b12125 CPU x64 ZIP | 24,730,921 | `968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8` | Same digest as published GitHub release attestation |
| Extracted llama-cli.exe | 9,728 | `02042615249dd4852f47cc2d4dd5caabba91766b5161dd4642ed6272f788172e` | Exact staged binary; its loader dependencies were incomplete |

Nineteen staged EXE/DLL hashes matched the stager receipt on independent
`Get-FileHash` readback. The official ZIP actually contains 51 distinct native
EXE/DLL files, and the stager's allowlist extracted only 19. Four missing
shared-library candidates essential to the CLI dependency closure were found
in the *already verified* ZIP (no extra download):

| Missing from staged bin | Bytes in ZIP | SHA-256 streamed from ZIP |
|---|---:|---|
| `llama-cli-impl.dll` | 3,375,104 | `e60450e7897b025c4ce1eba615dcf66e2d95cea90d022aa161d31ef1d4f13eb1` |
| `llama-server-impl.dll` | 10,104,320 | `5cb7e275a9a5c3a697c1a6a95da866e28f8ccd379f5b7ba1a3df367ffbe53569` |
| `llama-common.dll` | 8,360,960 | `9f1e8a5714baabd1fb72187698f1ec358793d86c6be308681fc1dd49b757cf54` |
| `mtmd.dll` | 2,531,840 | `5713c5478ba954848ae62ed02ecdc49a0524612646f887406483503af95bea71` |

**Important:** ZIP membership and upstream CMake linkage corroborate a broken
staging allowlist. The exact Windows PE import closure and prior raw child exit
code were not captured, so DLL-missing remains a likely startup explanation,
not a conclusively observed exit-code diagnosis.

## Native containment validation

- Revised policy: 12/12 Windows Python offline unit tests PASS.
- Harmless fixed Windows Job selftest: trial
  `d684127ab28f40a3af7e758b7830b0f4`, child PID 14284, exit code 90,
  `SELFTEST_FORCED_STOP`, Job `ActiveProcesses=0`, original process handle
  signaled; independent readback showed child gone, Ollama PID 5300 alive.
- Trial staging directories: owner `VERA-NODE-1\Vera`, protected DACLs, no
  tested model or runner reparse-point files. At preflight Ollama `/api/ps`
  reported no loaded models. No exclusive host-wide lease was established.

## First model attempt: aborted/unknown, NO REPLAY

- One-use plan nonce: `f48318bf1c3645cba6b6cd6920f13635`.
- Plan SHA-256: `ef163e4655176bfee8a4b5bc831130e3b473174119d5f0e6a874efd9d6882c27`.
- `--run` consumed the nonce; `receipt-f48318bf1c3645cba6b6cd6920f13635.json`
  states `UNKNOWN_NO_REPLAY`, failure_type `Win32Hold`, 0.082-second elapsed.
- Console error: `Job child still active after signaled process exit`.
- Both `llama-...out` and `llama-...err` were 0 bytes. The child's numeric
  exit status was lost to the accounting exception.
- Subsequent independent host probe found **no llama-cli process**, Ollama
  PID 5300 still alive, roughly 7.24 GB free memory; no relevant Application
  Event Log entries during the bounded launch window.
- No meaningful model inference, conversational response, persona retention,
  adapter, or official-Qwen lineage has been demonstrated.
- The consumed plan stays consumed. **Do not relaunch or generate a replacement
  run nonce without fresh specific authorization.**

## Source repairs after the aborted attempt (not physical qualifications)

1. Win32 Job accounting must confirm signaled child and latch exit code, close
   process/thread handles, then poll Job accounting to zero with a bounded
   deadline; the original code queried while retaining process references.
   Microsoft documents the reference-release dependency for `ActiveProcesses`:
   https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information
2. Add the four verified shared-library entries, enforce mandatory presence,
   and stage to a new isolated leaf only. Preserve the original 19-file bin
   and failure evidence.
3. Perform a harmless fast-exit JobObject regression (no model) under the
   corrected accounting source; capture numeric exit code and Job drain.
4. Before any second model authorization, review the exact PE dependency
   closure, pin every selected native EXE/DLL SHA-256, qualify loader/CLI
   behavior in isolation, and inspect no-service-change/host resource gates.

## Evidence taxonomy

`SOURCE_INSPECTION`: two potential defects found. `TEST`: 12 offline Windows
policy tests passed. `RUNTIME_PROBE`: staging SHA verification and harmless
Job selftest passed. `RUNTIME_PROBE`: first model launch failed/unknown.
`MODEL_REVIEW`: independent reviewers challenged process accounting, dependency
closure, Ollama races, and lack of OS filesystem/network sandboxing. None of
these sources is formal verification or model personality qualification.

## Follow-up harmless Windows regression (2026-11-05 UTC)

After the failed model attempt, the independent reviewer conditionally approved
`quick_exit_probe.py` (SHA-256 `e783ddac111e4d2b32af3bab96950a681e5d0ce63b6e1d2b82e59deba93b8e8b`)
for one harmless Python child test using the repaired Job source. Transferred
source was independently rehashed on VN1; 17/17 Windows offline tests passed.

The native probe returned `PASS`: nonce `9f9b7e66326a4df5a89e67758b09bba4`,
child PID 220, exit code 0, `CHILD_EXITED`, process handle signaled,
`ActiveProcesses=0`, two accounting polls, reconciliation wait 0.05058 seconds.
The separate host check confirmed PID220 gone and Ollama PID5300 still alive.
The original failed Q4 receipt was unchanged (SHA-256
`cf4a47e8a52a877bf574d454773a53d2607ff4eed977416be4885a243fa3ddc6`),
and its nonce remains consumed. This verifies the fast-exit accounting fix
on actual Windows, not that the GGUF runner loads or produces text.
