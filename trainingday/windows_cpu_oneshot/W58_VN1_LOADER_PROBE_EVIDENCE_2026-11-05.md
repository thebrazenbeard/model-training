# W58 VN1 loader-only probe — physical evidence (2026-11-05)

**Disposition: LOADER FAILED / HOLD. No automatic retry.** This record concerns
one authorized `llama-cli.exe --version` process under the corrected Windows
JobObject supervisor. It does not prove Q4 loading, inference, Vera identity,
training quality, or official Qwen derivation.

## Exact subject and authority

- Repository: `thebrazenbeard/model-training`, Draft PR #347.
- Reviewed parent head: `5712ddfa49c88556bff5e37f3ec66af39f715103`.
- Loader-only wrapper commit: `5d78d375edacff3e5c753ee8f6c44f2f39021ed8`.
- Wrapper: `trainingday/windows_cpu_oneshot/supervisor/loader_probe_v2.py`;
  local, GitHub and VN1 bytes were compared, SHA-256
  `eba5517de12468cb8b45751b770305064e3fe1c316f2b874e56862d10efb7946`.
- Host and principal: `VERA-NODE-1`, `VERA-NODE-1\Vera`.
- Binary: `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2\bin\llama-cli.exe`.
- Fixed arguments: **only** `--version`; no GGUF path, inference prompt or training arguments.
- Stage receipt SHA-256: `8ebffb4a8ad78acc6742e5b1972ef3c7da1c56e397828ce932233d1ba1b3c408`.
- All 23 staged EXE/DLL bytes were independently checked in W57 and re-pinned
  by the loader probe; the initial 19-file v1 policy was not used for launch.

## Bounded physical process

- Windows suspended process assigned to Job before resume.
- Job caps: 512 MiB job memory, 20% CPU hard cap, eight-second wall bound;
  repeated Ollama/host occupancy guard, one-child process cap and kill-on-close.
- Single-use consumed marker, nonce `2c9edff0f2254218bfd00695f96b104f`.
- Child PID **15448**; supervisor wall measurement **0.731 s**.
- **Exit code 3221225477 = `0xC0000005`**; `reason=CHILD_EXITED`,
  `child_handle_signaled=true`, `job_active_processes_after=0`,
  `accounting_polls=2`, accounting wait **0.05047 s**.
- stdout **0 bytes**; stderr **0 bytes**; both SHA-256 equal the empty-file hash
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `version_reported=false`; receipt `state=FAIL`.
- Receipt path (preserved):
  `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2\state\loader-probe-v1\receipt.json`.
  Receipt SHA-256 `cb8d710f0d3d4399ab02e05f3534dde3a7cd568d637ebd59c9bef9e72a851aed`.
- Separate readback: PID 15448 absent, Ollama PID 5300 still alive,
  `/api/ps` loaded models **0**. Trial state contains consumed marker, empty
  `loader.out` and `loader.err`, and final receipt. The leaf inherits only
  `SYSTEM` and `VERA-NODE-1\Vera` ACEs from a protected parent; leaf owner is
  Administrators and inheritance itself is enabled (do not claim its own ACL is protected).

## Windows application crash evidence (independent from wrapper receipt)

- Application log Event ID **1000**, provider `Application Error`.
- Faulting app: `llama-cli.exe`, process `0x3c58` (**15448**), version 0.0.0.0.
- Faulting module: `C:\Windows\SYSTEM32\MSVCP140.dll` version **14.32.31332.0**,
  fault offset `0x00000000000132f8`, exception `0xC0000005`.
- Report ID `1db652bf-9ef5-4996-bc3e-fcf9fe8eafad`.
- System: Windows 10 IoT Enterprise LTSC, x64, build **19044**;
  Intel Core i7-6770HQ.
- VC++ x64 runtime registry version `v14.32.31332.00`, `Installed=1`.
  `MSVCP140.dll` Authenticode status valid, signer Microsoft Corporation.
  System32 MSVCP140, VCRUNTIME140 and VCRUNTIME140_1 report version 14.32.31332.0.
- MSVCP140 SHA-256:
  `72fe4598f0b75d31ce2dc621e8ef161338c6450bb017cd06895745690603729e`.

## Evidence judgment and next boundary

**FACT:** Windows access violation during CLI startup/version reporting. The
faulting module is not proof that its DLL itself is defective: caller memory
corruption, incompatible ABI, a native dependency, or another early-init fault
could yield the same crash. The installed VC++ runtime may predate the
compiler used for the upstream release, but build-tool version is not yet
verified. The absence of model arguments does **not** formally prove that no
GGUF file was accessed (no filesystem tracer was installed).

> **Hostile review:** The missing four DLLs from W56 are now present and pinned,
> so their prior omission must not be promoted as the cause of this new access
> violation. A clean JobObject reap proves custody, **not** CLI loader success.
> The next safe action is read-only WER/import-graph inspection; installing a
> new redistributable, swapping DLLs, or launching another binary requires
> separate effect authorization.

The prior model inference plan remains consumed. This newer loader probe is
also consumed. Do not replay either nonce. Do not merge/deploy the candidate
as model qualified. No shared services were stopped or restarted.
