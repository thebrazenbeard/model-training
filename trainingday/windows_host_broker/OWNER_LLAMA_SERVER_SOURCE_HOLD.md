# Lappy llama-server owner lease — SOURCE CANDIDATE / PHYSICAL_RUN HOLD

This is a **separate local-inference design**, not a TrainingDay optimizer or
data-rights grant. Its source lives beside the Windows native Job fixture for
review convenience, but the installed SYSTEM-owned TrainingDay broker is
**unchanged**, and Python `physical_launch_authority.py` stays unconditional
`HOLD`. Nothing here installs a service, imports a model, opens a port, or
authorizes Qwen inference on Lappy.

## Exact source candidate

- Host: `LAPPY`, native Windows owner-attended session (not WSL for this lane).
- Model: `D:\VERA\models\gguf\model_q5_k_s.gguf`, exactly
  `3,072,993,056` bytes, SHA-256
  `15150f534dc90ee15c82320ccc014463db7985205a7161228cff89b925fd1216`.
  The hash was measured on Lappy in a low-priority read-only owner job; it
  matches the provided primary Hugging Face LFS pointer. **R13 adapter
  compatibility and GGUF tokenizer behavior remain unproven.** No LoRA flag.
- Executable: `llama-server.exe` only, *absolute canonical owner-pinned path
  and SHA-256 to be determined*. Do not assume the historical VN1 b12098
  binary works on Lappy or copy anything until provenance/runtime qualifies.
- Fixed ordered arguments: `--model <pinned Q5> --host 127.0.0.1 --port 11435
  --n-gpu-layers 0 --threads 2 --parallel 1 --ctx-size 2048 --no-webui
  --offline`. These are source candidates based on upstream llama.cpp server
  documentation; an exact installed binary must accept them before launch.
- Isolation proposal: one Windows JobObject with 5 GiB *maximum committed
  memory*, 25% CPU hard cap, `KILL_ON_JOB_CLOSE`, child created suspended,
  attached then resumed, monitored until observed exit. The tested harmless
  Job fixture in Draft PR #342 supports ABI/limit readback mechanics only.
- Owner lifetime: 180-second maximum startup-and-conversation test lease, no
  automatic renewal or restart; explicit owner cancellation and process-handle
  exit observation. The planned bridge at `127.0.0.1:11439` allows 75 seconds
  per request, with the UI gateway at `127.0.0.1:8765` limited to 90 seconds.

## Fail-closed host-native admission before any physical launch

1. Independently authenticate Windows owner SID/session and host `LAPPY` from
   OS tokens, not submitted JSON fields, remembered identity or an HTTP header.
2. The owner-local policy and executable directory must have verified ACLs,
   realpath/reparse checks and stable handle identity. Rehash the **actual**
   protected executable and pinned GGUF under stable handles; no path-swap
   between verification and create. Unknown file identity means HOLD.
3. Measure CPU/disk process contention and physical memory recently: at least
   5 GiB Job cap **plus 2 GiB host headroom**, CPU at most 20%, disk at most
   10%, telemetry age no more than 15 seconds. Reject active competing
   inference/training work; host capacity is an observation, not a reservation.
4. Refuse if TCP `127.0.0.1:11435` is occupied; after launching, independently
   verify the loopback listener is owned by the retained child process. A
   check-then-release TCP bind is **not** an exclusive port lease. No broad
   `0.0.0.0` binding and no automatic process killing to recover the port.
5. Create the Job and configure/query the cap in the kernel **before**
   creating the new child suspended. Assign, check membership, then resume.
   Refuse nested/foreign Job interactions that prevent limit enforcement.
6. An owner-held process and Job handle must persist for the entire bounded
   test and terminate all children on cancellation, owner exit or expiry.
   Verify termination using the original process handle, not a PID lookup.
7. Only the local authenticated bridge may use the model endpoint; health and
   model-ID readback are required before chat. These are behavioral checks,
   not host custody or model/base-lineage proof.

## Source tests vs physical approval

`OwnerLlamaServerLaunchCandidate.cs` accepts **synthetic, untrusted** evidence
solely for negative policy regressions. Even a perfectly matching synthetic
fixture always reports `PHYSICAL_RUN_HOLD`; it has no `CreateProcess`, network
or model-loading entrypoint. Tests include wrong host/SID, forgery, wrong
binary/model hashes, path traversal, altered GPU/network flags, low memory,
busy CPU/disk, occupied port, concurrent inference, stale telemetry and
missing owner-path custody.

This source layer is not an owner-host witness, executable ACL verifier,
exclusive process/port lease or native runtime test. A reviewed production
launcher, protected host policy and explicit owner-approved physical test
window are still required. The **correct decision remains NO-GO** until the
exact target host proves all of the gates above.

Primary option reference:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
