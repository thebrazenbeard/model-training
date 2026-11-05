# VN1 one-shot Q4 baseline supervisor — candidate only

Status: **SOURCE CANDIDATE / NOT INSTALLED / NO PHYSICAL QUALIFICATION**.

This is a disposable local Python 3.13 process owner for **one public,
unverified third-party Q4, base-only** conversation smoke. It is not a
TRAININGDAY training broker, never authorizes private data, does not load R16
LoRA, cannot certify Qwen-derived quantization provenance, and does not remove
any existing `physical_launch_authority.py` HOLD.

## Execution boundary

* Windows host/account are exactly `VERA-NODE-1` / `Vera`.
* No network listener, server, dynamic command line, arbitrary model path,
  user-supplied prompt, GPU layers, private chat, or training operation.
* Harmless `--self-test` writes its lock, outputs and receipt **only** under
  `C:\Users\Vera\AppData\Local\Temp\w56-q4-selftest`, never under the
  model trial leaf. The staging writer may therefore create
  `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot` atomically afterward.
* Pinned runner path
  `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot\bin\llama-cli.exe`;
  pinned model path
  `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot\models\Qwen3.5-4B-Q4_K_M.gguf`.
* Runner SHA-256 must match both the plan and an independently approved digest
  compiled into installed `policy.py` as `REVIEWED_RUNNER_SHA256`; its source
  default is deliberately **blank/HOLD** until qualified. The exact native
  `.exe/.dll/.pyd` bundle must also match the independently reviewed
  `REVIEWED_RUNTIME_FILES_SHA256` map; this is likewise empty/HOLD by default.
  Each bundle file handle is pinned before and until the child exits. The model
  SHA-256 must match the publisher's exact advertised digest
  `b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca`.
  A checksum verifies bytes, **not** derivation from official Qwen weights.
* Files are opened with Windows read sharing only (no write/delete sharing)
  and hashed through those retained OS file handles; leaf and ancestor
  reparse points are forbidden.
* Preflight: >= max(5 GiB, capped job commit + 1.5 GiB) available physical
  memory, no known running model executables, <=60% host CPU over 1 second.
  The **harmless** fixed `--self-test` alone ignores a resident `ollama.exe`
  daemon/UI process name while still rejecting `ollama_llama_server.exe` and
  other model/trainer executables. `--run` continues to reject **any**
  `ollama.exe` or other listed workload; no daemon is stopped or restarted.
* Both modes share one kernel file-region lock in the self-test temporary
  directory; the Q4 run cannot race a concurrent self-test.
* Launch uses Win32 `CreateProcessW(CREATE_SUSPENDED)`; checks Job membership
  **before** resume. Job has `ACTIVE_PROCESS=1`, `KILL_ON_JOB_CLOSE`,
  `JOB_MEMORY<=4 GiB` and Windows CPU hard cap <=25%; child has below-normal
  priority, `-ngl 0`, `-t 2`, `-c <=1024`, `-n <=64`, seed0/temp0.
* Job commit is **not** a complete RSS/file-mapping limit. A separate physical
  free-RAM watchdog stops the job below 2 GiB, and a process working-set
  monitor stops it above the job memory limit. No host-wide CPU lease or
  guaranteed interference exclusion is claimed.
* Wall-time <=180 seconds, no replay for a consumed plan ID, one cross-process
  Windows file lock slot. Crash/UNKNOWN consumes plan; manual review needed.
* Stdout/stderr and final JSON receipt stay in the private scoped state dir.
  New output files are exclusive-create and cannot overwrite prior attempts.

## Required independent approval and Windows verification

1. Review `policy.py`, `windows_job.py`, `supervisor.py`, and tests against the
   exact `model-training` source contracts and current VN1 OS state.
2. Establish private owner-only ACLs for both staging directories, the Python
   script and state directory. This package **does not install an ACL** or
   claim its source can authenticate its owner by comparing a username.
3. Pin/qualify the exact Windows `llama-cli.exe` binary, provider, SHA-256,
   and supported CLI flags. An arbitrary `llama-cli.exe` renamed from another
   binary must not be trusted merely because the path matches.
4. Install Python3.13 source in a separately approved temporary source
   directory outside the `w56-q4-oneshot` leaf, then run exactly
   `py -3.13 supervisor.py --self-test` under the bounded host window. Verify
   returned `PASS`, child PID exited, process tree reaped, CPU cap and memory
   cap read back from Windows, and no other workloads disrupted. This source
   does not substitute for those independent physical observations.
5. Stage the publisher-pin Q4 atomically into its still-nonexistent trial leaf
   only after the reviewer confirms free disk,
   publisher hash, published license, and explicit third-party lineage label.
6. Materialize a single `state/approved_q4_baseline_plan.json` with a unique
   32-lowercase-hex `plan_id`, exact pinned runner SHA, fixed model SHA, and
   approved caps. Never change plan ID and retry after ambiguous failure.
7. Run exactly `py -3.13 supervisor.py --run` once, inspect `receipt-*.json`,
   output logs, actual process exit and resource telemetry before any claim.

Plan fields are fixed: `schema`, `host`, `plan_id`, `executable`,
`executable_sha256`, `model`, `model_sha256`, `context`, `output_tokens`,
`threads`, `wall_seconds`, `job_memory_bytes`, `cpu_rate_percent`. No other
fields or per-turn arbitrary prompts are allowed. `policy.py` has the exact
schema and limits.

## Current tests

`python -m unittest discover -s tests -v` validates plan limits and refusal
paths on Linux or Windows. It **does not execute any Windows native API**.

> Hostile review: Kernel Job custody is a necessary containment mechanism,
> not a full operating-system scheduler reservation, an immutable source
> attestation, or proof of trust against an attacker with the same Windows
> account. Do not claim an exclusive host lease or model authenticity.
