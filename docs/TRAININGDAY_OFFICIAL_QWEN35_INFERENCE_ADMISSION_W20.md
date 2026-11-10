# VN1-first official Qwen3.5-4B inference — source-only W20 admission kit

**Status:** source implementation and synthetic tests only. No official weights
downloaded, converted, loaded, or benchmarked; no host broker grant. This kit
does not supersede `trainingday/physical_launch_authority.py`, whose physical
TrainingDay launch HOLD remains unconditional. GGUF inference is distinct from
the BF16/NF4 neural-weight training qualification.

## Identity and byte requirements

The exact upstream model is `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (not the Aggressive fork and
not `Qwen3.5-4B-Base`). Two upstream Git LFS pointer pins, **not yet measured
local payload hashes**, are:

| File | Physical byte length | SHA256 LFS OID |
|---|---:|---|
| `model.safetensors-00001-of-00002.safetensors` | 5,329,398,688 | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` | 3,990,429,408 | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |

Combined 9,319,828,096 bytes, excluding the seven necessary metadata files.
The metadata (`config.json`, `tokenizer_config.json`, `tokenizer.json`,
`model.safetensors.index.json`, `chat_template.jinja`, `merges.txt`, `vocab.json`)
lack independently trusted full-byte SHA256 pins. `hash-audit` records their local
hashes; it never implies their upstream custody has been admitted.

## CLI: no external effects

Use an isolated, user-approved target directory *on VN1* (not Lappy, not a
shared existing model cache). Commands shown are examples; the target path is
deliberately illustrative, **not an assertion that this directory exists**.
Python 3.10+ standard library suffices for preflight, planning, hashing and
log parsing; actual host runtime requirements must be separately inventoried.

```sh
# 1. Read-only preflight and pinned acquisition argv. Always dry-run.
python -m trainingday.official_inference_admission plan \
  --host VN1 --snapshot /path/approved/isolated/qwen35-official-851bf6e

# 2. Only AFTER a separately authorized operator performs the printed
#    'hf download' argv on VN1, hash all 9 downloaded local files.
#    This is a LOCAL HASH AUDIT, never official admission.
python -m trainingday.official_inference_admission hash-audit \
  --snapshot /path/approved/isolated/qwen35-official-851bf6e

# An old 'verify' subcommand is retained for compatibility, but its exit code
# is 3 (ADMISSION HOLD) even if the official shard hash audit passes, because
# trusted full-byte metadata pins and independent custody are unavailable.

# 3. After externally authorized GGUF conversion and SHA receipt, inspect the
#    candidate GGUF and runtime. This also ONLY prints a probe command.
python -m trainingday.official_inference_admission benchmark-plan \
  --host VN1 --engine llama.cpp \
  --gguf /path/approved/isolated/qwen35-official.gguf \
  --binary /path/approved/pinned/llama-cli

# 4. Optional, explicitly opted-in bounded version diagnostics; no model load.
python -m trainingday.official_inference_admission version \
  --binary /path/approved/pinned/llama-cli --probe-version

# 5. Parse stdout/stderr CAPTURED BY THE AUTHORIZED HOST OPERATOR,
#    recording exit status and measured wall-time from that external run.
python -m trainingday.official_inference_admission record \
  --stdout /path/approved/receipt/probe.stdout \
  --stderr /path/approved/receipt/probe.stderr \
  --exit-code 0 --wall-ms 1234 --gguf-sha256 <64-hex-digest>
```

**Nothing here has a `--download`, `--execute`, or `--authorize` switch.** The
planning CLI does not invoke `hf`, invoke llama.cpp/KoboldCPP inference,
convert safetensors to GGUF, start a server, create directories, delete files,
change broker status, or modify processes. `--probe-version` invokes only the
operator-supplied executable's `--version`, with a five-second timeout; use
only with an independently trusted binary. It records executable byte SHA256
and exact version output/exit status rather than guessing a compatible
llama.cpp build. KoboldCPP version and API formats vary; this kit emits no
unverified KoboldCPP inference syntax. A reviewed version-specific adapter is
required before claiming a KoboldCPP benchmark.

The Hugging Face plan uses the current official `hf download` CLI contract:
explicit positional filenames, immutable `--revision`, `--local-dir`, no
wildcards, no `--force-download`, no metered services. Only an empty, isolated
target can generate a candidate argv. Actual HF transfer must be external to
this CLI and followed by `hash-audit` on the same physical snapshot. Free space
reserve requires shard bytes + 24 GiB for staging. The inference candidate
requires at least 6 GiB *available* RAM and 24 GiB disk free, normalized
1-minute CPU load no more than 80%, no known existing llama/KoboldCPP process,
and observed host name matching the expected VN1/VN2 name. Unknown process,
memory or load state is HOLD. These are coarse preliminary bounds, not proof
that full checkpoint conversion or GGUF inference fits memory. For Windows,
available RAM and CPU come from system APIs and running-process names from
`tasklist`; for Linux, from `/proc` and `getloadavg`. Hostnames are not
cryptographically attested. **This source revision fails closed on native
Windows for all local file hashing and log ingestion**: no Windows native
handle-anchored/no-reparse-point equivalent has been implemented or verified.
The `fd_anchored_hashing_supported` preflight gate is false on Windows, so a
native Windows execution path remains HOLD. A specifically approved and
attested POSIX execution environment can be evaluated independently; calling
WSL itself a verified VN1 machine/runner without host identity attestation is
not sufficient.

### W22 security repair: exact scope and residual races

Local byte verification and log ingestion now open from the filesystem root
through **each** POSIX directory component using retained directory FDs,
`O_DIRECTORY`/`O_NOFOLLOW`, and a no-follow regular-file leaf. The nine snapshot
members share one retained snapshot directory FD. File and log reads use that
same open FD for size check, bounded read, SHA, and pre/post `fstat` identity,
size, mtime and **ctime** comparison. Log ingestion refuses more than 2 MiB
even if the path grows or is swapped during a read. Malformed timing text
produces null TPS fields, not a parser exception.

This mitigates ordinary symlink ancestors and in-place rewrites with restored
mtime. It does **not** establish immutable snapshot semantics: a privileged
actor might manipulate timestamps; a hard link or bind mount may refer to
another object; bytes may be modified *after* the final `fstat`; a live
converter/runner could open different bytes later. A host-owned trusted
staging directory, snapshot lock/ownership, SHA bound to the process-opened
file descriptor and independently authenticated execution receipt are still
required for physical custody proof. No CLI flag supplies that authority.

**W30 pathname-replacement repair:** The first shard's original file descriptor
could previously close before the second shard was hashed, leaving a passing
SHA receipt even when its name had already been replaced. `hash-audit` now
retains all nine original file descriptors through the scan, rechecks their
post-hash inode/time stamps, then opens **every current filename again** under
the same anchored directory FD with `O_NOFOLLOW` and compares identity/signature.
Any mismatch is `HOLD` (CLI exit 2), not `shards_verified: true`. This detects
the reproduced post-hash first-shard and metadata path swaps, but the final
per-name checks are sequential, **not atomic**, and do not prevent a later
rename after the last check. Physical use still requires host-owned immutable
custody and a process-bound independent receipt.

**W31 ancestor-replacement repair:** The retained directory FD alone could
still point to a valid old snapshot after an *ancestor* was renamed and its
original lexical path redirected to a bogus tree. After all nine per-file
checks, the verifier now also rewalks the snapshot's full absolute path from
the filesystem root, enforcing `O_NOFOLLOW` on every component, and compares
the newly reached snapshot directory's `(device, inode)` with the original
retained handle. A symlinked ancestor or replacement plain directory now
causes `HOLD`/exit 2, rather than a false stable-path receipt. This remains a
point-in-time **non-atomic** path check, not a guarantee against post-return
changes or privileged mount namespace manipulation.

### Exact physical VN1 authority request

Patrick or his authorized VN1 host steward must explicitly approve **one
bounded CPU-first inference staging job on VERA-NODE-1**, by authenticated
VN1 host-controlled channel, with all of the following fields:

1. **Transport and actor:** authorize use of the *actual authenticated*
   VN1 execution transport for `mkdir`, official HF download, local SHA256
   readback, dependency/version inspection, and one bounded CPU inference
   process. The current WorkBridge read/write/process capabilities were
   disabled; the alternate transport had read-only diagnostic scope. Approval
   must expressly extend that specific host transport to these bounded
   effects, or nominate a different already authenticated host-owned runner.
2. **Immutable subject:** `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, exactly nine named
   upstream files (2 shards and 7 metadata), max expected shard payload
   9,319,828,096 bytes plus bounded metadata and HTTP transfer overhead.
3. **Filesystem custody:** a *new absolute VN1-local directory* selected and
   checked for non-collision by the host owner; no Lappy permanent weights,
   VN2 mirror, existing model cache overwrite, shared service, or user-data
   removal. Define the download/cache/temp locations and a free-space floor.
4. **Workload reservation:** current VN1 identity, CPU/RAM/disk/process
   inventory, duration/time window, CPU thread cap, memory cap, and abort
   conditions. Inference execution must be explicitly admitted by the
   host-owned launcher, with process-owned stdout/stderr, wall-time, RAM/VRAM
   measurements. Do not touch active VN2 `llama-server` or Lappy broker.
5. **Trust and separation:** approve the exact reviewed code commit and
   declared `hf`/GGUF converter/runtime versions. Require full measured
   shard hashes, separately trusted metadata-byte provenance, GGUF conversion
   receipt and hash before attributing text-only output to the official
   model. This inference permission is **not** a TrainingDay optimizer/GPU
   lease, owner signer enrollment, rights waiver, or permanent deployment.

**Suggested exact authorization text:**

> I authorize one isolated, zero-cost, CPU-first Qwen3.5-4B official inference
> staging and probe on VERA-NODE-1 via [named authenticated transport and
> operator], in [new absolute VN1-local directory], during [time window],
> capped at [CPU threads, RAM, disk/write bytes, runtime]. Download only the
> nine pinned files at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a,
> hash and report them, and only if separate GGUF provenance and runtime
> checks pass, execute one bounded text-only probe under host-owned workload
> supervision. Do not alter any existing cache, service, VN2 load, Lappy
> weights, host broker, or training dataset. Stop on collision, resource
> conflict, trust failure or quota exhaustion.

Bracketed fields must be filled from actual verified host state and owner
choice. A chat permission sentence by itself does **not** turn the installed
TrainingDay broker's unconditional GPU/training HOLD into a launch grant.

## Telemetry contract and limits

`record` emits `TRAININGDAY_TEXT_ONLY_INFERENCE_TELEMETRY_W20_V1` JSON with
`sample_prompt`, GGUF SHA256 *operator declaration*, exit code, elapsed wall
time, stdout excerpt, llama.cpp prompt and generation token timing/TPS when
present, and memory peak fields. The memory peaks are **null** until a real
host monitor has independently measured and bound them; the current parser
never fabricates peak RAM/VRAM. Log text and exit status may be staged or
synthetic, so `decode_verified_by_trusted_witness` is always false and the
record has `UNATTESTED_OPERATOR_SUPPLIED_LOGS` evidence class. A successful
exit/log is not itself proof of official GGUF provenance or model decoding.

To elevate evidence, a trusted host-owned execution wrapper must record its
own immutable process command, PID/start/end times, SHA256-verified GGUF and
runtime executable, stdout/stderr digests, exit code, peak RSS/VRAM and token
throughput, with host/owner identity and an independently verified custody
chain. No such wrapper or resulting receipt is claimed by this source change.

## Validation

Run tiny synthetic tests, which never acquire HF weights or start an engine:

```sh
python -m unittest -v tests.test_trainingday_official_inference_admission
```

The tests intentionally reject a fake two-shard model as official even when
its local synthetic SHA files can be hashed correctly. They test path and
collision HOLDs, unknown/busy host resource HOLDs, pinned argv, GGUF command
generation, ancestor symlink races, log swaps, restored-mtime rewrites,
admission-vs-hash exit codes, and malformed TPS text. Source-ready ≠ verified official
weights ≠ compatible GGUF ≠ actual inference ≠ trusted benchmark.
