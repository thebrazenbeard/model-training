# TRAININGDAY Tier 1 Contract B: immutable public-fit request (source only)

Patrick selected **B**: one bounded *nonexclusive* WDDM experiment may be
engineered for Lappy. This does **not** issue a physical launch grant.

`trainingday/tier1_public_fit_plan.py` produces canonical ASCII JSON and a
SHA-256 plan digest for a future owner-signed Windows HostBroker grant. The
source-only receipt cannot sign, launch, download weights, mount protected
data, alter Q6, or promote a checkpoint. The currently installed broker and
`physical_launch_authority.require_physical_launch_authority` remain on HOLD.

Fixed request scope:

* Host `LAPPY`, policy `WDDM_BOUNDED_COEXISTENCE`, foreign GPU clients allowed
  without falsely claiming hardware exclusivity.
* Official `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with upstream **Git LFS
  object IDs** for the two official model shards. These are not verified local
  weight bytes. `config.json`, `tokenizer.json`, `tokenizer_config.json`, and
  `model.safetensors.index.json` currently have **UNKNOWN trusted byte hashes**;
  prior unsubstantiated config/tokenizer SHA claims were withdrawn. The old
  repository catalog's `rodrigomt/*Uncensored-Aggressive*` base is **not**
  admissible. All snapshot files require separately authenticated source
  provenance and exact local-byte readback; no caller hash declaration suffices.
* Candidate `trainingday.lowmem_experiment.one_step_kernel` already contains
  official NF4/BF16 LoRA synthetic batch 1, sequence length 64 and precisely
  one AdamW optimizer step, but is presently **unreachable** behind its
  unconditional physical HARD HOLD. Its source uses `device_map=0` and reports
  `actual_cpu_offload_used=False`: **no demonstrated CPU offload backend, no
  measured official 4B training fit in Lappy's 4 GiB GPU**. A separate ZeRO-3
  offload configuration is a hypothesis, not tested kernel behavior. Hardware
  fit remains `UNKNOWN_NOT_ESTABLISHED_ON_LAPPY` even if free-VRAM preflight
  is favorable. No alternate Python flag authorizes launch.
* Maximum wall time 1,800 s; **observational** GPU-memory target 3,072 MiB
  plus 256 MiB driver headroom for initial preflight. WDDM has no guaranteed
  per-process hard GPU-memory quota. A valid preflight does not prove a lease;
  future foreign allocation, ambiguous telemetry, or unsafe memory requires
  abort/hold. Earlier Lappy samples had only 585–718 MiB free and would HOLD.
* No R201/DEV, evaluator, NAS Q6, protected mounts, unattended repeated steps,
  or checkpoint promotion. Weights are downloaded only on demand to VN1/VN2;
  no permanent model copy on Lappy.

Before *any* physical launch: independently installed host broker must verify
an owner signature bound to exact `plan_sha256`, `manifest_sha256`,
`executable_sha256`, host, TTL, nonce, monotonic Tier1-local replay epoch and
resource caps. It must retain real process identity and own a dedicated WSL2
Linux process group or cgroup, with targeted termination and after-crash
no-orphan readback. Killing a broker-owned `wsl.exe` via Windows JobObject is
not proof that a nested Linux CUDA worker died. Do **not** terminate the entire
shared Ubuntu distro or kill other applications to manufacture free VRAM.

The candidate must be reviewed and installed/tested in Windows+WSL with forged,
expired, replayed, wrong-host, wrong-model, OOM, timeout, PID-reuse and broker
crash negative cases. Source-only tests cannot establish this host behavior.

This source's `manifest_sha256` is the SHA-256 of canonical ASCII JSON for the
two upstream shard-LFS object IDs, missing-metadata list and evidence class;
it is **not** the independently verified complete local snapshot. It is not a
substitute for authenticating the config, tokenizer, index, other signed
bindings, or current model file bytes. The proposed companion
Tier1 grant validator uses `model_revision`, `vram_target_mib`,
`protected_access=NONE`, `outputs=UNPROMOTED_ONLY`, and
`wsl_cleanup=TARGETED_WSL_PIDTREE_REQUIRED`; the host must independently
compute the expected source-manifest digest and file/executable bindings before
accepting an owner signature. Until that integration is implemented and tested,
both source candidates remain unconnected HOLD contracts.

Source-only checks:

```sh
python -m unittest discover -s tests -p test_trainingday_tier1_public_fit_plan.py -v
python -m trainingday.tier1_public_fit_plan
```

The latter prints a request digest only; it has no execution switch.
