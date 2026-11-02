# TrainingDay FSDP2 synthetic CPU-offload qualification (2026-11-02)

**Evidence classes:** LIVE LAPPY WSL2 hardware experiment, source-only safety tests, and
GitHub CI. This is **not** a Qwen3.5-4B fit, authorized TRAIN run, model-quality
evaluation, GPU lease, or host-owned broker installation.

## Why this path

Lappy has 16 GiB configured WSL2 RAM, 8 GiB swap, and a locally attached RTX 3050
with 4 GiB VRAM. Its Python 3.12.16 training venv uses PyTorch 2.14.0+cu130 and
can perform ordinary synthetic NF4/LoRA CUDA updates. DeepSpeed 0.19.6 was not
installed; a bounded attempt to install it without replacing existing packages
failed at wheel build with `MissingCUDAException: CUDA_HOME does not exist`.
GCC, NVCC and Ninja were not installed in the Ubuntu environment, and noninteractive
`sudo -n` was unavailable. No root packages or Linux GPU drivers were installed.

PyTorch FSDP2 has an existing `CPUOffloadPolicy(pin_memory=False)` that places
sharded parameters, gradients, and optimizer states on CPU, with forward/backward
work on CUDA. See [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).

## Runnable, non-admission test

On Lappy, inside Ubuntu-26.04 WSL2 and its isolated training environment:

```bash
/home/vera/.local/share/vera-trainingday/.venv/bin/python \
  -m trainingday.fsdp2_cpu_offload_smoke
```

The module constructs only random `64 -> 64 -> 8` layers, initializes a
single-rank local NCCL process group using a temporary file rendezvous, applies
FSDP2 CPU offload, runs three CUDA forward/backward plus CPU AdamW optimizer
updates, checks final sharded parameter CPU residency and changed parameter
SHA-256, emits JSON and destroys the process group. It has no model paths,
train-data input, checkpoint output, CLI permissions, or protected-evaluator
access. CPU/GPU resource availability and NCCL support remain runtime-dependent.

## Physically observed result on Lappy

The prototype and the exact revised GitHub-published source at commit
`1a69f3802f8375ff8e8ef4ed25cb767befade227` were both executed on Lappy WSL2
and returned matching losses and parameter digests. The revised version also
asserted **actual CPU gradient and AdamW optimizer-state residency**, not merely
parameter placement. Results:

| Step | Loss | Synchronized wall time |
|---|---:|---:|
| 1 | 0.94081330 | 0.87601 s |
| 2 | 0.77570283 | 0.01534 s |
| 3 | 0.63003927 | 0.02129 s |

Status: `SYNTHETIC_FSDP2_CPU_OFFLOAD_STEPS_PASS`.
Before SHA-256: `1e1b655408d33a4b0eccf0baa78bc233f358d902736da9fbf818461d9f643c92`.
After SHA-256: `baa0fecf4f1f36c0ecfb261d22e16fc6c24cec67fbaa61c1415e0d6c3e87ec4c`.
Peak process-local PyTorch CUDA allocation: **17,079,296 bytes**.
`qwen_weights_used=false`, `training_authorized=false`.
Measurements are of a **tiny synthetic model**, not scaled to 4B.

## Remaining hard gates

The single-rank FSDP2 experiment establishes training-capable CPU offload
mechanics; it does not demonstrate a full 4B BF16 or NF4 model load, memory-fit,
offload throughput, trusted official Qwen bytes/tokenizer, rights/grants,
protected EVAL custody, host-owned cgroup/process and GPU lease, durable adapter
checkpoint, or generalization. Single-rank FSDP cannot be presented as
multi-GPU sharding. The installed `VeraTrainingDayHostBroker` remains deny-only,
and `trainingday.physical_launch_authority` stays an unconditional HARD HOLD.
Do not splice this smoke into protected model or corpus launch.

The standalone `tests.test_trainingday_fsdp2_cpu_offload_smoke` uses stdlib AST
checks and can run on a CPU GitHub CI host; **those tests do not execute FSDP2**.
A future exact-commit WSL reread with full host/authority/evaluator evidence is
required before any protected Qwen optimization attempt.
