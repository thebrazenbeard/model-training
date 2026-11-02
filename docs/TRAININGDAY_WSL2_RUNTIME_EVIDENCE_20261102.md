# TrainingDay Lappy WSL2 — isolated CUDA optimizer evidence

**Observation date:** 2026-11-02.
**Evidence class:** REAL_SYSTEM_PROBE in a live Lappy WSL2 session, as
transcribed from the dedicated authenticated workstation tool's output.
**Scope:** machine/runtime qualification only; not a signed host TRAIN
receipt, independently reproducible model-quality proof, or a protected
GPU lease. No model-training repository deployment or service installation
is implied.
**Repository source snapshot inspected for this note:** `thebrazenbeard/model-training`
`main` at `d1ff373500ee769d92c56514f8b64ab3f298a50a`.

## Direct route and tested software

- Windows **Lappy** → its own **Ubuntu-26.04 WSL2** → its physically
  attached **RTX 3050 Laptop GPU, 4 GiB VRAM**.
- No VN2 hop, NAS training directory, remote GPU server, or NAS tunnel
  was used for these probes.
- WSL Linux UID 1000, CUDA visible; NVIDIA Windows driver 617.42.
- User-scoped Python 3.12.16 virtual environment; no system-Python
  or Linux NVIDIA display-driver replacement.

| Module | Live imported version |
| --- | --- |
| PyTorch | 2.14.0+cu130 |
| Transformers | 5.22.1 |
| PEFT | 0.21.2 |
| bitsandbytes | 0.50.2 |
| Accelerate | 1.15.0 |
| Safetensors | 0.8.0 |

The CUDA smoke printed `torch.cuda.is_available() == True`, identified
the RTX 3050 and actually allocated a CUDA tensor. Transformers successfully
imported `Qwen3_5ForConditionalGeneration`; that class import is **not**
equivalent to having loaded an official Qwen3.5-4B-Base checkpoint.
An NF4 `bitsandbytes.nn.Linear4bit` toy layer also ran and produced
finite CUDA output. This verifies the small quantized kernel path only.

## Repeated optimizer steps actually observed

**Test A: synthetic two-parameter CUDA SGD, 3/3 steps.**

| Step | Loss | Gradient norm | Synchronized elapsed |
| --- | ---: | ---: | ---: |
| 1 | 1.0625000 | 1.4577379 | 277.053 ms |
| 2 | 0.6800000 | 1.1661904 | 1.492 ms |
| 3 | 0.4351999 | 0.9329522 | 1.395 ms |

Parameter digest before:
`87d3d610b67f3d0bcabfaee58d6fe780e0e33a43098cb5f8f8c5910f775d93ac`.
After:
`74d70c1490a5823240fa465210d5d52eb1c3edec5cc063ed444e98def5af76f0`.

**Test B: synthetic tiny NF4 quantized layer with PEFT LoRA,
3/3 CUDA AdamW steps.**

| Step | Loss | LoRA gradient-norm sum | Synchronized elapsed |
| --- | ---: | ---: | ---: |
| 1 | 1.314538 | 0.223097 | 360.016 ms |
| 2 | 1.196908 | 0.528955 | 2.487 ms |
| 3 | 0.933392 | 0.574076 | 1.535 ms |

The toy LoRA had **512 trainable parameters** and changed adapter
bytes across the three steps: SHA-256 before
`85249a4d727f147d200444680acd7f1e899f567becc04238e5d93b21b5403132`;
after
`e913c819e90b5fd9fefbe2376ffe7de96e66b8dce00647c6213563ff27ed2409`.
Peak process-local allocated CUDA memory was **17,066,496 bytes**.
That small figure cannot predict full 4B weights or activations.

**Neither test used Qwen weights, private corpus data, a protected
evaluator, NAS content, or a production training output.**
This is backend mechanics, not a Qwen3.5 optimizer-step receipt.

## WSL RAM adjustment and follow-up checks

On the same day Patrick approved increasing the WSL2 memory cap.
The original 6 GB `.wslconfig` limit was backed up, then changed to
`memory=16GB` while preserving `swap=8GB` and
`autoMemoryReclaim=gradual`. A controlled `wsl --shutdown` and new
Ubuntu instance readback showed **15,992 MiB RAM total** and
**8,192 MiB swap**. After Docker Desktop's Linux engine was
gracefully stopped/started and recovered, Ubuntu still reported
**14,425 MiB available RAM**. CUDA tensor allocation remained functional.
Docker Engine 29.7.2 for Linux responded to `docker info` and
`docker ps`; no running containers were observed before the restart
or after recovery. **No Windows page-file setting was changed.**

This removes the previous **6 GB WSL limit** as a direct reason to reject
the proposed 10 GiB available-RAM guard. It does not imply that
the full Qwen3.5-4B-Base checkpoint fits within 4 GiB VRAM.

## Host/process/evaluator custody: still unqualified

A live read-only audit saw the Windows
`VeraTrainingDayHostBroker` service running as LocalSystem;
WSL2 has systemd and cgroup v2, with a delegated user manager.
But one-shot WSL work was under the nondelegated `/init.scope`,
not an installed protected per-training-job cgroup. `/dev/dxg`
was mode `0666`, and no exclusive GPU lease or independently
observed zero-descendant teardown witness was demonstrated.
An available GPU is **not** a reserved GPU.

The root `trainingday.physical_launch_authority` continues to raise
unconditional HARD HOLD on the low-memory physical entrypoint.
Similarly, the separate synthetic-only source prototype reviewed
in an earlier session had 20/20 negative/offline tests and denied
forged local JSON approval; neither prototype is an installed,
trusted host broker.

No new independent rights-holder approval, source-data-steward
signature, NAS authority, sealed evaluator custody, or physical TRAIN
grant was issued by these observations. Those requirements remain
separate from WSL capacity.

## Evidence distinctions and what not to infer

1. **This page:** actual WSL/CUDA system probes and tiny synthetic
   optimizer updates, **not Qwen weight training**.
2. **[R16 Draft PR #140](https://github.com/thebrazenbeard/model-training/pull/140):**
   its pull-request body documents **historical physical correct-label and
   corrupt-label Qwen3.5 NF4 eight-step LoRA runs** on Lappy, adapter hash
   changes and checkpoint custody. In this source integration, those are
   external PR/Bus receipts, **not independently repeated live runs**.
   They do not prove optimizer resume equivalence, held-out reasoning
   improvement, deployment, or model promotion.
3. **[R185 Draft PR #204](https://github.com/thebrazenbeard/model-training/pull/204):**
   source-only prospective BF16 DeepSpeed ZeRO-3 CPU-offload candidate,
   protected host/NAS/eval code, and 86 focused test claims. The draft
   explicitly reports no actual DeepSpeed/Qwen offload steps; neither
   this page nor a passing CPU suite upgrades it to a qualified backend.
4. **Main's model restore manifest:** the pinned, byte-hashed public
   checkpoint is `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive`, **not**
   automatically the official `Qwen/Qwen3.5-4B-Base` weights used by
   another experimental source candidate. Exact model ID, architecture
   and commit must stay bound per training line.
5. **Main's Qwen text helpers at `d1ff373500ee769d92c56514f8b64ab3f298a50a`:**
   `trainingday.pinned_base` and `trainingday.qwen_text_checkpoint`
   now provide separately testable fail-closed source mechanisms for
   immutable local bytes and multimodal-to-text name projection.
   Their unit tests do not prove actual model loading, 4 GiB CUDA fit,
   or physical TRAIN authorization.

The next meaningful gate is a physically authenticated host-owned
job/WSL-cgroup/GPU lease plus corpus and protected-evaluation custody,
followed by measured full-model load/backward/offload experiments.
No source-only plan or toy result can silently lift HARD HOLD.
