# W53: CPU Q4 inference is not BF16 conversion admission

Status: **SOURCE-ONLY, NO EXECUTION AUTHORITY, HARD HOLD**. This is a factual
policy-boundary addendum to the existing W20 and real-inference probe screens
and W51 conversion screen. No new model source, output GGUF, host lease, or tested conversational
deployment is introduced.

## Distinct capacity decisions

| Proposed work | Existing source-only screen | What it proves |
| --- | --- | --- |
| A size-bound Q4 GGUF **probe** | `trainingday.real_qwen35_inference.require_probe_resources`: max(5 GiB, 1.3 x GGUF bytes + 1 GiB) available RAM; for a 2,783,446,880-byte candidate this is exactly **5 GiB** | A rejection screen contingent on caller-supplied idle acknowledgement. Its `probe()` is **unconditionally disabled**. Neither that assertion nor RAM passes is a host lease. |
| VN1-first generic existing-GGUF **benchmark plan** | `trainingday.official_inference_admission`: >=6 GiB available RAM and >=24 GiB stage reserve, plus CPU/process checks | Separate, stricter numeric planning path, not owner authorization |
| Full official safetensors -> BF16 GGUF -> Q4 quantization | `trainingday.offline_conversion_capacity`: >=12 GiB host/WSL guest available RAM, >=1 GiB swap, >=25 GiB stage | Conservative caller-supplied rejection only; **always** `launch_authorized=false` |

The 5, 6 and 12 GiB source thresholds are distinct rejection policies, **not
measured memory requirements for Qwen3.5-4B**. A reported 6-7 GiB available
can make an existing 2.8 GB Q4 inference attempt worth considering once all
other gates are met. It does not make 9.3 GB BF16 conversion safe, and it does
not establish that KV cache, hybrid Gated DeltaNet allocations, or a model
load can fit. The 5 GiB path cannot execute because `probe()` always raises
`InferenceHold`. GPU/CPU offload is inference placement, not model conversion.

## W53 physical evidence (2026-11-05 UTC)

- Executor identified `VERA-NODE-1` with `FreePhysicalMemory=7,025,528 KiB`
  (~6,861 MiB) and `FreeVirtualMemory=6,726,412 KiB`. A targeted Windows
  `Get-Service VeraTrainingDayHostBroker` query returned no service on VN1;
  this does **not** prove no other lease implementation exists.
- A targeted `Get-Process -Name 'llama*','kobold*'` returned no names. This is
  not a complete process ownership check or an exclusive process reservation.
- Bus #404 W47 reports VN1/VN2 custody for official-revision HF snapshot files
  (`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) and
  October-30 correct8 R16 adapters. The adapter bytes were freshly hashed on
  both hosts in W47; the ~9.3 GB base shards were **not independently rehashed
  in W53** (nor in W47). The earlier shard hash receipt is a different,
  historical evidence class. These are not a served GGUF or verification of
  quantized derivative ancestry. Prior scoped inventory found only a
  third-party Q3 and donor Qwen2.5-Coder GGUFs, not an independently verified
  official-derived Q4 GGUF.
- No fresh conversion, model load, adapter-to-GGUF mapping, output memory
  measurement or inference occurred in W53.

## Required independent physical gates before a CPU Q4 conversation

1. Owner-designated exclusive process slot and externally enforced resource
   budget, not a caller-set `lease=true` or local cooperative lock.
2. Verified Q4 file bytes, quantizer/toolchain chain, authenticated official
   source revision, tokenizer metadata and correct R16 LoRA conversion. A
   third-party file naming the official revision does not prove equivalence.
3. Compatible pinned llama.cpp or KoboldCPP binary, explicit CPU-only context
   budget, available RAM/CPU and another check of active workloads.
4. An owner-attended bounded initial load with peak RSS/latency measurement,
   rollback and separate-process base vs R16 response comparison.

**Decision:** A *possible* Q4 CPU route survives as a hypothesis. The existing
5 GiB function is a size-bound refusal check, not permission to lower the W20
6 GiB plan threshold or bypass a host-owned lease. Authentic
official Vera inference is **not yet demonstrated** and is not authorized by
these source checks. Preserve the W20 and W51 denial boundaries until measured
evidence and host ownership change them explicitly.

Offline regression: `python -m unittest -q
tests.test_trainingday_w53_dual_lane_capacity_policy`. No weight or server
is accessed by that test.
