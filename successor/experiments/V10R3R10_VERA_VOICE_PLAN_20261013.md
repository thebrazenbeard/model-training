# R10 training decision: small local Vera, not Dolphin 24B

Target: **Vera**, the voice of **V.E.R.A. (Virtual Environment for Reciprocal Agency)**. Preserve independent, candid, imaginative judgment without granting a language model undelegated execution power. This target is the exact Qwen3.5 base plus verified R9 LoRA adapter on Lappy.

## Correction after user feedback

An earlier thread pursued the 23.57B Dolphin model as if it were the main Vera on Lappy. That model's 7.2 GB low-precision inference is slow and RAM-hungry. It has not been trained as Vera. **R10 never loads Dolphin weights**, and the R10 trainer must reject model substitutions. The Dolphin project is an optional behavioral design reference, not the parent adapter.

## Frozen source and exact selection

- Base: `D:\VERA\models\latest-trained\base`, Qwen3.5 language backbone, 4-bit NF4 QLoRA, 4 GB RTX 3050.
- Parent: verified R9 adapter SHA-256 `dc2a67a66c16c28a3b8cd5295a3e7a6ba243fb12ffb181540a0066b32047e495`, **28 cumulative completed optimizer steps**.
- Corpus parent: pinned R9 50k JSONL SHA-256 `aad72387cf803162230344a306dff127ac521197c2ed55fb1590cf4969cbad5c`.
- Vera voice seed: frozen `vera-model` non-personal seed SHA-256 `a66c85c036dbede26e9b0744c49672965e6a500cd42e50288942dba036ca770c`, originally designed for 24B but filtered for Qwen.
- Selected **32 identity/agency/style/correction** rows and **32 ordinary ability rehearsal** rows, interleaved 1:1 at corpus positions 224–287. The other 49,936 positions in the 50,000-row corpus are unchanged.
- Any example whose answer asserts Dolphin/Mistral/GLM/24B identity is automatically rejected. In fact, the first build found the misleading "Stop Qwen, train Dolphin" example and refused training; that sample was excluded.
- Vera-model's **25 held-out voice challenges stay excluded** from training; source SHA-256 `bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f`.

## Bounded next effect

R10 config uses exactly 8 incremental Qwen optimizer steps, low `2e-6` learning rate, 64 rows, 512-token ceiling, 8-bit nonpaged optimizer, rank-4 LoRA **resumed from R9**. CPU preflight passed: exact parent adapter, source spec, full corpus digest, Qwen tokenizer and 64/64 examples, max 396 tokens; 8 curriculum unit tests passed.

The R9 heldout evaluator was started but failed **before inference** with a GPU allocation OOM; no R9 results exist. Preserve its failed evidence and retry evaluation only through a separately resource-reviewed path. The R10 development optimizer, even if completed, is not independent quality validation.

No paid compute, forced process termination, GPU route takeover, merge, model qualification, or deployment. If resource loading fails, record the attempt and stop; no blind retry. An output receipt must separately prove completed steps, weight change, and adapter byte hashes before claiming training success.

## Fresh R10 launch result, 2026-10-13

The `preflight_v10r3r10_launch.py --train` invocation verified exact Qwen3.5 model type, frozen R10 50k corpus, R9 parent adapter SHA and 28-step receipt, correct eight-step target, then returned `HOLD_NO_WEIGHT_CHANGE`. At readback: 3,962 MiB GPU memory free, 5.31 GiB host memory free and only 0.75 GiB free virtual memory. The flags `LOW_HOST_RAM` and `LOW_WINDOWS_VIRTUAL_MEMORY_COMMIT` prevented an optimizer launch. No R10 adapter namespace was created.

Runtime-only receipt: `D:\VERA\.scratch\vera-v10r3r10-qwen35-identity-20261013-v1\r10-launch-gate-20261013.json`; SHA-256 `e4a6121db55c38ca2a37879191156ab3a3f3e7457c5f6f0885cb8778bd39cce3`. The receipt is a resource/provenance HOLD, not behavioral qualification or evidence of weight updates.

Memory diagnostics found `vmmemWSL` using around 10.6 GiB of private committed memory, with other browser and desktop processes using more. No WSL shutdown, global paging settings, or unrelated process termination was authorized or performed. Recover system memory/pagefile headroom before a **new uniquely recorded** attempt; never overwrite this receipt or blindly repeat after an unknown effect. The 24B Dolphin Job remains unlaunched.
