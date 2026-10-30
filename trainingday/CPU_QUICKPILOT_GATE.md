# TRAININGDAY — official 4B CPU LoRA fallback (source/experimental)

**Status:** real optimizer/tensor updates demonstrated on Lappy; not general reasoning gain, not the frozen GPU NF4 recipe, not deployed. Event: `VERA::TRAININGDAY`; coordination [Bus #404](https://github.com/thebrazenbeard/chat-communication-bus/issues/404). Base is official `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and local frozen binding SHA-256 `1d9ae5bde63775ce1d5c6f008e0d3bb708d0d60dd7b9ce3bc3b78ce82df9259b`.

## Why this branch exists

Lappy has 4096 MiB RTX3050 GPU, and `torch.cuda.mem_get_info()` reported ~3305 MiB free after initializing CUDA on an idle device even while Windows GPU reporting showed ~3962 MiB free. Existing 3500 MiB post-context floor prevented loading; regression test isolates an admissible context overhead while preserving failure for insufficient memory. **Passing this check does not imply 4B NF4 fits:** the actual Qwen3.5 4B GPU loader subsequently failed the legitimate Transformers quantizer check because some modules needed CPU/disk placement. Do not replace that actual failure by lowering thresholds further.

The CPU alternative in `trainingday/cpu_quickpilot.py` loads the same pinned frozen official base in BF16 on CPU, adds small FP32 LoRA `r=2` to `q_proj/v_proj` (229,376 trainables), uses completion-only training targets and conservative 30 GiB physical/14 GiB available memory guards. It writes adapters only to a caller-specified exclusive output path. It is a distinct experimental recipe and must not be compared as if matched with the GPU NF4 r4 all-linear controls.

## Exact local data and receipts

Pinned structural holdout v2 manifest on Lappy:
`D:\VERA\scratch\trainingday-corpus-20261029\corpus-v2-structural-100k\manifest.json`, SHA-256 `90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836`. The train-only admission chose 64 deterministic verified arithmetic records, ID SHA-256 `cba7debb7f059b23586f580fda51e57437cdf399d42be4896a9ff9c2b89ba56b`. All code-only rows and evaluation plaintext excluded from answer-supervised CPU training.

Output root: `D:\VERA\.scratch\trainingday-resource-clear-20261029\output`. Three real runs completed with exit code zero and 921,688-byte `adapter_model.safetensors` files; an independent PowerShell readback recomputed every file's SHA-256 and compared it to the persisted receipt:

| Mode | Optimizer steps | Trainable FP32 digest after | Adapter SHA-256 |
|---|---:|---|---|
| zero | 0 | `5b40d9f1ca510b9e1781941f466d3e2932b8eabbdc56931882b471cfbdc90b30` | `c1882253359f83f0e41a718722f3a34386923d0d97b6005ff667cd8eec64e09f` |
| correct | 1 | `0fb339a06f011acd964bd0945e93dbe2719fce2e95e52331bd9bafe034e9c19c` | `29c92c1df914cd031887592242927533fc1c863678aa0205fa061569d6e70130` |
| corrupt | 1 | `facb61503fa7bb72e9d14590a90a2c4cf07cea2bf9a160e4ab77217b57e55d4b` | `86b9a3c1dd649cd950e89f7e213657333d608da0f36348c1df7ae9adb71cd60b` |

All three began from the same digest `5b40d9f1ca510b9e1781941f466d3e2932b8eabbdc56931882b471cfbdc90b30`. Correct loss 4.082547; corrupt loss 4.232924. A separate fresh Python process loaded the CPU correct adapter with the base and saved `cpu-correct-reload-receipt.json`, status `REAL_SEPARATE_PROCESS_BASE_PLUS_ADAPTER_RELOADED`. Independent static safetensors audit compared 32 common adapter keys and found 16 changed matrices in each trained arm, but this **does not demonstrate improved reasoning**.

An initial four-arm CPU evaluation only advanced through part of a 64-case evaluation. Do not report incomplete scores as results. A separate neural lane has a real official 4B GPU NF4 adapter with frozen/correct/corrupt controls and is pursuing larger 8-step training; do not train competing 4B jobs on Lappy while that runner owns GPU/memory. No user-private source material is in this training corpus.

## Source qualification / safety

The current branch contains tests for frozen pinned intake, checksums, memory gates, completion-only labels, CUDA context accounting and trainable digest changes. At exact source head `a45c5b1f619b38b059a940d34a4a44c9056598f2`, a separate Lappy offline `uv` run of six TRAININGDAY test modules returned **44 passed, exit 0**, with a clean Git tree. This is source validation, not model effectiveness or external CI.

No merge, service restart, installed `vera-local:latest` replacement, paid HF compute, or model promotion was performed. CPU data and adapters stay private on Lappy. Preserve all receipts and existing working models until a genuine independent untouched-case evaluation beats frozen, sham and corrupted labels.
