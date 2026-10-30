# TRAININGDAY: CPU-only MeZO-LoRA vs AdamW multiseed method calibration

Status: **REAL small-model CPU optimizer effects, not official Qwen3.5-4B training or improved reasoning qualification**. This draft stacks on neural PR #140 at source head `9a6c3b72ef8ad425fa4df371ee10cf700972c8cd`. Keep it separate from R18 Lappy GPU owner and all protected structural banks.

## Purpose and exact method

`trainingday/mezo_cpu_ablation.py` holds an original random 8→16→1 backbone fixed and trains only a 16×2 A and 2×1 B LoRA path (34 trainable FP32 parameters) on synthetic linearly-oracled sign predictions. Each run derives 48 train and 32 separately seeded holdout examples. MeZO uses symmetric two-sided forward differences and no `.backward()`, AdamW uses one autograd forward/backward per optimizer step. Both have frozen, correct and corrupted-label arms. The initial tensor, training data and heldout hashes are compared across methods and controls. The full per-arm tensor digests change only after real updates. None of this uses Vera's Qwen3.5-4B model, protected V2/V3/R16 evaluation, quantized 4B NF4 weights, shared GPU, or installed adapter.

`trainingday/mezo_multiseed.py` adds a repeatable five-seed sweep with 40 and 200 optimizer-step budgets, 50 arms total (2 frozen plus 8 trained arms per seed). It refuses duplicate/out-of-range seeds, source mismatches, and populated output paths. The main report is structured JSON, with a separate summary and digest. This is an exploratory, not tuned/blind benchmark: method hyperparameters differ and elapsed-clock data include initialization/warmup.

## Physical experiment, Lappy 2026-10-30

- Exact experiment checkout head before documentation: `0d5b8819bf7c520c2c62a8d3dd4966fc700dca67`.
- Five seeds: `29037,29038,29039,29040,29041`; training steps `40,200`.
- Raw results: `D:\VERA\.scratch\trainingday-mezo-pilot-20261030\multiseed-5seeds-40-200\raw-receipts.json`, SHA256 `ca16fa107eaa86685c210eae335385819bf9bc1c120d9d829a53602a9c9fc18d`.
- Summary file SHA256 `376388b7e0014efca5c538270a0c358f0518e29ded5b9f0239d0b8866a9a1b85`.

| Training arm | Mean correct/32 | Mean holdout MSE | Total forward passes across 5 seeds |
| --- | ---: | ---: | ---: |
| Frozen | 17.4 | 1.049057 | 0 |
| MeZO correct 40 steps | 17.2 | 1.017528 | 400 |
| MeZO correct 200 steps | 23.2 | 0.648045 | 2,000 |
| AdamW correct 40 steps | 26.6 | 0.511839 | 200 |
| AdamW correct 200 steps | 27.6 | 0.455756 | 1,000 |
| MeZO corrupt 200 steps | 8.2 | 1.965322 | 2,000 |
| AdamW corrupt 200 steps | 4.2 | 2.851025 | 1,000 |

Other arms: 40-step corrupt MeZO 16.6/32 MSE 1.081011; corrupt AdamW 5/32 MSE 2.614791. Correct MeZO improved with more steps, but AdamW with 40 steps exceeded MeZO with 200 on this task. The methods have different per-step costs; forward count is NOT equal time or equal FLOPs. In the initial single-seed six-arm experiment, all six 272-byte Safetensors adapters were physically saved, hash-matched and reload-tested; each zero control preserved its trainable digest.

## Independent cross-node readback

After this five-seed run, the raw report was gzip-compressed and transferred by authorized Executor to separate scratch areas on VN1 and VN2. Each used Python standard library only to decompress, verify the exact raw SHA, check all 50 arms for 34 trainable parameters, common within-seed initial/train/heldout hashes, correct optimizer step and forward counts, zero/update digest invariants, no 4B claim, and independently recompute mean accuracy. Both passed and reproduced `17.2,23.2,26.6,27.6` for the four correct-label method/budget groups. **This verifies report integrity and arithmetic, not genuine model execution on those nodes, and not independent hostile semantic review.**

- VN1 receipt: `C:\VERA\trainingday-mezo-verify-20261030\verified-receipt.json` SHA256 `10d57dc9b9962271727a180179a0350a7dfe1dd22a17d8fbed16330e21f161c0`.
- VN2 receipt: `/home/patrick/.local/state/trainingday-mezo-verify-20261030/verified-receipt.json` SHA256 `1b65b920944ca7aa9cc59585d98533817e6bab5ce1b34885d26c67501f7d9d53`.

## Test evidence and limits

Source tests: **75/75 TRAININGDAY tests pass across 10 test modules** at exact implementation head `0d5b8819bf7c520c2c62a8d3dd4966fc700dca67` in Lappy model-env + offline pytest. A broader repository-wide `pytest tests` was attempted but aborted at collection with six import errors due to absent `bs4` and `fastapi` in this environment. Do not call this a passing full repository suite. `git diff --check` passed; source worktree was clean before documentation.

> Hostile challenge: A toy 34-parameter linear sign task says little about memory-fit or convergence of a 4B NF4 quantized model. The MeZO 200-step gain may reflect additional forward passes, not superior memory efficiency. AdamW warmup introduces timing bias. Do not extrapolate toy accuracy or implied VRAM savings to Vera.

Next R18 owner gate: run matched *official Qwen3.5-4B NF4 frozen-base + LoRA only* MeZO perturbation experiments **only if real CUDA load fits**, compare with QLoRA at equal actual GPU time and corpus selection plus frozen/corrupted-label controls, validate saved adapter and untouched structural holdout, then publish physical peak RAM/VRAM/accuracy and admit or reject. No optimizer swap or Vera promotion from this toy result.
