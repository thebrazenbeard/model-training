# VERA::TRAININGDAY — neural training lane (source candidate)

**Scope:** `thebrazenbeard/model-training`, branch
`trainingday/neural-runner-r14-20261029`. Independent corpus-builder lane:
`thebrazenbeard/labyrinth` Draft PR #59 at exact inspected head
`188d3340e4498e69d8c54d66ebee9dabd6582d31`.
Coordination: https://github.com/thebrazenbeard/chat-communication-bus/issues/404 .

This code **does not train by being imported**. Its tests are CPU-only and
do not qualify a GPU, model, adapter, protected evaluation, or local release.

## Boundary with the producer

The actual Labyrinth v1 manifest uses `vera.trainingday.corpus.v1`,
`train/*.jsonl`, `eval/*.jsonl`, `licenses/*.txt` and shard keys
`split/path/count/bytes/sha256`. This differs from the initial proposed
`VERA_TRAININGDAY_ADMITTED_SHARDS_V1` interface. Both are recognized, with
the Labyrinth adapter at `trainingday/labyrinth_intake.py` bound to the
producer's exact Git head. A manifest path that does not yet exist is a HOLD,
not a pretext to generate fake rows or silently switch datasets.

For the Labyrinth schema, the neural reader:

1. Verifies completed-state manifest, MIT source revisions and exact copied
   license notices; checks every train/eval shard byte hash and reported size.
2. Parses **only train** JSONL rows. Eval shard text is never parsed or used.
3. Replays each integer operation chain independently, reconstructs the exact
   prompt and result, checks case/content digests and a disk-backed duplicate
   index, then selects a bounded deterministic set of verified reasoning rows.
4. Counts licensed code source rows but does **not** present unverified code
   files as supervised question/answer solutions. A distinct code-continuation
   objective and controls are required before code enters weight training.

The other schema, `VERA_TRAININGDAY_ADMITTED_SHARDS_V1`, accepts reviewed
per-row provenance, unique IDs/prompts across train/dev/test, and explicit
license grants. It is not a rewrite of the Labyrinth producer.

## Frozen model and controls

Intended *new* base: `Qwen/Qwen3.5-4B`, pinned Hugging Face revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (retrieved 2026-10-29).
Its two `.safetensors` shards total 9,319,828,096 bytes. The local GGUF is
not a BF16 training donor. Do not substitute the historical
`rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` derivative or silently resume R13;
its parent corpus provenance is held for release. This runner creates a
fresh, separate `r=4`, `all-linear` LoRA under NF4 double quant/BF16 compute,
nonpaged AdamW8bit, with trainable FP32 tensors and completion-only loss.

The base binding JSON (`TRAININGDAY_PINNED_BASE_V1`) requires model ID, exact
40-hex revision, text-only loader class, quantization policy, absolute local
snapshot path and SHA-256 for each base file (including every safetensors
shard, `config.json`, and `tokenizer_config.json`). The binding is a custody
claim: independently verify the Hugging Face download revision and local
file hashes before approval. Do not fabricate this binding for an absent base.

Match controls on source manifest, selected IDs, base revision and seed:

- `zero`, 0 optimizer steps: save initialized LoRA and record FP32 digest.
- `correct`, 1–8 steps: require the **exact zero-arm initial digest** and
  nonzero before/after tensor delta.
- `corrupt`, matched steps: per-family deterministically wrong labels, require
  the same initial digest and a separate saved adapter. An improvement also
  found in this arm does not prove reasoning learning.

Example *source preflight* (replace with **existing**, verified paths; it
loads no GPU and does not train):

```powershell
& 'C:\ProgramData\ProRun\model-env\Scripts\python.exe' -m trainingday.train_neural `
  --manifest D:\VERA\scratch\trainingday-corpus-20261029\corpus-v1-100k\manifest.json `
  --producer-git-head 188d3340e4498e69d8c54d66ebee9dabd6582d31 `
  --base-binding D:\VERA\.scratch\trainingday-neural-r14-20261029\base-binding.json `
  --work-root D:\VERA\.scratch\trainingday-neural-r14-20261029 `
  --output-root D:\VERA\.scratch\trainingday-neural-r14-20261029 `
  --run-id zero-r14a --mode zero --steps 0 --cases 64 --preflight-only
```

Remove `--preflight-only` only after every source and host gate passes. The
Lappy GPU has 4 GiB VRAM; source enforces >=3900 MiB capacity, >=3500 MiB
runtime-free VRAM, and >=9 GiB free system RAM. Pausing Ollama/PreActive is
authorized by Patrick for this training work, provided process identity and
restart state are read back, service restoration is verified, and unrelated
jobs remain untouched. Do not disable protections, delete model caches, or
reuse an output run ID. A failed run's files remain for forensics.

## Evidence ceiling

The in-job receipt records manifest hash, selected-ID hash, mode-specific
training-input hash, base revision, FP32 before/after digest, optimizer steps,
save hashes and runtime. Its status is deliberately
`ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED` until a separate fresh-process
loader confirms recovery. A saved adapter alone does not demonstrate semantic
transfer or a good Q4_K_M local GGUF. Run frozen/zero/corrupt comparisons on
newly reserved operation chains and unseen program structures before claim
promotion. The final V10 bank remains outside this source lane.

## Tests

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_trainingday_admission.py \
  tests/test_trainingday_labyrinth_intake.py \
  tests/test_trainingday_neural.py
```

No training data, BF16 model weights, neural adapter output, sealed test
plaintext or private ChatGPT export belongs in this Git branch.
