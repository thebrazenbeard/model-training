# VERA::TRAININGDAY — neural training lane (source candidate)

**Scope:** `thebrazenbeard/model-training` source candidate. Independent
corpus-builder lane: `thebrazenbeard/labyrinth`. The earlier inspected
Draft PR #59 at `188d3340e4498e69d8c54d66ebee9dabd6582d31` is a
historical reference, **not** an automatically authorized current producer.
Reconfirm the intended producer PR/head and exact corpus before every pilot.
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

The base binding JSON (`TRAININGDAY_PINNED_BASE_V1`) requires model ID, the
explicit source-pinned revision above, text-only loader class, quantization
policy, absolute local snapshot path and SHA-256 for **every** regular snapshot
file, including nested tokenizer/templates, weights and configuration. Unbound
files and symlinked snapshot or binding ancestors are rejected. The binding is a custody
claim: independently verify the Hugging Face download revision and local
file hashes before approval. Do not fabricate this binding for an absent base.

Match controls on source manifest, selected IDs, prompts, base binding and seed.
The zero arm emits a `control_source_sha256` commitment. Each R15 trained arm
must supply that value as `--expected-control-source-sha256` in addition to
the zero arm's `--expected-initial-digest`. This checks equality against the
operator-frozen control source; it does not independently authenticate the
producer, rights, host, or control-arm execution.

- `zero`, 0 optimizer steps: save initialized LoRA and record FP32 digest.
- `correct`, 1–8 steps: require the **exact zero-arm initial digest** and
  nonzero before/after tensor delta.
- `corrupt`, matched steps: per-family deterministically wrong labels, require
  the same initial digest and a separate saved adapter. An improvement also
  found in this arm does not prove reasoning learning.

Example *source preflight* using generic verified paths (loads no GPU and does
not train). The selected producer revision below is only an example argument:

```bash
python -m trainingday.train_neural \
  --manifest "<verified-corpus-manifest>" \
  --producer-git-head "<verified-producer-commit-sha>" \
  --base-binding "<verified-base-binding.json>" \
  --work-root "<isolated-work-root>" \
  --output-root "<isolated-output-root>" \
  --run-id zero-example --mode zero --steps 0 --cases 64 --preflight-only
```

Remove `--preflight-only` only after every source and host gate passes. The
source enforces at least 3900 MiB GPU capacity, 3200 MiB torch-visible free
VRAM, and 9 GiB available system RAM. Host ownership, GPU exclusivity,
temperature/power budget, cancellation and restoration require separate live
authorization and verification. Do not stop unrelated workloads, disable
protections, delete model caches, or reuse an output run ID. Preserve failed
run artifacts for forensics.

## Evidence ceiling

The in-job receipt records the frozen control-source digest, manifest hash,
selected-ID hash, mode-specific
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
