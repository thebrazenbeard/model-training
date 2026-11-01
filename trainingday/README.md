# VERA::TRAININGDAY — neural training lane (source candidate)

**Scope:** Source candidate in `thebrazenbeard/model-training`.
The original R14 reader came from branch `trainingday/neural-runner-r14-20261029`.
Its Labyrinth producer reference is historical; the later R16 checkpointable
contract uses a different, explicitly pinned corpus. Neither version is an
installed release. These source candidates are **not runtime-qualified**
models; source and runtime qualification are separate gates.
Coordination: https://github.com/thebrazenbeard/chat-communication-bus/issues/404 .

This code **does not train by being imported**. Its tests are CPU-only and
do not qualify a GPU, model, adapter, protected evaluation, or local release.

## Boundary with the producer

The version distinction matters before any owner-host preflight:

| Source contract | Labyrinth producer/source reference | Manifest SHA-256 | Qualification ceiling |
| --- | --- | --- | --- |
| **Legacy R14/v1** | Original reader reference `188d3340e4498e69d8c54d66ebee9dabd6582d31`; the completed v1 data was generated later at `55e2639a1d3e01c71bf1555160eee89a132a2772` | `c722cd38b3cb42905079696691279c950de6b2d740287acc235a12d764f6106d` | Historical in-distribution development corpus; not the current R16 checkpointable selection |
| **Current R16/v2** | Labyrinth Draft [#59](https://github.com/thebrazenbeard/labyrinth/pull/59) head `1d38e4f25b5663bf8a84ff5e139875d4152e3680`; v2 corpus generated at `d0f2d63863873cf60747383740cc87d8f4e075a4` | `90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836` | Structure-disjoint **development** evaluation only, not a sealed final bank or training result |

The current R16 pair is enforced in `trainingday/train_neural.py` by
`R16_PRODUCER_HEAD` and `R16_V2_MANIFEST_SHA256` when
`--r16-checkpointable` is selected. These are source pins, not a fresh readback
of the owner-controlled on-host manifest, its shard bytes, notices, source
rights or evaluation permissions. The documented historical owner-host audit does
not establish present route availability or authority to train.

The Labyrinth corpus schema used by both v1 and structural-v2 manifests is
`vera.trainingday.corpus.v1`, with
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

R18's official Qwen3.5-4B snapshot is **multimodal**. Its indexed language
weights are named `model.language_model.*`, whereas the R18 text-only class
requires `model.*` and the nested `Qwen3_5TextConfig`. Both R18 arms now require
the exact SHA-bound `model.safetensors.index.json` and shard inventory, build a
complete meta-device text parameter projection, and reject missing/mismatched
physical loader keys or an untied output head. Vision/MTP keys are not training
weights. This addresses the prior source loader shape mismatch; a synthetic
CPU load is **not** a real official-weight reload. The physical host-owned
lease/rights/route launcher remains a mandatory HARD HOLD and cannot be
enabled by a source flag or self-declared token.

Match controls on source manifest, selected IDs, base revision and seed:

- `zero`, 0 optimizer steps: save initialized LoRA and record FP32 digest.
- `correct`, matched bounded steps: require the **exact zero-arm initial digest** and
  nonzero before/after tensor delta.
- `corrupt`, matched steps: per-family deterministically wrong labels, require
  the same initial digest and a separate saved adapter. An improvement also
  found in this arm does not prove reasoning learning.

The original R14/v1 machine-specific command is historical and intentionally
not a current executable example. An R16 *source-only* preflight requires
owner-verified existing values for `--manifest`, `--base-binding`,
`--producer-git-head`, `--work-root` and `--output-root`, plus an unused run ID,
`--r16-checkpointable` and `--preflight-only`. R16 trained arms require 8–32
optimizer steps on multiples of eight; a zero arm requires zero steps. Any
checkpoint restart also requires a separately pinned custody receipt.

**Physical launch: HARD HOLD.** `trainingday/physical_launch_authority.py`
currently raises `PhysicalLaunchAuthorityUnavailable` before GPU or checkpoint
effects because no independently authenticated host-owned lease, route,
process identity, data-rights grant and immutable run plan are installed.
Removing `--preflight-only` does **not** confer authority or make training
runnable. Verify resources and service ownership live; do not disable
protections, delete caches, take over unrelated processes or reuse an output
run ID. Failed-run files remain for forensics.

## Evidence ceiling

The in-job receipt records manifest hash, selected-ID hash, mode-specific
training-input hash, base revision, FP32 before/after digest, optimizer steps,
save hashes and runtime. Its status is deliberately
`ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED` until a separate fresh-process
loader confirms recovery. A saved adapter alone does not demonstrate semantic
transfer or a good Q4_K_M local GGUF. Run frozen/zero/corrupt comparisons on
newly reserved operation chains and unseen program structures before claim
promotion. The final V10 bank remains outside this source lane.

The intended *future serving destination* is a separately converted,
quantized **GGUF** for **KoboldCPP** through the **v-e-r-a** interface and
eventually VeraOS. An official BF16/safetensors base plus a saved LoRA does
not by itself produce a correct GGUF, an incorporated adapter, a tested
KoboldCPP deployment or a runtime-qualified model. Conversion, quantization,
adapter application and fresh-process serving tests remain separate gates.

## Tests

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_trainingday_admission.py \
  tests/test_trainingday_labyrinth_intake.py \
  tests/test_trainingday_neural.py
```

No training data, BF16 model weights, neural adapter output, sealed test
plaintext or private ChatGPT export belongs in this Git branch.
