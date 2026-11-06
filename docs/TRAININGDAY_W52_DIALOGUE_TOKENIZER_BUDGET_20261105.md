# W52 real local tokenizer preflight — 80 fictional dialogue candidates

**Result:** `MEASURED_PINNED_PUBLIC_CASES_ONLY` (read-only tokenizer probe, Lappy,
2026-11-05 UTC). This is **not a model inference, training, rights-admission,
or full runtime/persona-prefix test**. The diagnostic opened tokenizer/config
files, not official model weight tensors; no downloads, installs, explicit
output files, or training operations were requested. Ordinary Python import
cache or operating-system bookkeeping effects were not independently audited.

## Exact input and tokenizer provenance

- Corpus: Draft PR #335 source-only `trainingday/dialogue_curriculum.py`
  candidate at commit [`b605a31d6922689d572a09c4b8365e3cd90e7165`](https://github.com/thebrazenbeard/model-training/commit/b605a31d6922689d572a09c4b8365e3cd90e7165),
  canonical candidate SHA-256
  `4fbe54d97504628aa88ac7be37d994ed5f39be8d2a73db99f4cf54684989f31e`.
- Host: `Lappy`, Windows, read-only WorkBridge PowerShell `py`; Python 3.14.5,
  `transformers==5.14.1`, `tokenizers==0.22.2`.
- Exact installed offline tokenizer snapshot (no weight loading):
  `D:\VERA\scratch\trainingday-official-qwen35-4b-851bf6e-20261101`.
- Snapshot's `TRAININGDAY_PUBLIC_SOURCE_RECEIPT.json` reports official
  `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, recorded 2026-11-01;
  this is source receipt provenance, **not a fresh verification of the 9.3GB
  model weight shards**. Current independently computed tokenizer-file SHA-256:

| File | Current SHA-256 (matches the Nov 1 source receipt) |
|---|---|
| `chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |

`AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
trust_remote_code=False)` returned `Qwen2Tokenizer`. `apply_chat_template`
was applied to every conversation; inference prompt lengths exclude the last
assistant answer and add a generation prompt; training full-sequence lengths
include that answer. **No system/persona prefix was supplied**, since an
independently identified, legal, exact, versioned full rendered prefix
(including gateway note) was not available to this probe.
The reproducible source now fails closed on the exact frozen candidate
SHA-256, three exact tokenizer file hashes, and local source receipt's model
ID/revision and recorded file hashes. This binds measured *local bytes* to
the November 1 receipt; it does not independently authenticate the remote
official revision or the pretrained weight shards.

## Actual per-case aggregate token counts

Default chat-template mode (all 80 cases, no private system prefix):

| Split and view | N | Min | Median | Max | `>64` | `>96` | `>128` | `>256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN full sequence | 64 | 30 | 45 | 71 | 9 | 0 | 0 | 0 |
| DEV full sequence | 16 | 34 | 45.5 | 70 | 2 | 0 | 0 | 0 |
| TRAIN prompt-only | 64 | 14 | 21.5 | 56 | 0 | 0 | 0 | 0 |
| DEV prompt-only | 16 | 18 | 21 | 55 | 0 | 0 | 0 | 0 |

With `enable_thinking=False`, full-sequence counts remained identical; prompt
only counts became TRAIN min16/median23.5/max58 and DEV min20/median23/max57,
all under64. Both modes used the same pinned tokenizer files.

**Important implementation falsifier:** host Transformers returned a two-key
`BatchEncoding` (`input_ids`, `attention_mask`). The first provisional script
incorrectly used `len(BatchEncoding)` and misleadingly returned **2** for
every case. That result was **rejected**, replaced with
`len(encoded["input_ids"])`, independently checked against a 16-token sample,
and regression-tested. Only the corrected counts above are valid.

## Qualification boundaries

> The full Vera persona and gateway system prefix remain **UNMEASURED**. This
> 80-case pilot fits 96 tokens *only without that prefix*. The documented
> ~1,751-character owner persona seed plus any gateway note could change every
> count substantially. No inference 64/96/128/256 context conclusion follows.

- Do not confuse SFT **whole input+answer sequence** token budget with
  runtime **prompt prefix + user dialogue + response-generation cap**, or
  with KoboldCPP context allocation/KV cache.
- `Qwen2Tokenizer` + official HF Jinja measurements are not a fresh GGUF
  conversion/tokenizer-equivalence proof. KoboldCPP stopping behavior,
  thinking mode, real generation and hidden state remain untested here.
- These tiny fictional examples have a rights HOLD, and their short lengths
  say nothing about natural conversation quality, learning, or generalization.
- When the precise full rendered runtime prefix is identifiable and locally
  authorized, run `python -m trainingday.dialogue_token_budget_probe
  --snapshot <existing-local-snapshot> --system-file <private-rendered-prefix>`.
  This prints aggregate counts only; never commit the private prefix file or
  raw prompt content. The prefix must already include the actual versioned
  owner persona plus gateway system note, with no silent truncation. The probe
  can measure supplied bytes but does **not independently authenticate** that
  the file contains the complete installed runtime prefix.

Focused source-only regression tests:
`python -m unittest tests.test_trainingday_dialogue_token_budget`.
