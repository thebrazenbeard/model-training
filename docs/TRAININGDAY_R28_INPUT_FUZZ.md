# TRAININGDAY R28-02 — bounded public training-input adversary

Evidence class: **SOURCE + local stdlib CPU test**, not GPU/model, real-world
rights verification, live-host sandbox proof, or private-bank custody.

This is a *new isolated nonauthorizing public-toy harness*, based on exact
model-training Draft PR #160 head `4699c2ee79a75604e7ffad7f4604ee98667fc79f`.
It calls the existing `trainingday.admission.admit` through a disposable corpus
directory, after bounded byte/shape checks, and reuses the R27 supply-chain
metadata-only archive verifier without extraction. The production training
runner does **not** call this new harness; this work does not remediate every
upstream admission or source-custody threat.

## Deterministic test protocol

```bash
python3 -m unittest tests.test_r28_input_adversary -v
python3 -m unittest tests.test_r27_supply_chain tests.test_r28_input_adversary -v
git diff --check 4699c2ee79a75604e7ffad7f4604ee98667fc79f..HEAD
```

The `generate_adversarial_public_cases()` corpus is a fixed minimized list of
24 separate in-memory input abuses. Each case must return `REJECTED` with
`gpu_training_authorized=false`; permitted public fixtures return only
`PUBLIC_TOY_ADMISSION_ONLY` (also `gpu_training_authorized=false`).

The cases cover:

- Untrusted manifest paths: traversal, absolute path, Windows separator,
  forged path/split, duplicate shard spec, and excessive row counts.
- Malformed JSONL/manifest: duplicate keys, invalid UTF-8, invalid nonfinite
  numbers, oversized bytes, row shape, and independent UTF-8-surrogate guard.
- Forged corpus scope: row split, evaluation-only training use, held-out train
  flags, duplicate case IDs, normalized train/test prompt leakage.
- Declared rights: revoked rights/privacy, unsupported provenance origin,
  unapproved license, and purported OpenAI-output inclusion.
- Archive metadata only: `../` member, compression-bomb ratio, nested archive;
  no archive member is extracted or executed.
- A prompt containing the literal text of a Python shell command is carried
  as inert data; no evaluator executes it.

## Bounds and trust model

- Exactly three known synthetic shards; at most 16 KiB manifest, 32 KiB each
  shard, 64 KiB total; 32 rows/shard, 8 KiB/line; 4 MiB observed archive.
- Reject duplicate JSON keys, nonfinite numbers, invalid UTF-8 and unpaired
  surrogate code points; cap nested JSON containers to depth 12.
- Reject unlisted shard paths **before** staging anything. Only fixed
  `shards/train.jsonl`, `shards/dev.jsonl`, `shards/test.jsonl` paths are written
  below a fresh `TemporaryDirectory`; original admission runs on those copies.
- SHA checks bind only caller-provided bytes to caller-provided digest claims.
  Synthetic license/rights declarations are not authenticated.
- A public-toy acceptance is **not** proof of safe source code, live model
  training, noncontaminated protected evaluation, independent human ratings,
  or any right to train/deploy/promote.

## Negative findings and next integration gate

This wrapper deliberately tightens the public fixture ceiling and JSON parser;
the original `admission.admit()` remains callable independently and does not
inherit those limits. The current source loader does not have this new 16 KiB
preflight requirement. Thus, tests show a working *falsification harness*, not
global train/eval admission hardening. Integration into a production runner
would require a separate exact-head patch, authoritative corpus-license review,
hardware/resource lease verification, and host sandbox enforcement.

R18 historical MeZO `TIME_BUDGET_OVERRUN_INVALID` remains invalid. No paid
compute, live Executor changes, GPU job, private OOD access, merge, deployment,
or model neural-weight improvement occurred in R28-02.
