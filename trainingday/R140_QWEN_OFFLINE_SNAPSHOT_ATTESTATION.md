# R140 — offline Qwen3.5-4B snapshot inventory witness

This is an additive, **read-only, CPU/source-only** follow-up to R75. It
retains the official pinned base ID `Qwen/Qwen3.5-4B` and revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It does **not** make
the experimental R18 MeZO/QLoRA training runs eligible to start.

Primary source observations at the exact pinned revision:

- [Official model tree](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a):
  two `model.safetensors-0000X-of-00002.safetensors` shards plus index;
  `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`,
  and `chat_template.jinja`.
- [Official shard index](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors.index.json):
  `metadata.total_size` and per-tensor shard binding. Index reports a total
  of 9,319,737,856 tensor data bytes (distinct from total on-disk shard size).
- [Official tokenizer config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer_config.json):
  declares `Qwen2Tokenizer`, EOS, and a chat template.
- [Transformers Qwen3.5 docs](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5):
  `Qwen3_5ForCausalLM` supports text-only use with `Qwen3_5TextConfig`,
  whereas the official full configuration declares
  `Qwen3_5ForConditionalGeneration`. This is a compatibility *possibility*,
  not evidence that the installed loader can load the pinned checkpoint.

## Invocation (only after separately authenticating the pins)

```bash
python -m trainingday.r140_qwen_attestation \
  --base-binding /path/to/existing/base-binding.json \
  --expected-binding-sha256 <independent-64-lowercase-hex> \
  --expected-config-sha256 <independent-64-lowercase-hex> \
  --expected-tokenizer-sha256 <independent-64-lowercase-hex> \
  --source-root /path/to/pinned-model-training-checkout
```

The witness rehashes every base-binding file, parses the source/config
declarations through R75, checks the pinned tokenizer/template assets, parses the
safetensors index and each shard's bounded header, verifies tensor
shape/dtype/byte-offset consistency and rejects index/shard mismatches,
unindexed shards, malformed JSON, symlinks at inspected file paths and
unbound/missing tokenizer assets. It never deserializes tensor data.

The only successful status is
`SOURCE_SNAPSHOT_INVENTORY_VALID_RUNTIME_HOLD`; every authority/runtime
boolean stays false. In particular:

1. Caller-supplied SHA pins are not authenticated by this witness. A producer
   can forge a binding and matching hashes; upstream artifact provenance must
   be independently checked against the official pinned source.
2. Reopening by pathname is not immutable host custody. Concurrent path
   swaps or in-place changes may defeat a repeatable snapshot observation.
   Host-owned descriptor/handle custody and authority receipts are separate.
3. Reading safetensors headers and hashing file bytes does **not** load weights
   through Transformers. A host-authorized fresh process must record actual
   loader/config class resolution, strict loading warnings, tokenizer IDs,
   tensor placement, finite text-only forward loss, hardware fit and adapter
   continuity under an exclusive CUDA owner lease.
4. No synthetic CPU fixture, model review, or this source inspection grants
   the R19-01 host lease, unlocks a sealed evaluation, merges a PR or asserts
   real neural learning.

Tests (no torch/transformers dependencies):

```bash
python -m unittest discover -s tests -p 'test_r140_qwen_attestation.py' -v
python -m unittest discover -s tests -p 'test_r75_qwen_loader_preflight.py' -v
```

This module is **not wired into the active R18 runner**, intentionally; host
authority/immutability and actual Qwen runtime testing must be reviewed first.
