# Official Qwen3.5-4B source-only custody candidate

This is a **separate identity** from
[`qwen35-4b-public-base-20261102.json`](qwen35-4b-public-base-20261102.json),
which records the modified third-party
`rodrigomt/Qwen3.5-4B-Uncensored-Aggressive@d61dd146…` public cache.
That existing cache record is retained for its historical OCD use, but is **not**
the official `Qwen/Qwen3.5-4B` TrainingDay checkpoint.

The [official identity manifest](qwen35-4b-OFFICIAL-trainingday-source-20261104.json)
pins upstream repository, commit, shard **physical** sizes, and Git LFS
SHA-256 object identifiers from public metadata at immutable revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Neither shard was downloaded
or locally hashed by this review. Primary upstream sources:

- [Pinned checkpoint index](https://huggingface.co/Qwen/Qwen3.5-4B/raw/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors.index.json)
- [Shard 1 LFS pointer](https://huggingface.co/Qwen/Qwen3.5-4B/raw/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors-00001-of-00002.safetensors)
- [Shard 2 LFS pointer](https://huggingface.co/Qwen/Qwen3.5-4B/raw/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors-00002-of-00002.safetensors)
- [Pinned nested config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)

The indexed tensor sum is `9,319,737,856` bytes; physical shard sum is
`9,319,828,096` bytes (90,240 bytes of packaging/header difference). No local
source hash, trusted approval or separate owner signature is implied.

The official checkpoint's `config.json` names
`Qwen3_5ForConditionalGeneration`, with nested GDN/hybrid text tower and
`model.language_model.*` checkpoint keys. Text-only training proposes
`Qwen3_5ForCausalLM` only **after** a separately verified complete projection;
the latter class is not the checkpoint's upstream `architectures` value.
KoboldCPP/GGUF inference is a different qualification path from real BF16/NF4
optimizer training and cannot stand in for it.

## Contract and non-claims

`python -m unittest discover -s tests -p 'test_trainingday_official_public_base.py'` tests that
the source manifest exactly matches compiled official metadata pins and rejects
third-party/revision/architecture/shard substitutions, including those with
otherwise similar shapes. This is an offline, mock-free metadata test; it does
not load actual weights.

The existing `trainingday/pinned_base.py` checks a **caller-supplied** local
binding and hashes local files. That local match alone cannot authenticate a
trusted source. A physical broker must independently verify:

1. Immutable reviewed code and manifest digest, trusted owner/signature trust
   root and unique non-replayable grant, bound to the exact host/job/expiry.
2. Official pinned config, tokenizer, tokenizer_config, index, chat template,
   BPE merges and vocab SHA-256 byte values from independently authenticated custody (not caller assertions),
   then exact local byte hashes/sizes of every shard immediately before use.
3. Complete text checkpoint projection and tensor-load evidence, with no
   fallback to a shape-compatible fork or a GGUF inference artifact.
4. Contract B's bounded, **nonexclusive WDDM** headroom and foreign-process
   reservation, abort/restore behavior, authorized public/synthetic-only data,
   and any host-specific lease/consent conditions.

Until those gates are independently satisfied this is **SOURCE ONLY / NOT
SIGNED / NOT BROKER ADMITTED / NO PHYSICAL LAUNCH**. No weights belong in Git;
VN1/VN2 may restore artifacts on demand under separately verified worker
authority, preserving Patrick's no-permanent-Lappy-weight policy.

## 2026-11-04 partial-weight hostile counterexample

On the source main branch, the earlier `require_official_base()` accepted a
caller-provided 5-file binding with one tiny fake safetensors shard. Both a
one-shard and a two-shard fake fixture reproduced this defect in red-first
Linux CI. The proposed production-facing function now demands the **exact two**
upstream shard names, independently reviewed full-byte SHA-256 OIDs, official
physical byte lengths, a structural two-shard index, and every required
config/tokenizer/index/ChatML template/vocabulary artifact. Synthetic hash
fixtures remain testable only by
`inspect_untrusted_base_binding_for_native_test_only`—a deliberately
non-authoritative inspection function.

The seven small metadata artifacts (`config.json`, `tokenizer_config.json`,
`tokenizer.json`, `model.safetensors.index.json`, `chat_template.jinja`,
`merges.txt`, and `vocab.json`) still lack **independently admitted full-byte
SHA-256 anchors**. Production `require_official_base()` has an empty trusted
metadata pin set and therefore **HOLDs all candidates**, even if both real
weight shards are present. The official repo uses Xet backing for the two
weight objects; pointer text and declared sizes are metadata, never the
physical tensors. No trusted tokenizer/chat-template equivalence, full tensor
header integrity, owner signer, WDDM fit, GPU lease or optimizer step is
implied by this source work.
