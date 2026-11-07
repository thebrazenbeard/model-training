# Neuro-Sync PEFT adapter-only → TRAININGDAY source handoff (HOLD)

**Scope: source-only compatibility. Not model, weights, training, evaluator, or host validation.**

`trainingday.neurosync_adapter_only_compat` accepts a Neuro-Sync
`NEUROSYNC_PEFT_ADAPTER_ONLY_INTENT_V1` and its matching
`NEUROSYNC_TRAININGDAY_PEFT_ADAPTER_ONLY_ENVELOPE_V1`. It checks the
complete external V1 field topology and declared PEFT geometry, verifies
the envelope's canonical self-declared intent hash/digest role join, and
requires the exact **official** Qwen3.5-4B source revision and ordered shard
metadata from `trainingday.official_public_base`.
Train, sealed holdout and shuffled-control manifest hashes must also be
distinct from every declared weight/tokenizer/config role, even if a caller
assigns coherent byte sizes to the same digest.

```python
from trainingday.neurosync_adapter_only_compat import (
    check_adapter_only_source_handoff,
    expected_official_source_bundle,
)

source_only = check_adapter_only_source_handoff(
    neurosync_untrusted_intent,
    neurosync_untrusted_envelope,
    expected_official_source_bundle(),
)
assert source_only["trainingday_ready"] is False
assert source_only["physical_launch_authority"] == "HOLD"
```

### Two hashes are **not** interchangeable

The official Qwen source metadata pins **two separate safetensors shards**
(5,329,398,688 and 3,990,429,408 bytes); the expected source bundle
provides an ordered, domain-separated `manifest_sha256` over the pinned
repository/revision/shard *metadata*. It is **not** a SHA-256 of concatenated
weights, a single shard, nor a verified copy of loaded model weights.

Neuro-Sync's one `base_weights_sha256` remains an *opaque untrusted declared
model-byte reference* until a separately specified, reproducible multi-shard
full-byte identity/custody procedure exists. This bridge rejects using either
one official shard hash or the source metadata bundle hash in that field; it
cannot distinguish an otherwise invented digest from real model bytes.

The other `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` public inventory in
this repository is a different model and revision. Its two shards cannot
substitute for the official pinned training base. The official checkpoint
source metadata still lacks trusted byte pins for seven ancillary files,
including tokenizer/chat-template assets, as recorded in
`trainingday.official_public_base.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS`.
One opaque tokenizer SHA is not proof of the full tokenizer/chat-template
bundle either.

### Still unverified—strictly no launch authority

- Actual local official shard and ancillary byte identity, custody and signer.
- Training data rights, adapter weights/config integrity and optimizer receipt.
- PEFT load/merge/target compatibility, actual resumed training, loss curves.
- Donor-disconnected independent cold reload, untouched and shuffled controls,
  sealed dialogue evaluation, GGUF conversion, and serving PID binding.
- Host-owned GPU lease, WorkBridge route, and physical launch authorization.

The return value always has `base_weight_bytes_verified=false`,
`training_data_rights_verified=false`, `adapter_loaded_and_evaluated=false`,
`trainingday_ready=false`, and `physical_launch_authority=HOLD` even when
`source_bundle_aligned=true`. It must never be used to clear the separate
physical gates in model-training's Draft #351. No model is loaded or trained.

Cross-repository contract under review: Neuro-Sync Draft #61 (adapter-only
source intent), Draft #38 (local PEFT inspector). This is a proposed consumer
seam only; external draft branches are not installed runtime dependencies.
