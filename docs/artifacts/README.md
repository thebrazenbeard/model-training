# Public Qwen3.5 4B base-model restore

The [pinned manifest](qwen35-4b-public-base-20261102.json) records the publicly retrievable upstream base for the `qwen35-4b-hf-line` training track. It does **not** contain locally trained checkpoints, adapters, private datasets, or any assertion that a training run is qualified.

The two model-weight shards (9,319,828,160 bytes total) were independently SHA-256 checked against Hugging Face blob metadata at the immutable revision `d61dd146c8fd44c9a49cdb7f59f34e17b61902d8` on 2026-11-02.

## Restore only when a job needs the base

Install the `huggingface_hub` CLI from its official distribution, then run on the intended worker:

```sh
hf download rodrigomt/Qwen3.5-4B-Uncensored-Aggressive --revision d61dd146c8fd44c9a49cdb7f59f34e17b61902d8 --local-dir /absolute/worker/qwen35-base
```

On Windows, substitute a selected absolute directory for the `--local-dir` argument. The pinned upstream repository was public and ungated when checked; future service availability, licensing, and network access are separate prerequisites.

After download, verify each weight shard's exact size and SHA-256 against the manifest. On Linux use `sha256sum`; on Windows use PowerShell `Get-FileHash -Algorithm SHA256`. Both expected digests and file sizes are in JSON.

## Retention boundary

This is **base-model provenance**, not proof of independent recovery for fine-tuned weights. Do not delete any training output or local cache merely because this manifest exists. Before evicting an on-disk cache, check current process and scheduled-job dependencies, record an intended restore location, and confirm that the pinned upstream can still be fetched. The cache may be rehydrated on demand rather than kept permanently on a laptop.
