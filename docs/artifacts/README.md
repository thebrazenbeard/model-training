# Public Qwen3.5 4B base-model restore

For five separately checked public Hugging Face cache weights (Whisper and MiniLM), see the [pinned public-cache manifest](public-hf-cache-models-20261102.json). That record is not proof of a running consumer or authorization to delete local caches.

## Shared OCD verification catalog (seven public weight files)

The [generated OCD verifier manifest](ocd-verify-public-hf-20261102.json) combines **two Qwen weight shards** and **five distinct Whisper/MiniLM weights** into one read-only verification interface, while their original source manifests remain authoritative. Regenerate its contents with `python -m trainingday.public_artifact_catalog` and check the committed catalog with `python -m trainingday.public_artifact_catalog --check`. These standard-library commands do not download, install, load, or delete anything.

After using the individual pinned `hf download` commands below on an intended worker, verify each downloaded file from the worker's selected root, for example:

```sh
ocd artifact verify --manifest docs/artifacts/ocd-verify-public-hf-20261102.json --destination-root ABSOLUTE_WORKER_DIRECTORY --artifact faster-whisper-tiny-en/model.bin --json
```

Run the command from this repository checkout with OCD installed. Use an actual absolute root in place of the placeholder. The other six `--artifact` paths are listed in the manifest. OCD verification checks *only* the selected file's size and SHA-256: it does **not** prove a complete runnable model, live worker authorization, source rights, upload retention, a restored job load, or permission to remove Lappy's fallback weights. This public cache catalog is separate from the private BT2 training outputs in `thebrazenbeard/bt2-model-artifacts`, and separate from official Qwen/Qwen3.5-4B R18 TrainingDay source binding.

Replace the ABSOLUTE_WORKER_DIRECTORY placeholder in a manifest command with the selected absolute destination on VN1, VN2, or another worker. Each command selects its own model-specific subdirectory to prevent unrelated models named model.bin from overwriting one another. Download only what a job needs and verify the restored file's SHA-256 and size; single cached weight entries do not necessarily contain every tokenizer/config file a model runtime needs.

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
