# Official Qwen3.5-4B + R16 inference: independently gated probe

This is a **source-only local probe**, not an installed Vera model, trained
checkpoint recovery receipt, authenticated conversion, or approved training
lease. It is separate from the randomly initialized CPU fixture in Draft #316.

## Required artifacts

1. Original `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
   snapshot, with two shards checked against the pinned public SHA256 values
   in `real_qwen35_inference.py`. Artifact custody/transfer evidence remains
   separate from this script.
2. A genuine GGUF derivative **converted from those exact weights**, with
   the converter source revision, command, output SHA256 and source checksums
   recorded separately. This program verifies the GGUF bytes against the
   provided SHA but cannot cryptographically reconstruct its ancestry.
3. A GGUF LoRA derivative converted from the **correct-label** R16
   `r16-v2-correct8-20261030-b` adapter; preserve the original `.safetensors`,
   `adapter_config.json`, and an independent SHA+conversion receipt. The
   deliberately corrupted-label adapter must never be labeled trained Vera.
4. A locally installed `llama-server` binary with a pinned, independently
   verified source commit and local SHA256. For current synthetic CI baseline,
   `llama.cpp@149f5641b931aaf33697d70f9abc90c3e5300247` was used; do
   not infer installation on any user host from CI.

Two older upstream Qwen3.5 LoRA conversion bugs, ggml-org/llama.cpp
[#20450](https://github.com/ggml-org/llama.cpp/issues/20450) and
[#21125](https://github.com/ggml-org/llama.cpp/issues/21125), were CLOSED
by merged [#20489](https://github.com/ggml-org/llama.cpp/pull/20489) and
[#28324](https://github.com/ggml-org/llama.cpp/pull/28324). The pinned
November 3 converter commit postdates those fixes. This does **not** prove
that this *specific* 8M-parameter R16 adapter converts; require a real test.

## Caller-created manifest (not independent provenance)

Save as a private file on VN1 or the inference host. Replace the placeholder
paths and hashes with physical artifacts. Never publish private host paths,
weights, transcripts, or potentially sensitive receipts in the public repo.

```json
{
  "schema": "TRAININGDAY_QWEN35_REAL_GGUF_PROBE_V1",
  "base_id": "Qwen/Qwen3.5-4B",
  "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "official_shard_sha256": {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
  },
  "base_gguf": {"path": "/absolute/local/base-q4_k_m.gguf", "sha256": "replace-with-sha256"},
  "llama_server": {"path": "/absolute/local/llama-server", "sha256": "replace-with-sha256"},
  "r16_correct8_adapter_gguf": {
    "path": "/absolute/local/r16-correct8.gguf",
    "sha256": "replace-with-sha256",
    "source_run": "r16-v2-correct8-20261030-b"
  }
}
```

Use the actual host-local absolute path form: `C:\\...` on Windows or `/...`
on Linux. For each mode run `preflight` before `probe`:

```bash
python -m trainingday.real_qwen35_inference preflight --manifest PRIVATE.json --mode base
python -m trainingday.real_qwen35_inference preflight --manifest PRIVATE.json --mode trained
python -m trainingday.real_qwen35_inference probe --manifest PRIVATE.json --mode base
python -m trainingday.real_qwen35_inference probe --manifest PRIVATE.json --mode trained
```

No `probe` call occurs during `preflight`; no model server is started. The
explicit `probe` command verifies files, starts **one owned temporary process**
on a random loopback port, uses `--n-gpu-layers 0`, runs a fixed public prompt,
records API output/timing and terminates only its child. On a host with
insufficient free RAM or an active competing workload, **do not invoke** it.
All commands presume sufficient CPU memory and a validated conversion; those
preconditions must be checked independently.

### Evidence boundary

- A successful manifest check proves *local bytes match caller-supplied hashes*
  and the public official source digest declaration matches the pinned constants.
- It does **not** prove that the derivative GGUF came from the official shards,
  that an adapter is the original R16 `.safetensors`, or that either file was
  independently admitted for training. The receipt reports these as false.
- A fresh successful `probe` proves one actual `llama-server` instance loaded
  the named GGUF and returned one completion through a bounded local request.
- Neither a successful load nor different generated text proves learning,
  reasoning generalization, quality improvement, correct production stopping,
  GPU fit, VeraOS integration, or a persistent serving deployment.

## Focused validation

Run `python -m unittest -q tests.test_real_qwen35_inference` from repo root.
No weight download, GPU, model server, or external service is needed for the
offline negative contract tests. Only the real-model integration exercise may
claim a loaded official derivative, and only after conversion ancestry checks.
