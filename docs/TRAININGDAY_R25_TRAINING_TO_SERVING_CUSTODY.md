# R25 — Training-to-serving custody candidate (source-only)

This is a CPU-sized rehearsal for a possible future Qwen3.5-4B local serving
candidate. It does **not** load, quantize, install, run, deploy or replace any
model. In particular, `vera-local:latest` and all running services are untouched.

## Inputs and evidence classes

`trainingday.r25_serving_envelope.assess_serving_candidate(pinned, candidate)`
requires pinned exact official `Qwen/Qwen3.5-4B` architecture/revision and
separate base, tokenizer, corpus, adapter bytes and LoRA tensor-manifest
digests. The candidate must match all pins. Tensor metadata must be uniquely
named FP32 LoRA A/B with sane matrix shapes and a canonical metadata hash.
Each A/B pair must have the same module path and named adapter namespace, with
matching low-rank dimension (`A: [rank, in]`; `B: [out, rank]`). Missing partners,
cross-layer or cross-adapter A/B substitutions, rank mismatches and nonmatrix
tensor metadata are rejected. A separate **64 MiB declared FP32 adapter
metadata ceiling** prevents plainly absurd tensor claims after repinning. It
is a source admission limit, **not** evidence that an actual 4 GiB GPU can fit
the quantized model, adapter or training state.

The transformation envelope binds the NF4/double/BF16 training source to a
*declared* GGUF Q5_K_S output, including converter binary/argument, output
and transformation-receipt SHA256 values. A different model family, a hybrid
or derivative architecture, a substituted adapter, incompatible tensor
layout, missing converter identity, malformed digest, or unexpected filesystem
path field is rejected.

The optional `fresh_process_reload` is an **untrusted claim**. Matching digest
strings do not prove that independent code actually loaded the model or that
the quantizer produced a behaviorally equivalent output. All decisions remain
`status=HOLD`, `authorizes_deployment=false`, and
`authorizes_model_promotion=false`.

## Independent proof required before a real candidate decision

1. Verify source-host artifact bytes against a separately protected immutable
   base/tokenizer/adapter inventory, with real tensor names, shapes and dtype.
2. Record exact quantizer/converter binary and flags, input/output hashes,
   architecture and tokenizer compatibility, and quantization quality loss.
3. Load the candidate in a **fresh independent process** and compare deterministic
   public prompts and adapter tensor hashes with the pinned source model.
4. Measure peak host/GPU RAM, load time, throughput, and retained capabilities;
   run frozen, zero, correct-label and corrupted-label controls on untouched
   admitted public probes. Preserve protected evaluation bank isolation.
5. Require independent resource leases, host power/standby evidence,
   authorized release approval, and a physical atomic rollback receipt.

No such hardware/behavioral/independent attestation is claimed by this code.
The historical R18 overrun remains `TIME_BUDGET_OVERRUN_INVALID` and cannot
be repurposed as a fair equal-time MeZO/QLoRA or release comparison.

## Reversible state rehearsal

`simulate_reversible_cutover(previous, candidate, fail_at=...)` records a
pure in-memory five-transition trace (`STAGE`, `VERIFY`, `SWAP`, `POSTCHECK`,
`COMMIT`). Each synthetic injected failure restores the previous *simulated*
active hash and preserves the previous source reference. Even a synthetic
success marks `production_active_untouched=true` and has no executable
service endpoint. It is neither an atomic Windows/Linux cutover nor rollback
evidence for a running model.

## Acceptance/negative probes

`tests/test_r25_serving_envelope.py` covers identical pinned fake artifacts,
hybrid/derivative architecture refusal, adapter/quantizer/tensor substitution,
explicit fake fresh-reload claim, malformed/extra/private-path fields, and
failure injection at every reversible transition. Synthetic hashes are public
fixtures; no private data, real credentials, or actual Qwen tensor bytes are
used.
