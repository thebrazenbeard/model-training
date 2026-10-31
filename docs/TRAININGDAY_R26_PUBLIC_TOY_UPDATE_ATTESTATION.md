# TrainingDay R26-02 — Public Toy Neural-Update No-Op Falsifier

**Evidence class: source inspection + real local CPU toy tensor bytes + stdlib tests.**

This is not evidence that the Qwen3.5-4B model was trained, generalized,
survived quantization or is safe to promote. It never starts a GPU job,
opens the private bank, reads a corpus shard or changes a running service.

`trainingday/r26_toy_update_attestation.py` accepts an intentionally tiny,
public, fixed two-parameter linear SGD fixture. The source reads actual
little-endian FP64 bytes from `adapter_before.bin` and `adapter_after.bin`,
checks external caller-provided base/tokenizer/initial-adapter pins, and
replays claimed steps against fixed public orthogonal correct/corrupted labels.
Per-step digest commitments and final tensor bytes must match independently
computed deterministic updates. A fresh Python interpreter separately loads
the saved tensor bytes, parses two finite parameters, and reports a digest.

Three independent control folders must be present: `zero`, `correct`,
`corrupt`. A zero arm has no update; a correct-label arm must improve the
fixed public *toy training-family* squared-error score; an intentionally
corrupted-label arm moves in the opposite direction. The caller cannot
relabel an unchanged adapter as positive SGD simply by recomputing hashes.
The three controls must share an identical bounded learning rate, and the
two trained arms must share the same optimizer-step budget; an unmatched
schedule fails closed instead of yielding a misleading comparison.

Each folder contains these exact names (no dynamic executable paths):

- `base.bin` and `tokenizer.txt` — public synthetic identity bytes.
- `adapter_before.bin` and `adapter_after.bin` — the toy weight bytes.
- `manifest.json` — fixed schema, mode, 0–16 claimed optimizer steps,
  bounded learning rate, four exact file SHA-256s, and step-by-step hashes.

The independent pins and step declarations are **not signed or verified by an
independent custodian**. A fully compromised caller who regenerates data,
fixtures and all pins can manufacture a coherent *toy* receipt. Fresh Python
reloading the fixed tensor format is not a `torch`/`peft`/Qwen model reload,
nor a new-process behavioral test. The positive toy-score change is not
unseen-task reasoning improvement. Even when all source checks pass:

`qwen35_neural_update_verified=false`,
`behavioral_generalization_verified=false`,
`host_lease_authenticated=false`,
`authorizes_gpu_start=false`, and `authorizes_promotion=false`.

Run focused tests from the model-training repo root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest -q tests.test_r26_toy_neural_update_attestation
```

The positive path demands actual changed toy weight bytes and a fresh
interpreter response. Negative probes include a forged unchanged adapter,
misdeclared optimizer step count, repinned corrupted-label arm, mutated
base/tokenizer/initial bytes, malformed large/boolean/NaN steps, duplicate
JSON keys, enormous integer rates/weights, mismatched correct/corrupt
budgets or learning rates, zero-update mislabeling, and a symlinked base file.

To advance the physical Qwen3.5-4B claim, a separate independently controlled
test must load genuine frozen-base and LoRA tensors from verifiable artifacts,
check actual optimizer and process receipts, independently rehydrate the
adapter, and run untouched scored behavioral controls under a valid host lease.
The R18 original remains `TIME_BUDGET_OVERRUN_INVALID` and does not provide
an admitted MeZO-versus-QLoRA comparator. This module intentionally cannot
promote any of those claims.
