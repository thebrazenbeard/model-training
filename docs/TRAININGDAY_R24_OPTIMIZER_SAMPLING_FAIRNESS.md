# TrainingDay R24: optimizer defaults and case-exposure falsifiers

**Status:** SOURCE / SYNTHETIC-CPU ONLY — `HOLD`, not a physical training gate.

Source reference for this isolated R24 branch: model-training Draft #149,
remote head `f7a4c70efae1987fd06c5f5576320e545baf8518`, source-equivalent
local Git tree `2f977515ff6571b0e6bf8debb886597e842a1d85` before this patch.

The R18 comparison already uses target-token-weighted supervised loss in both
arms, but identical scalar loss objectives do not imply identical *optimization
or data presentation*. Two differences must be made explicit:

1. `trainingday/r18_qlora.py` builds `AdamW8bit(..., lr=lr)` **without** a
   `weight_decay` keyword. Its effective value is not proven by source and
   depends on the installed optimizer package/runtime. The R24 AST inspector
   distinguishes omitted, dynamic, ambiguous and explicit constant syntax;
   even an explicit constant would not verify installed optimizer behavior.
2. `trainingday/r18_mezo.py` selects a per-step batch with modulo indexing,
   allowing examples to repeat after a full cycle. **Its effective batch is a
   frozen-plan input**, not source-hardcoded to eight. The R18 QLoRA runner
   fixes `EFFECTIVE_BATCH=8`, makes one sweep and rejects a repeated case.
   Equal elapsed time can therefore produce different example and token
   exposure. MeZO uses two forward presentations per update, while QLoRA
   performs one forward plus one backward exposure; these are **not** the
   same computational costs or gradient mechanics.

## Runnable synthetic evidence

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest \
  tests.test_r24_optimizer_sampler_audit -v
```

`trainingday/r24_optimizer_sampler_audit.py` provides:

- `inspect_adamw8bit_weight_decay(source_file)` — source-syntax evidence
  with SHA-256 of inspected file and no guessed optimizer default.
- `simulate_source_exposure(arm, public_or_synthetic_cases, timings, max_seq_length)`
  — per-step case IDs, per-ID repetition counts, unique/presented examples,
  input and supervised target tokens, forward/backward token counts, padded
  tokens, and declared optimizer-step elapsed time and budget.
  `examples_per_update` is explicit and checked; the default eight is a
  synthetic comparison assumption, not proof of the MeZO frozen plan.
- `assess_source_and_exposure(mezo, qlora, source_evidence)` — deterministic
  source-policy, missing/default decay, example repetition, per-case mismatch,
  token exposure, and optimizer-window mismatch findings. All outcomes are
  explicitly `HOLD` with `authorizes_gpu_start=false` and
  `proves_neural_improvement=false`.

### One falsifier

With 16 synthetic cases and batch size 8, two MeZO updates and two QLoRA
updates present each case once. A third MeZO update wraps around and repeats
the first eight cases; QLoRA's source policy has no matching third update.
Both could fit the same **declared** wall-clock budget, yet they have different
example counts and input/target-token presentations. This invalidates any
assumption that equal budget alone proves equal sampled data exposure.
Another negative fixture uses a four-example MeZO update against the source-
fixed eight-example QLoRA update; the audit flags the effective-batch mismatch.

## Required before physical comparability

- Record the exact official base/tokenizer/corpus hashes and match actual
  token-ID, label-mask and per-case selection ledgers across arms.
- Pin and attest the effective AdamW8bit package version, configured
  `weight_decay`, and runtime value, plus MeZO's lack of corresponding decay.
- Pre-register equal effective compute/time accounting and a case-exposure
  policy. If a MeZO cycle repeats cases while QLoRA stops, characterize it as
  unequal exposure; do not hide the difference by averaging losses.
- Independently witness exact host/GPU lease, elapsed active optimizer time,
  Modern Standby exclusions, tensors before/after and fresh-process reload.
- Keep frozen/zero-update, correct-label and corrupted-label controls, and
  untouched protected evaluation under independent custody.

This R24 module neither modifies the training runners nor has authority to
execute, unseal, evaluate, merge, release, or retrospectively validate the
historical R18 MeZO time-budget overrun (`TIME_BUDGET_OVERRUN_INVALID`).
