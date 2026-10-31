# TrainingDay R30 — matched four-arm control schedule (public CPU toy)

**Source evidence only.** This opt-in module extends R29's optimizer-scalar
CE witness with a disjoint check on the *declared control quartet*. It does not
read a host, a model, a private corpus, or a protected evaluation bank.

R29 already covers target-token ID ordering, microbatch-weighted causal LM
cross-entropy, MeZO modulo cycling versus a single QLoRA sweep, optimizer
step/sample-token exposure, declared clock/Modern Standby integrity and an
unknown effective AdamW8bit default. R30 does not redefine those source
contracts or infer real training from them.

`assess_control_quartet()` accepts exactly four public toy arms in order:
`FROZEN`, `ZERO_UPDATE`, `CORRECT_LABEL`, `CORRUPTED_LABEL`. It copies their
bounded builtin data into private immutable dataclasses before calling the
R29 toy validators. It then checks:

- Same case IDs, case order and masked prompt input across four controls.
- Same supervision masks; only completion target labels can change for the
  corrupted-label control, which **must** actually alter at least one target.
- Same explicit correct/corrupted case-presentation order, cohort coverage,
  labeled optimizer update count and declared initial adapter bytes.
- Frozen and zero controls have zero presentations/updates and unchanged
  declared adapter hashes, whereas labeled controls report updates and
  distinct before/after adapter hashes.
- Identical toy frozen logits in the three controls that declare exactly the
  same uncorrupted input and labels; no requirement that corrupted completion
  logits match after the adversarial answer changes its autoregressive inputs.
  Correct/corrupted logits **must** match at every supervised prediction
  position whose full causal input prefix is identical. A target-label edit
  alone cannot explain different frozen-model logits for the same prefix.

The module emits only SHA-256 commitments, declared presentation counts,
explicit failed source conditions, and permanently false independent
custody/physical-tensor/behavior/release flags. No individual corpus text,
token IDs, labels or logits are returned. A positive synthetic consistency
result is **never** a model-improvement result or training permission.

## Tests

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q tests.test_r30_comparator_controls
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_r2*.py' -q
```

The new tests include a same-ID/different-prompt adversary (which R29's
declared control hashes cannot themselves detect), an equal-count but
reordered exposure, unchanged supposedly corrupted targets, supervision-mask
drift, frozen-logit and same-prefix logit drift, unequal initialization,
zero-update mutation, no trained update, malformed role order, and
post-validation caller mutation.

## Qualification boundary and negative findings

The four declarations still come from one untrusted source; their adapter
hashes and per-case token IDs are not independently read from disk, and the
private snapshot is not an OS-atomic capture of a concurrent external system.
Model training cost, actual optimizer work, batch/sequence fit, independent
controls, control behavior, and adapter byte changes require fresh physical
evidence. The R18 MeZO run's **531s** planned vs **19,813.562s** elapsed,
including **19,698.314s** Modern Standby, remains
`TIME_BUDGET_OVERRUN_INVALID` as a fair equal-time pair. R18 QLoRA source
omits explicit AdamW8bit `weight_decay`; a source scan cannot establish the
runtime effective default. Physical equal-time MeZO-vs-QLoRA comparison
remains unqualified until a leased host and admitted identical corpus are
available; `training_authorized` and `model_promotion_authorized` stay false.
