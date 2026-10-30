# TrainingDay R22: prospective equal-time work forecast (source only)

This small calculator consumes an already validated R18 Qwen3.5-4B plan and
two strictly train-only, **untrusted** throughput declarations. It commits to
the same pinned base revision, admitted training selection, corpus bytes, seed,
531-second illustrative optimizer-wall budget, and four controls per method:
`frozen`, `zero`, `correct`, `corrupt`. The calibration ceiling comes from the
existing plan's `probe_budget_seconds`; a declaration beyond it is rejected.

```powershell
python -m trainingday.r22_equal_cost --plan R18-PLAN.json --profiles R22-PROFILES.json
```

Output is `UNTRUSTED_FORECAST_NOT_AN_EXPERIMENT` with exit **3**, not a launch
permit. Incomplete/invalid input yields HOLD with exit **2**. GPU and model
promotion authorization are false in every case. No real profile is bundled or
fabricated by this tool; the user must supply actual declared calibration
figures and independently authenticate them elsewhere.

## Mathematics and interpretation

- MeZO and QLoRA both optimize a **supervised-target-token weighted mean** of
  microbatch means: `sum(loss_i * target_tokens_i) / sum(target_tokens_i)`.
  Microbatches of 1 and 3 target tokens with losses 2 and 8 yield **6.5**, not
  the naive unweighted **5**. This matches the R19 objective correction, whose
  local source must still be integrated into a qualified runnable build.
- Both methods get the same preregistered optimizer-wall budget; this is
  intentionally **not** a claim of the same presentations, useful tokens,
  padded tokens, forward passes, backward passes, or optimizer steps. Those
  rates vary by method, fit configuration, and hardware conditions and must be
  reported separately. MeZO's two-sided perturbation requires at least twice
  its useful input tokens as forward sampled tokens; QLoRA must report actual
  backward token work.
- The report projects each declared counter linearly by
  `training_budget_seconds / calibration_elapsed_seconds` solely to expose
  scale differences. Linear extrapolation does not predict thermal throttling,
  memory pressure, GPU scheduling, or final performance. A declaration hash is
  **not** a host attestation, tensor hash, runtime verification, or measured
  training run.

## Guards needed before any equal-time comparison

1. Exclusive GPU/OS lease with independent owner, sustained awake/power state,
   measured memory fit and historical Windows power events checked.
2. Immutable same-base/admitted-corpus/seed/initial-adapter bindings and
   source-level token-weighted objectives; forbid checkpoint cherry-picking
   and cross-process resume in the strict timed comparison.
3. For each method, execute frozen, zero-update, correct-label and
   corrupted-label controls against the same admitted training source; do not
   unseal the untouched independent evaluation bank until externally attested
   eight-arm freeze and physical adapter reload.
4. Record optimizer wall seconds, external process wall seconds, elapsed
   monotonic/wall disagreement, examples, useful and padded forward/backward
   tokens, supervised target tokens, actual memory scope and independent
   adapter hashes. Distinguish *equal time* from *equal token exposure* and
   *behavioral improvement*; the latter needs independent scoring.

Historical R18 MeZO: **531-second budget, 19,813.562 seconds elapsed** with
Modern Standby overlap. Classification remains
`TIME_BUDGET_OVERRUN_INVALID`, never repurposed as equal-time evidence. R22's
forecast does not rehabilitate it or authorize new training.

## Verification

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest -v tests.test_r22_equal_cost`

Synthetic tests show the unequal-target-count objective counterexample,
identical time but unequal token work, invalid/misbound corpora, private eval
leakage, duplicate method profiles, impossible MeZO/QLoRA physical-token
accounting, calibration overrun, invalid numeric losses, and CLI nonauthorization.
