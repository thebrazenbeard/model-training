# TrainingDay R25 — Public multigeneration retention lineage (SOURCE ONLY)

This source-only, fail-closed evaluator extends the single-generation R20
public retention canary **declaration** comparison. It models adapter
generation **A → B → C**, while checking both adjacent transitions and A → C.
It neither imports nor opens a private evaluation bank.

## Inputs

- A frozen public-only bank manifest with ordered case IDs, source/base/initial
  adapter SHA-256 commitments, and an explicit ban on training with answers.
- Exactly three generations, each with distinct adapter SHA-256, parent adapter,
  frozen-base and public-case order bindings, score/training receipt hash
  *declarations*, a per-case public rubric-pass vector plus exact canonical
  `score_vector_sha256` binding to the ordered case commitment and budget/leak
  flags. A stale vector digest is rejected; a forged vector+fresh forged digest
  is **not** independently authenticated.
- Strict zero-case-regression policy and a computed 95% Wilson lower bound
  used as a **descriptive** sample-size warning. The score sample may be
  contaminated or correlated; Wilson is not a generalization certificate.

## Behavior and limits

Malformed, misordered, duplicated, unpinned, overbudget, private-bank accessed,
or training-contaminated declarations raise `AdmissionError`. Valid-shaped
declarations yield public aggregate pass counts and regression/recovery counts
for A→B, B→C and A→C with explicit hold reasons. No raw answers are emitted.

All valid results return `status=HOLD`, `promotion_allowed=false`,
`neural_improvement_proven=false`, `actual_tensor_change_verified=false`, and
`independent_model_scoring_verified=false` — even when all three synthetic
generations have 100% declared pass counts and no regressions.

> **Hostile qualification:** A self-declared score receipt hash is not proof
> that an independently scored model ever produced those answers. A syntactically
> consistent A→B→C digest chain can be fabricated without loading any adapter.
> Cross-process fresh-adapter hash verification, independently observed scorer
> identity, untouched benchmark custody, zero/corrupted controls and statistical
> uncertainty are separate, still-open physical qualification gates.

This code is **not wired into the live training/serving runner**. It does not
replace the R20 per-generation four-arm comparison or permit graduation.

CPU-only tests use synthetic case IDs and fabricated outcomes, and must never
be interpreted as a real Qwen3.5-4B retention experiment.
