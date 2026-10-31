# TRAININGDAY R28-03 — public-synthetic calibration and uncertainty shadow

**Evidence class:** local SOURCE + stdlib CPU tests, not real Qwen3.5-4B inference.
This module is intentionally unable to access a protected evaluation bank,
load weights, train an adapter, start an Executor job, or approve release.

## Inputs and custody

`preregister_calibration(pilot, pilot_receipt, plan)` binds a fixed public
split tag, run ID, R27 answer-free case-commitment digest, ten (or another
preregistered 2–20) equal-width confidence bins, Wilson/Hoeffding two-sided
nominal interval alpha, and calibrated nonregression tolerance. It produces a
canonical SHA-256 commitment **without scores or answers**. Scores are
supplied later as caller-declared `CalibrationSample` values for the four
opaque R27 control arms. The scorer requires exact case IDs and prompt hashes,
refuses missing/extra arms and outcome maps, refuses nonfinite confidence, and
rejects internally inconsistent refused/invalid-format samples marked correct.
Every sample identity, prompt digest, arm-map key, receipt commitment and
authorization flag is checked for an **exact primitive type before equality**.
This blocks objects with adversarial overloaded equality from masquerading as
committed hashes, case IDs, or a false-zero control.

The train/pilot overlap test is the R27 manifest's *declared* hash inventory;
neither rights nor independent data isolation is authenticated. All four
arm-role mappings remain caller-declared. The returned receipt always says
`oracle_authenticated=false`, `blinded_roles_authenticated=false`,
`physical_model_outputs_verified=false`,
`neural_improvement_claim_authorized=false`, and `release_authorized=false`.

## Metrics, bounds, and nonregression

- Accuracy and Wilson two-sided intervals per arm; Bernoulli IID assumptions
  are **unverified** in a declared-data shadow.
- Ten preregistered equal-width confidence bins by default. The rightmost
  bin includes 1.0. Empty bins are recorded with count zero.
- Expected calibration error (ECE), Brier score, and rate of wrong answers
  assigned confidence >= 0.90.
- Separate refusal and output-format failure rates; a correct answer cannot
  coexist with a declared refusal or invalid output format.
- A two-sided conservative Hoeffding interval for the paired candidate-minus-
  frozen correctness mean; nominal IID range-two bound, not an authenticated
  confidence interval for real model improvement.
- Inherits R27 paired exact sign tests against FROZEN, ZERO_UPDATE, and
  CORRUPTED_LABEL, Bonferroni adjustment, gain threshold, and family floors.
- **Strict per-case zero-update parity** with FROZEN: correctness, numeric
  confidence, refusal and output-format flags must be exactly equal in this
  public toy. This is committed as part of the calibration policy before
  outcomes are read. A backend-specific physical floating tolerance is not
  silently granted; it would require new preregistration and justification.
- Candidate ECE and Brier cannot regress beyond the preregistered tolerance;
  high-confidence errors, refusal, and format-failure rates have **zero**
  allowed increase above frozen. A model can gain synthetic accuracy and still
  fail the calibration gate.

The four possible statuses are `NO_CLAIM_ZERO_UPDATE_DIVERGENCE`,
`NO_CLAIM_CONTROL_FAILURE`,
`NO_CLAIM_CALIBRATION_REGRESSION`, and
`SYNTHETIC_CALIBRATION_THRESHOLD_MET_NO_CLAIM`. The latter is **not** model
qualification: it only means the *declared toy bits* met source-coded gates.

## Reproduction

From the isolated source root:

```bash
python -m unittest -q tests.test_r28_calibration tests.test_r27_pilot_preregistration
git diff --check BASE..HEAD
```

No private tasks or labels are embedded in source/tests. Future physical work
must obtain an exclusive host GPU lease, an independently held public/hidden
oracle, rights-verified disjoint train/dev/test custody, real Qwen adapter
hashes, fresh-process adapter reload, matched controls, and a fair valid
elapsed-time MeZO/QLoRA pair; historical R18 MeZO elapsed-time overrun is
invalid for that comparison. Protected OOD bank remains sealed.
