# TrainingDay R29 — PUBLIC TOY calibrated A→B→C retention

**Evidence:** SOURCE + deterministic CPU unit tests. **Not** model inference,
neural training, independent scoring, protected OOD evaluation or release authority.

R29 composes three previously separate declaration-only gates:

- R25: public-case order/score-vector commitments, adapter A→B→C lineage,
  and per-case retention across A→B, B→C **and A→C**.
- R27: preregistered multi-skill pilot inventory and frozen/zero/correct/corrupt
  matched arms, with declared correctness and no independently verified oracle.
- R28: per-case confidence, refusal, format and correctness, fixed-width
  calibration bins, Brier score, expected calibration error (ECE), and
  zero-update exact per-case parity.

`preregister_calibrated_retention` binds the R25 source/case order, R27 pilot
manifest, R28 calibration manifest, *opaque arm-to-role map*, and allowed
abstention case IDs into a canonical SHA-256 commitment before scoring. The
same frozen, zero-update and corrupted-label synthetic controls are required
across A/B/C. `score_synthetic_calibrated_retention` rejects malformed or
conflicting source, cohort, role, hash and outcome declarations and always
returns **HOLD**.

## Fail-closed decisions

Any previously correct public toy case changing to incorrect is counted as a
regression even when new successes improve aggregate accuracy. This includes
the A→C comparison, which detects forgotten original competence after a
temporary B recovery. Independent per-transition checks detect:

- A→B/B→C/A→C per-case correctness loss, new refusal, and format failures.
- ECE/Brier degradation beyond the **already preregistered R28 tolerance**;
  any increased overconfident-error/refusal/format-failure rate.
- Correct-label declared abstentions outside the predeclared allowlist.
- Frozen/zero/corrupted control drift across generations, or frozen vs
  zero-update confidence/correctness/refusal/format divergence within a run.
- A change in public source, case order, prompt digest, control roles, or
  R25 Boolean vectors without updated bound commitments.

R30 source-integrity follow-up: R29 freezes the exact builtin arm/role mapping,
generation records, and full nested per-case sample maps **before** invoking
any parent assessor. Each assessor receives a separate copy; neither a
post-calibration confidence mutation nor a post-preregistration frozen/zero
role swap can alter the source state used for R29 aggregation. Two deterministic
monkeypatch regressions were introduced after independent hostile review found
these TOCTOU aliasing defects in the original R29 commit.

Example negative: case 0 passes in generation A, then fails in C, while a
different previously failing case begins passing. Aggregate success can
increase and ECE can improve. R29 still records the explicit forgotten case
and the `DECLARED_CATASTROPHIC_FORGETTING_RISK` reason.

## Reproduction

```bash
python -m unittest tests.test_r25_multigeneration_retention \
  tests.test_r27_pilot_preregistration tests.test_r28_calibration \
  tests.test_r29_calibrated_retention -v
```

All fixtures contain only invented case IDs, hashes, scores and confidences.
No model is downloaded or loaded; no private evaluation corpus is opened.
The statuses are policy checks of **caller-supplied declarations**, not
independent evidence of real Qwen3.5-4B improvement. Neither these checks nor
their Git hashes verify labels, training time, gradient updates, adapter tensor
changes, fresh-process reload, permissions, or release eligibility. An
independent review of the **exact resulting commit head** remains necessary.
