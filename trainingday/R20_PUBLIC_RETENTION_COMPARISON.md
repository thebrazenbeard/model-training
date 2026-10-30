# TRAININGDAY R20 public retention comparison (source-only)

This utility complements the public, **nonsealed** five-case preflight on
Draft PR #142. It neither scores a model nor establishes that the frozen model
was competent on any case. The protected V10 evaluation bank is out of scope.

`compare_retention_receipts(preflight, receipts)` accepts the exact result of
`retention_canary.preregister()` and four rubric-judgment receipt dictionaries
for `frozen`, `zero`, `correct`, and `corrupt`. Each receipt must name:

- SHA-256 base model, trial, and common initial trainable binding; updated
  trainable digests on the correct/corrupt arms, unchanged on frozen/zero.
- Exact public source and case-ID digests, ordered five-row list of only
  `{id, rubric_pass}`; no prompts, responses or private labels in receipts.
- `PUBLIC_NONSEALED_RETENTION_ONLY` data partition and explicit declarations
  of zero private-evaluation access and zero canary training use.
- `DECLARED_RUBRIC_JUDGMENTS_NOT_ATTESTED` evidence method. Hashes and booleans
  do not independently prove GPU execution, model identity, honest assessment
  or actual adapter persistence.

The comparer reports case-level regressions/recoveries on frozen-passed cases,
not just the aggregate pass-count delta. It flags frozen/zero control
disagreement, and refuses mismatched cases, bases, trials, missing/duplicate
arms, sham tensor changes and contamination claims. Every output explicitly
sets `model_improvement_proven=false`, `promotion_allowed=false`; even a perfect
correct-label receipt cannot promote Vera or qualify neural learning.

Focused CPU tests use synthetic declared rubric outcomes, with no model calls.
Actual future scoring must freeze the human rubric, bind raw immutable model
responses, verify the exact adapter/base/controls, use an independent reviewer,
and keep the untouched final bank separate. This module is only a fail-closed
comparison and evidence-shaping layer, not that experiment.
