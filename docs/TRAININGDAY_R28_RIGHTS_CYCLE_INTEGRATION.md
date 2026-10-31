# TRAININGDAY R28 — public rights-to-cycle integration, SOURCE ONLY

The R26 rights-taint source and R27 claimed-consent source previously ran
independently. That permitted a *synthetic* consent source assessment to report
clean while a separate rights graph reported a revoked candidate. Both modules
still prohibited actual training; the issue was missing source interoperation,
not a GPU-authorization bypass.

`trainingday.r28_rights_cycle_integration.assess_rights_bound_cycle()` now
takes bounded, independent plain-builtin snapshots of the caller's rights
graph/events and frozen R27 cycle request, baseline, consent, evidence, and event
dataclasses before either assessor runs. It then calls both assessors and binds one declared training-source SHA, proposed
adapter SHA, proposed checkpoint SHA, graph ancestry, and frozen no-event
rights-graph commitment. A later revoked/privacy/license-change event, an
unknown or review-required right, a disconnected second source, a mismatched artifact, a missing adapter
ancestor, or ambiguous multi-source lineage makes
`synthetic_source_conditions_satisfied` false with an explicit blocker.

The v1 graph binds the **current declared training source and candidate**,
not the frozen base model and every prior-generation adapter/checkpoint. The
R26 schema has no base-model node and the v1 one-source policy cannot express
distinct historical corpora faithfully. Therefore the receipt always states
`previous_baseline_lineage_verified=false`. Even a synthetic-clean receipt
cannot establish full learning-cycle checkpoint continuity. A separately
committed historical graph and a multi-generation provenance schema are
needed before that stronger assertion is possible.

Run the public CPU regressions:

```bash
python -m unittest -q tests.test_r28_rights_cycle_integration
python -m unittest discover -s tests -p 'test_r2*.py' -q
```

All inputs, including the frozen graph hash, rights flags, event actor claims,
consent and scores, can be supplied by one untrusted caller. They are not
signatures or physical file/hash/rights observations. Source consistency can
NEVER authorize a real optimization step, access to protected evaluation,
unlearning, promotion, quantization or deployment; every effect flag remains
false even when the synthetic source conditions are satisfied.

These private snapshots prevent a caller from changing inputs between rights
validation and artifact binding. This is **not** an atomic capture of a live
rights system during concurrent mutation, and is not independent custody.

An independently enforced rights custodian, separate evaluation principal,
exclusive host/GPU resource lease, same-object source reads, admitted corpus
rights, and fresh-process 4B adapter/behavioral evidence remain unqualified.
Existing R18 overrun is permanently invalid for fair equal-time comparison.
