# Neuro-Sync ↔ TRAININGDAY: seven-arm weight-transfer preregistration

**Evidence class:** SOURCE_ONLY_PREREGISTRATION. This directory is an isolated
design artifact. It does **not** authorize a TRAININGDAY physical launch,
declare model progress, verify data rights, or grant a state/adapter commit.

Source pinned for design (2026-11-06):

- `thebrazenbeard/neuro-sync` main `89fdef1dd83536cf29875d3de7c2acb07e6f2cfa`
- `thebrazenbeard/model-training` main `2b83430dd36076c5f65624709968e372b170ea36`

`neurosync_experiment_design.validate_study_plan` accepts only seven canonical
arms, each paired with an exact intervention and exposure-guard code:

| Arm | Proposed change to receiver | Interpretation |
|---|---|---|
| untouched | none | Nontransfer baseline |
| context_only | temporary prompt/context; **no weight change** | Distinguish in-context borrowing from retained learning |
| full_copy | donor adapter copied onto same-base receiver | Positive mechanical transfer control; *not* architecture-agnostic |
| delta_transfer | Neuro-Sync exact shared-base donor delta | Tests whether delta-route differs from copy |
| distillation | recipient-native training on rights-approved donor output | Tests native skill transfer vs simple access |
| coupling | proposed trained cross-model bridge with a pinned implementation | Research arm; **blocked from execution** until actual bridge and physical authorization exist |
| shuffled | train on donor outputs with deliberately shuffled association | Negative control for spurious gains and answer leakage |

## Required independent run contract (not implemented here)

1. Lock canonical repo heads, source weights, tokenizer and approved training
   manifest. Prove data rights and obtain independent host-owned physical launch
   authorization; this Python code cannot do either. For a same-base pilot,
   each arm starts with the same immutable recipient base.
2. Predeclare all seven conditions, randomized arm execution order, equal
   **evaluation** budget, and intervention-token *caps* (unused intervention
   budgets are not evidence of a fair amount of actual compute). Preserve
   training exposure, token counts, optimizer, seed, runtime and resource usage
   per arm in separate sealed receipts. The necessary differences between
   context/copy/train arms must be reported, not disguised as matching compute.
   The untouched baseline must receive the same IO scaffolding and evaluation
   budget while receiving **no donor information or new context**. Purge
   temporary context before the cold evaluation; preseed the shuffled control.
   The validator checks those *declared guards*, not their actual observance.
3. Keep a heldout corpus sealed with an independent evaluator. This plan records
   only the manifest hash, 50 fictional item counts in five strata, and blinded
   arm-assignment commitment. Never hand examples, answers, rubrics or
   arm-to-label mapping to donor, receiver, retrieval channels, or this file.
4. Score self-identification, conversation continuity, memory humility,
   stylistic consistency, and general capability. Each task needs a predeclared
   observable rubric, repeated paraphrases, and independent human adjudication
   for subjective judgments. Avoid training an identity claim as a hidden
   answer: reward accurate self-identification, uncertainty and no invented
   personal memories, not confabulated autobiography.
5. Restart the **neural** receiver after each intervention, disconnect donor,
   disable external retrieval/context state, and run blinded heldout tasks. A
   fresh Python worker alone is insufficient proof of neural cold restart.
6. Record baseline and final checkpoint/adapter SHA-256; successful retention
   requires a materially changed weight artifact, a true cold restart, and
   improved independent heldout scores over appropriate controls. A label or
   source test cannot establish any of those observations.
7. Treat independent scorer/custodian declarations as **pending** until external
   verification receipts bind exact artifact SHA-256, actor, process/runtime,
   source revision, and host-executed work. Reject all self-certified verdicts.
   `require_independent_result_qualification` deliberately has **no local
   success path**: a claimed score or self-signed JSON receipt always HARD HOLDs.
   A future verifier must bind the plan digest, each arm's effective starting
   and ending weights, actual exposure/budget, blinded assignments, heldout
   custody, restart observation, and independent scoring to immutable evidence.

The plan preregisters eight causal contrasts, including treatment against
untouched, context-only, direct-copy, and deliberately shuffled control arms.
Consequently, nobody may report only the prettiest comparison while hiding
the more informative controls.

The validator rejects unknown answer-bearing fields, but cannot detect covert
answer or prompt content embedded inside an otherwise allowed text field.
It also rejects wrong/duplicate/missing arms, protocol-label
substitution, changed baseline exposure guards, dropped controls/contrasts,
claimed execution, reused train/heldout commitments, incomplete strata,
absent blinded scoring, and broken cold-receiver preregistration. The
fixture uses obvious dummy hashes, **not** real manifests or rights evidence.

> **Hostile falsifier:** `full_copy` and `delta_transfer` may work simply because
> donor and receiver share exact architecture. Distillation may outperform both
> with different data exposure or compute. A gain on personality-style questions
> may measure mimicry rather than retained identity, autobiographical accuracy,
> or general reasoning. Model agreement is not independent scoring.

> **Hostile falsifier:** A changed adapter hash can mean random perturbation or
> corrupted weights. An improved heldout score can mean contamination, weak
> baselines, or a grader told the arm identity. Neither observation alone proves
> donor-induced semantic learning.

**Non-goal:** Neuro-Sync's separate StateBus N/V/P/D/X/F/H controls measure
external-state and protocol utility. Do not replace the seven weight-transfer
arms with bus routing arms, and do not call a context retrieval success weight
plasticity.

**Cross-component join remains required:** The training preflight's separately
valid source/head, donor/receiver checkpoint, and train/evaluation hashes must
be compared against this plan's commitments under one exact run/plan identity.
Two source validators passing independently do **not** prove that they describe
the same experiment. Until the separate bridge join is independently validated,
the combined experiment is **UNBOUND / NOT RUN**.

Offline verification:

```sh
python -B -m unittest discover -s tests -p 'test_neurosync_experiment_design.py' -v
```
