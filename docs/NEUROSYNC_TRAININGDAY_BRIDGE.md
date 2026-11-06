# Neuro-Sync ↔ TRAININGDAY — source-only compatibility and study contracts

**Status:** experimental source code; no approved model training, no verified transfer,
no persistent StateBus authority, no host install. This is an optional research
integration, not a new positive stage in the TRAININGDAY release graph.

**Design-time Git sources (2026-11-06):**

- `thebrazenbeard/neuro-sync` main `89fdef1dd83536cf29875d3de7c2acb07e6f2cfa`
- `thebrazenbeard/model-training` main `2b83430dd36076c5f65624709968e372b170ea36`

These pins are provenance **declarations**, not proof of checkout identity,
rights admission, checkpoint bytes, hardware readiness, or executed inference.
Review against the actual current source and artifacts before using a plan.

## What this adds

1. `trainingday/neurosync_transfer_preflight.py` checks **declared** exact
   architecture, base revision and digest, tokenizer and chat-template hashes,
   adapter targets/rank/alpha/scaling, dataset/rights receipt identifiers,
   experimental controls, donor disconnection, receiver cold restart, and
   source revision formatting. A result is `SOURCE_COMPATIBLE_HOLD` or
   `INCOMPATIBLE_HOLD`. It never grants physical launch or state commit.
2. `trainingday/neurosync_experiment_design.py` preregisters seven independent
   comparisons: untouched, context-only, full donor-adapter copy, shared-base
   delta transfer, recipient-native distillation, trained coupling research,
   and shuffled-donor negative control. It requires planned independent
   blinded heldout scoring of self-identification, conversational continuity,
   memory humility, voice consistency, and general capability.
3. `trainingday/neurosync_manifest_join.py` refuses to mistake two
   independently passing source contracts for one coherent experiment.
   It checks overlapping source/base/tokenizer identities, canonical digests,
   and an explicitly declared dataset-hash mapping. Even a matching mapping
   claim remains `STRUCTURALLY_MATCHED_UNVERIFIED_HOLD`.
4. `trainingday/neurosync_statebus_gate.py` is a **read-only metadata
   receipt-recovery prototype only**, not a production authenticated broker
   or atomic transaction boundary. It cannot commit StateBus events or
   authorize a weight update. Its limitations remain below.

## Tests — entirely synthetic and offline

```sh
python -B -m unittest -v tests.test_neurosync_transfer_preflight
python -B -m unittest -v tests.test_neurosync_experiment_design
python -B -m unittest -v tests.test_neurosync_manifest_join
python -B -m unittest -v tests.test_trainingday_neurosync_statebus_gate
```

Run the repository's existing TRAININGDAY regression suites separately.
The tests inspect functions and synthetic data; they do not execute Qwen,
Granite, PEFT, adapters, CUDA workloads, conversion, or host state changes.
Passing unit tests must never be reported as model improvement or live
pipeline qualification.

## Eligibility and evaluation interpretation

- No model-produced rights receipt, source SHA string, policy boolean or
  independent-review claim is a substitute for host-authenticated evidence.
- Same-base adapter transfer proves neither cross-architecture transfer nor
  causal semantic improvement. Model agreement and changed hashes alone are
  insufficient.
- A valid seven-arm design is a **proposal**. An independent custodian must
  seal train/heldout separation, preserve blind assignments, and score
  genuinely cold, donor-disconnected receiver runs. Match or account for
  interventions' effective training exposure and evaluation budgets.
- A passing study and a passing transfer preflight do not identify the same
  experiment until their identities and dataset relationships are reconciled.
  Their split-byte and manifest hashes refer to different objects; an
  unverified mapping receipt cannot close the evidence gap.
- Neurological retention cannot be inferred from StateBus/context retrieval;
  externally supplied facts and retained weight changes are distinct outcomes.
- The receiver's claimed identity must be accurate about its runtime and
  uncertainty; mimicry, invented memories and rehearsed style are not
  evidence of a persistent independent personality.
- Training remains subject to TRAININGDAY's existing host qualification,
  owner-rights, GPU-lease, checkpoint/recovery, conversion, and protected
  evaluation gates. This addition wires **none** of those physical stages.

## Hostile findings still material

> **UNKNOWN:** Manifest metadata has not been matched against actual signed
> source, tokenizer, base and adapter bytes, and rights/heldout custody has
> not been independently verified. A matching SHA *field* proves nothing.

> **UNKNOWN:** A metadata gate cannot provide atomic permission revocation
> across external policy and a SQLite write. The raw Neuro-Sync ledger must
> remain host-private; a model must never instantiate the gate with its own
> policy callback or access the ledger directly.

> **UNKNOWN:** New Qwen training is not qualified on Lappy/VN1/VN2. Existing
> workloads must be left alone; no new GPU or physical run is requested here.

**Next qualification step:** independent, byte-bound artifact and data-rights
receipts with a genuinely isolated, randomized seven-arm experiment once a
host-controlled lease is available. Separate semantic improvements from
transport correctness, and retain negative as well as positive findings.
