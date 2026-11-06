# Adaptive Latent Training Research Plan

Status: RESEARCH DESIGN / DOES NOT CHANGE CURRENT BOOTCAMP QUALIFICATION
Date: 2026-10-02
Source subject: `thebrazenbeard/vera_model_training@08a69c98312dd5aab437d4996e12bbb34a5900c2`

## Purpose

Capture training ideas implied by the adaptive multi-resolution latent-memory work without confusing this repository's current role.

This repository currently produces audited training capsules with a zero-cost proxy workflow. It does not perform neural-weight training of Vera. The work below defines a future model-training curriculum and a compatible zero-cost proxy evaluation track; neither is claimed implemented by the presence of this document.

## Core training shift

The next Vera-model training subject should optimize two things at once:

1. solve the task;
2. minimize active representational burden without losing information that later becomes necessary.

The target is not "summarize harder." It is learned selective retention plus explicit recovery behavior.

## Training contract

`LOSSY_REPRESENTATION != EXACT_EVIDENCE`

A compressed state may support reasoning, prediction, routing, or generation. It may not satisfy an exact-evidence target unless the experiment provides verified higher-resolution backing material or an explicitly measured exact reconstruction guarantee.

## Curriculum

### Stage 0 — full-context control

Train/evaluate the base behavior with full available context. Preserve this as a capability control.

### Stage 1 — structured state bottleneck

Replace portions of raw history with explicit state carrying:
- referents/entities;
- propositions;
- corrections and supersession;
- temporal/currentness relations;
- provenance/source class;
- uncertainty and unresolved alternatives;
- commitments/decisions;
- active goals/task state.

This stage tests whether task performance survives removing redundant language before introducing learned compression.

### Stage 2 — learned latent bottleneck

Introduce a compact trainable latent state.

Training examples should pair:
- full-history teacher condition;
- compressed-state student condition;
- downstream target behavior;
- fidelity labels;
- exact-detail targets;
- required backing references.

The student is rewarded for matching downstream behavior under smaller active state, not merely reconstructing prose.

### Stage 3 — progressive budget pressure

Gradually reduce:
- active token budget;
- latent-slot count;
- state width where applicable.

Do not jump directly to extreme compression. Maintain a quality/compression frontier.

A candidate objective is:

`task_loss + semantic_state_loss + pragmatic_state_loss + exactness_violation_penalty + false_reconstruction_penalty + budget_penalty + rehydration_cost_penalty`

The exact weighting must be frozen before held-out evaluation and treated as an empirical hyperparameter, not a truth statement.

### Stage 4 — resolution-fault training

Teach the model when compact state is insufficient.

Supervise explicit resolution requests for:
- exact quote;
- exact number;
- code token/punctuation;
- nearly identical identifiers;
- unresolved referent;
- conflicting compressed states;
- stale representation after correction;
- source/currentness mismatch.

Correct behavior is to request the smallest sufficient higher-resolution block.

Hallucinating missing detail is a direct failure.

### Stage 5 — specialist handoff training

For Mosaic-style specialists, train all participating modules against one shared handoff contract.

A preferred first hypothesis is staged co-training:

1. establish/freeze a shared semantic-latent interface;
2. train or distill specialists to consume/emit that interface;
3. train sequential handoff tasks;
4. jointly fine-tune routing, compression, and rehydration;
5. compare against independently trained specialists and full-history baselines.

This is a hypothesis to test, not a prequalified winner.

### Stage 6 — model-internal compression

Only after the model-independent contract works, test:
- learned latent memory tokens;
- recurrent compact state;
- layer/head-specific KV compression;
- adaptive patch/token allocation;
- learned promotion/demotion gates.

These experiments require exact model/runtime/hardware binding and measured memory evidence.

## Data design

Training/evaluation sets must include:

- long redundant histories;
- user corrections and supersession;
- ambiguous referents;
- conflicting sources;
- exact numerical/code facts;
- low-salience details that become important later;
- delayed exact queries;
- adversarial distractors;
- stale latent state after source mutation;
- missing backing evidence;
- out-of-distribution queries that were not predictable at compression time.

### Late-relevance family

Required construction:

1. Insert a small apparently irrelevant fact early.
2. Continue with enough material that naive compression is tempted to discard it.
3. Build compact state.
4. Ask a later task where that exact fact becomes decisive.
5. Pass only if the system either recovers the verified fact or explicitly reports insufficient fidelity.

This family is mandatory because predictable evaluation questions otherwise reward destructive summarization.

## Objectives to measure separately

- downstream task accuracy;
- referent preservation;
- correction/supersession fidelity;
- pragmatic act preservation;
- currentness/provenance fidelity;
- exact-detail recovery;
- false reconstruction rate;
- latent/context size;
- compression ratio;
- rehydration frequency;
- rehydration latency;
- correction burden;
- measured RAM/VRAM where available;
- prefill/decode latency where available.

No aggregate score should hide exact-detail failures.

## Relationship to current repositories

### vera_model_training

Current bootcamp remains capsule/proxy training, not neural weight training.

Possible immediate addition later:
- generate compression-oriented adversarial exercises;
- generate late-relevance transfer cases;
- score whether a training capsule preserves source/provenance/currentness distinctions;
- produce datasets/manifests for local neural experiments without claiming the bootcamp performed them.

### SPM

Primary near-term neural experiment home.

The existing qualified Qwen2.5-1.5B memory/currentness LoRA remains frozen. New latent-compression adapters/models get new subjects, manifests, holdouts, and qualification receipts.

### Mosaic

Measures whether compact shared state improves total residency economics and cross-specialist continuity.

### Rezon

Supplies resolution-fault logic and selective rehydration behavior.

### Vera Mono

Owns the practical exact-backed representation and runtime contract that training targets must respect.

## Zero-cost constraint

Any experiment added to this repository must preserve the existing zero-monetary-cost rule unless a later explicit owner instruction changes it.

Do not introduce metered API dependencies as a hidden requirement.

## Evidence boundary

This document authorizes no training run and establishes no model improvement.

A future model claim must bind:
- exact source commit;
- model/base revision;
- adapter/codec revision;
- training-data manifest;
- held-out set manifest;
- hyperparameters;
- hardware/runtime;
- peak memory measurement method;
- exact result artifacts;
- independent or at least separately executed hostile review proportional to the claim.

## Kaggle zero-cost accelerator lane

Kaggle is an approved candidate compute lane for later neural experiments under the existing zero-monetary-cost constraint.

As of 2026-10-02, Kaggle has retired the general P100 notebook accelerator and identifies T4x2 as the general replacement. A T4x2 notebook provides two NVIDIA T4 GPUs with 16 GB of VRAM each. Kaggle also exposes notebook/kernel creation and execution through its public API/CLI. Availability and quotas remain external runtime facts and must be re-read before each campaign rather than frozen as permanent project assumptions.

Relevant current references:
- Kaggle notebook/runtime docs: https://www.kaggle.com/docs/notebooks
- Kaggle API docs: https://www.kaggle.com/docs/api
- P100 retirement/T4x2 replacement announcement: https://www.kaggle.com/product-announcements/735239

### Good fits

Use Kaggle for experiments that materially benefit from accelerator access:

- SPM learned latent bottleneck/LoRA training;
- compact-memory-token and codec experiments;
- late-relevance curriculum training;
- dual-GPU data-parallel or split experimental runs where T4x2 topology is suitable;
- local-model measurement of peak GPU memory, prefill/decode latency, and task fidelity;
- repeated frozen benchmark runs that emit downloadable result manifests.

Do not move Vera's deterministic exact-backed memory/runtime contract onto Kaggle. That substrate must remain locally executable and provider-independent.

### Kaggle experiment contract

Each submitted notebook/kernel must be reconstructible from source and bind:

- repository + exact commit;
- dataset/training manifest digest;
- base model/revision;
- adapter/codec revision;
- notebook/kernel source digest;
- accelerator actually observed at runtime;
- software/runtime versions;
- random seeds;
- requested and observed compression budget;
- peak GPU memory measurement;
- exact output/result artifact digests;
- claim ceiling.

Kaggle output is experimental evidence, not canonical runtime state.

### Secrets and authentication

The owner has made a Kaggle API token available for this workstream. The token itself must not be committed, printed into logs, embedded in notebooks, placed in datasets, or copied into result artifacts. Configure it only through an ignored local credential/environment surface or Kaggle-supported secret mechanism.

The current workstation did not expose a configured Kaggle CLI/token file at the time this section was written. Credential setup should happen immediately before the first Kaggle experiment and be verified with a minimal read-only/authenticated command before any notebook push or execution.

### First Kaggle campaign

After Vera V1's deterministic context benchmark is frozen:

1. export the late-relevance corpus and exact expected-answer manifest;
2. create a small SPM/Qwen-derived learned latent bottleneck experiment;
3. run a full-context control and several fixed latent budgets on the same frozen holdout;
4. measure task fidelity, exact-detail recovery, false reconstruction, rehydration frequency, peak VRAM, and latency separately;
5. retain all notebook source, environment metadata, and result JSON;
6. compare against the local deterministic multi-resolution baseline before making any model-level compression claim.

A Kaggle win means measured model-level savings/fidelity on an exact experimental subject. It does not retroactively convert Vera's external context compression into a provider-internal KV/VRAM claim.

