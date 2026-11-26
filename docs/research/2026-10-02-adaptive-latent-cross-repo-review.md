# Adaptive Latent Memory — Cross-Repo Exact-Head Review V1

This document binds the current implementation subjects for the 2026-10-02 adaptive latent-memory work. It is not a merge, deployment, provider-internal KV claim, or production qualification.

## Exact subjects

- vera-mono: `f58686141f1496cb7901ad10b4be6ee474daf697`
- rezon: `ab61e4d2b7677e444e7ccbbedfd0d8ecdbc04d8d`
- mosaic: `023dee325dca9f4d487ea6708b7c425c2055aa16`
- spm: `1d67f7f4a17f58c407494df24130ebbdae573546`
- lgcm: `28d17a8668bbee3792f32daf4fc3a4fc6c6a5cc0`
- vera_model_training: this branch contains the companion machine-readable manifest and late-relevance proxy package.

## Composition review

> **Hostile challenge: positional routing can silently become stale evidence.**  
> SPM's learned router emits positional routes such as `chunk:1`, while Vera/Rezon reason in durable evidence identities. If candidate order changes between inference and promotion, every component could pass its own unit tests while the composed system promotes the wrong evidence.

**Resolution:** vera-mono `f586861` adds an ordered `ResolutionRouteBinding` whose digest binds the exact block-id sequence. `chunk:N` is converted to a durable block id only through that binding; reordered candidates change the binding digest, out-of-range routes fail closed, and forged bindings are rejected.

> **Hostile challenge: exact recovery becomes ambiguous if one compact block can point at multiple exact sources.**

**Resolution:** current Vera V1 is stricter than the review initially assumed: every latent block already requires exactly one `source_ref`. New regression tests preserve that invariant.

> **Hostile challenge: a revision-pinned Kaggle model can still be weaker evidence than SPM's frozen baseline subject if the actual downloaded file inventory is never verified.**

**Resolution:** SPM `1d67f7f` binds the Kaggle campaign to Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775` *and* inventory digest `6080fc05cb5e0ccfa35e64523b11a902cc1f3e35672f85135a19eb16b722f8b8`. Real execution hashes the downloaded snapshot and refuses training on mismatch.

## Current ceiling

The deterministic runtime path and learned routing mechanism now compose more safely, but the central hardware claim remains open. The work has **not** yet established production GPU-VRAM reduction, provider-internal KV-cache compression, or learned exact-fact generation from lossy latents.

The next meaningful evidence is the pinned Kaggle run: route accuracy across 2/4/8/16 latent slots, unsafe DIRECT rate, safe insufficiency rate, latency, and measured CUDA peak memory versus a full-context control.
