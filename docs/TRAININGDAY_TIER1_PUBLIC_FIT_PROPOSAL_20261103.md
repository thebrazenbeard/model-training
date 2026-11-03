# TRAININGDAY: proposed unpromoted public-base hardware fit probe

**Status: DESIGN CANDIDATE ONLY. NO GPU LAUNCH AUTHORIZATION.** The live
Windows HostBroker and `physical_launch_authority.py` must remain unconditional
`HOLD` until an explicit owner decision, independent review, protected
installation, and host readback. This document does not relax them.

## Why distinguish two tiers

The current shared physical-launch gate blocks even public synthetic
hardware-fit experiments. Protected R201 training does require independent
corpus rights, evaluator custody and NAS Q6; a read-only load/fit check against
an independently licensed official *public* base and locally generated public
synthetic examples need not consume protected R201 data or change its state.
That separation can reduce unnecessary coupling **only if Patrick explicitly
approves a separately named, weaker GPU coexistence policy**. A user-created
file saying `approved=true` must never be a grant.

| Contract | Proposed bounded activity | Explicitly excluded |
| --- | --- | --- |
| Tier 0 (existing source-only) | CPU unit tests, package/hash checks, static memory planning; no GPU process | Physical GPU work |
| Tier 1 (not yet authorized) | On the owner's selected host, one time-limited official pretrained public-base load plus public-synthetic forward/backward/one-step **fit probe**, with unpromoted output and documented VRAM peak/OOM | Protected R201/DEV examples or evaluator, Q6 commit, trainable adapter promotion, private data mounts, unattended repeated optimizer jobs |
| Tier 2 (still protected HOLD) | Genuine authorized Vera training and checkpoint custody when *all* original independent rights/evaluation/NAS Q6/process/lease gates pass | Claims that a Tier 1 synthetic step establishes generalization or Tier 2 eligibility |

Tier 1 would require these exact separately reviewed predicates before a
future authorized broker exposes *any* such capability:

1. Owner-signed, scope-limited authorization naming host, GPU policy, immutable
   experiment digest, max VRAM/time, allowed public-base artifact hashes,
   requested executable and **Tier 1 only**. The Windows service independently
   validates the grant; no caller-supplied `true` boolean counts.
2. Independent confirmation that base weights/tokenizer are the **official
   public pretrained** target with appropriate use rights; reject a random,
   shape-compatible substitute, an altered tokenizer or opaque base download.
3. Broker-controlled immutable plan and retained kernel PID+creation-time
   process HANDLE in a kill-on-close JobObject; exclusive lease against *other
   authorized Vera workers*. The broker additionally owns a durable **host-local
   Tier 1** monotonic replay epoch/journal, rejecting stale/replayed requests
   and failing closed after ambiguous crash state. This is **not** the independent
   NAS Q6 monotonic witness or a mutation of the protected Tier 2 epoch. An OS
   file lock is **not** exclusivity against Blender, Edge, desktop and other
   nonparticipating WDDM clients.
4. Preflight real device memory/host RAM and foreign process telemetry, then
   bounded launch, sustained telemetry, controlled CUDA OOM/timeout cleanup,
   retained artifact hashes and explicit unpromoted status. Report unknown
   per-process WDDM VRAM and foreign future allocations as residual risk.
5. No protected data mounts, no evaluator keys, no Q6 writer or secret-bearing
   environment. Tier 1 cannot alter or advance Tier 2 epoch, signer, state,
   nor be promoted by a result flag or manually copied log.

## Negative acceptance matrix (must run on the independently installed host)

| Attack/failure case | Required result |
| --- | --- |
| Current strict physical-exclusivity policy, no Patrick tier decision | **HOLD** even if GPU utilization reports 0%. |
| Missing/forged/expired/wrong-host/wrong-plan/wrong-tier owner grant | **HOLD**; never turn a client receipt into host authority. |
| Concurrent or replayed Vera request, stale **host-local Tier 1** journal epoch, broker restart | At most one owned Vera Tier 1 process; fail closed on ambiguous state; do not write the **Tier 2 NAS Q6** epoch. |
| Foreign WDDM client / VRAM unavailable or telemetry insufficient | Explicit fail/unknown under chosen policy, never claim physical exclusivity. |
| Official public model/tokenizer hash mismatch, R201 mount or evaluator handle requested | Refuse launch; no protected data access. |
| GPU OOM, timeout, worker crash, broker crash, process PID reuse | No orphaned owned worker; durable terminal status; no Tier 2 mutation. |
| A synthetically successful one-step kernel | Label `TIER1_FIT_OBSERVATION` only; `R201_TRAINED=false`, `PROMOTABLE=false`, `TIER2_AUTHORIZED=false`. |

> **Hostile challenge:** Separating public data from protected R201 does **not**
> remove Windows WDDM foreign-process contention, prove the official Qwen3.5
> model can fit 4 GiB, or create a trustworthy host-owned lease. Nor does
> successful public synthetic one-step optimization show useful Vera learning.

**Required decision before implementation:** Patrick must choose whether to
retain strict hardware-exclusive Contract A (Tier 1 remains HOLD) or explicitly
permit limited nonexclusive WDDM coexistence under Contract B solely for Tier 1,
with named host, immutable plan and risk bounds. Even if B is selected, this
source document alone grants nothing; the protected broker installation and
negative-test qualification remain separate effects.
