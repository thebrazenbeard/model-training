# TrainingDay R26 — declared rights revocation and lineage taint

Status: **SOURCE/CPU-ONLY PROPOSAL / HOLD**. This is an independent branch,
not a mandatory training, evaluation, graduation, or deployment gate.

`trainingday.r26_rights_taint.assess_revocation(nodes, events)` accepts only
synthetic/public metadata. It never opens model weights, corpus records,
licenses, private assessment banks, hosts, or deployed services.

## Required graph declaration

The artifact graph is a bounded directed acyclic graph of SHA256-labeled
`TRAIN_SHARD`, `ADAPTER`, `CHECKPOINT`, `EVAL_RECEIPT`, and
`PROMOTION_CANDIDATE` nodes. Each derived node names its actual dependencies,
and a promotion candidate must depend on an evaluation receipt and an
adapter/checkpoint. The candidate's exact adapter/checkpoint IDs must equal
the directly evaluated IDs in its receipt(s): an ancestor adapter is **not**
interchangeable with a newer checkpoint even when it has valid lineage.
Every path must reach at least one declared train shard.
Training shards declare an immutable byte digest, a license-evidence digest,
and a rights status; all rights and byte digests are **UNAUTHENTICATED**.

Events have unique bounded IDs; `REVOKED`, `PRIVACY_TAINT`, and
`LICENSE_CHANGED` are the only valid actions. An event must name an existing
training shard and repeat its declared content digest; license changes must
bind the previous license-evidence digest and name a distinct replacement.

For each event, a taint status propagates to all descendants. Ranked status
is revoked > privacy taint > license changed > baseline rights review >
declaration-only clean. Every derived `ADAPTER`, `CHECKPOINT`,
`EVAL_RECEIPT`, and `PROMOTION_CANDIDATE` thus inherits the strongest concern
from its transitive sources; an unaffected sibling shard is not conflated
with the tainted source. Result includes sorted impacted artifact IDs and
per-artifact source causes plus a deterministic canonical claim digest.

The return value **always** states:

- `release_gate=HOLD`, `promotion_allowed=false`, `training_allowed=false`.
- `rights_attested=false`, `independent_custodian_verified=false`.
- `protected_eval_unseal_allowed=false`, `unlearning_verified=false`.
- `evidence_retention_action=NO_DELETION_ATTEMPTED` (not evidence persisted).

## Negative test coverage

Public CPU fixtures cover revocation→adapter→checkpoint→score→promotion
taint, license change review, privacy contamination, unknown rights,
priority when several events affect one shard, deterministic reorderings,
tampered source digest/old license, forged event authority fields,
ancestor-adapter versus directly evaluated checkpoint substitution,
duplicate IDs/digests, dangling parents, illegal dependency types, cycles,
orphan derived artifacts, invalid/unhashable types, pre-serialization count
caps, and post-call mutation of inputs.

No evidence is destroyed. The function does **not** erase learned information
from already-updated model weights. Revoking a shard requires future action
by an authorized custodian; actual machine unlearning or retraining would
need separate measured tensor/provenance/behavioral proof.

## Remaining authority boundary

The caller can fabricate a fully internally consistent graph and recompute
all SHA256 declarations. This source check does not authenticate an original
rights holder, deletion request, consent grant, full training-data lineage,
adapter tensor bytes, independently scored evaluation, sealed-bank custody,
or any real process. It does not import, replace, or bypass the R18 admission
or protected-evaluator gates. A future host-enforced, signed custodian and
mandatory pretraining/evaluation/release wiring would be needed before this
could serve as an operational blocking control.

R18 MeZO historical overrun remains `TIME_BUDGET_OVERRUN_INVALID`; this
module gives no evidence of Qwen3.5-4B improvements or equal-time optimizer
results. `SOURCE_REVIEW=PASS` would not grant merge, training, or promotion.
