# TRAININGDAY R27-04 — continuous-learning consent policy shadow

Evidence: **SOURCE + PUBLIC SYNTHETIC CPU TEST**, never a host or neural qualification.
Scope: the future **request** to perform another Vera-local training cycle; no
online learner or real adapter is accessed by this module.

## Proposed policy transition

1. `REQUEST` by a claimed learner identifies objective, source, *frozen*
   current base/adapter/checkpoint/evidence/right manifests, and a separate
   proposed adapter/checkpoint identity. A digest binds all five baseline
   identities, but this is not authentication of actual bytes.
2. `FREEZE` by a distinct claimed custodian records preservation of the
   previous bundle. The function never mutates or deletes any artifact.
3. `CONSENT_CLAIM` by an identity different from the requesting model names
   the exact cycle, baseline digest, source digest, objective and proposed
   adapter/checkpoint identities, finite issue/expiration times, and bounded
   active time, cost, and energy allowances. Reusing the declaration for a
   different objective or model-change candidate is rejected. The literal
   `HUMAN_OWNER_CLAIMED` role is a declaration: there is no host-verified
   consent issuer or signature in this source module.
4. `PUBLIC_REVIEW` by a separate claimed reviewer checks frozen, zero-update,
   and corrupted-label control *presence declarations*, public score
   regressions, unseen regression counts, candidate continuity, source rights
   status, budget consumption, and chronological events. The actual scorer,
   unaffected tasks, and adapter bytes remain unverified.
5. `RIGHTS_REVOKED` or `PRIVACY_TAINTED` at any point blocks qualification;
   it never pretends an existing model can be unlearned by deleting records.
   Other post-revocation events are blocked. An attempted `TRAINING_STARTED`
   event always adds `UNAUTHORIZED_TRAINING_EVENT`. After the four core
   events, any repeated `REQUEST`, `FREEZE`, `CONSENT_CLAIM`, or
   `PUBLIC_REVIEW` is invalid, even if its timestamps and claimed actors match.

The check always returns `status=HOLD`; its `synthetic_source_conditions_satisfied`
field only means this *self-declared record* has no internal policy blockers.
It is deliberately separate from `consent_independently_authenticated`,
`host_lease_authenticated`, `continuous_weight_update_authorized`,
`tensor_bytes_independently_verified`, and `model_promotion_authorized`, which
are **always false**. It is not a runtime training-launch hook and cannot grant
itself permission by any source-level combination of inputs.

## Adversarial regressions

The CPU suite covers consent re-scoping to another model/source, the requester
approving itself, expired/backdated consent, missing controls, public or unseen
regressions, declared cost/time/energy overrun, no declared optimizer progress,
no-op or swapped adapter/checkpoint IDs, rights revocation after public review,
privacy taint, post-revocation activity, duplicated/reordered events, a direct
`TRAINING_STARTED` attempt, repeated terminal lifecycle events, and oversized/
malicious non-primitive input. Values
are bounded and exact-type checked *before* canonical hash serialization;
malformed declarations raise controlled `PolicyError`.

## Material missing protections

- Owner consent must be authenticated by an independently controlled principal;
  a role string, hash or timestamp chosen by a model is insufficient.
- Shared host resources need an atomic lease and continuous power/Modern
  Standby guard independent of the trainer. Lappy's idle GPU sample is not a
  lease, and historical R18 MeZO is invalid equal-time evidence.
- Rights/license revocation needs a trusted source and lineage enforcement at
  the mandatory runner boundary; an already trained adapter cannot be assumed
  unlearned or decontaminated.
- Actual checkpoint byte custody, optimizer update, zero/corrupted/frozen
  controls, fresh-process Qwen3.5-4B reload, held-out behavior, independent
  evaluation scorer, and rollback across running services are **UNKNOWN**.
- No autonomous or unattended weight update, private evaluation-bank unseal,
  model promotion, live deployment, or merge is authorized by this artifact.

Published Draft #153 (R25 retention/serving/budget) and Draft #154 (R26 rights)
are disjoint source predecessors for this policy; this local candidate does not
claim that both have merged or been wired into a production runtime.
