# Training Pair A — real native Mono/Runner producer boundary execution V2

2026-10-18 · Training Lappy (L2 with Integration VN2; PORTAL coordinator) · SOURCE AND DISPOSABLE SQLITE ONLY.

## Exact source pins and execution

- Training parent #118 `b0f5b8f2c8e4ca050b409be591b7b76843a52c9d` (Draft unmerged; prior machine evidence explicitly UNKNOWN_NOT_EXECUTED).
- Mono #77 `8fd64c97998a8d24943439a96409f2114646905e`, task-ledger Git blob `b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de`.
- PORTAL #62 `6ef585f36194a538a01b751d0802a9e876815f0a`, WorkUnit `7c76c2b1f16b5e8aaf9522cc5d97adde53b0e721`, Runner local admit `5ca923567f1d4af6bfd997d7584d7c6255c8eed2`.
- VeraOS #38 `637ff4f4f9f7737d5075a3cc8fa9c74ee4201cb8`, unchanged in this work.
- Host Lappy Windows CPython3.14.5, physically available Executor, actual checked-out Mono/PORTAL immutable detached heads and Training parent exact head. No access to a verified selected receiver/provider credential, issuer or model.
- Native command reproducibly executed by `evaluate_mono_runner_native_source_v1.py`: `py -3.14 -B -m pytest -q -p no:cacheprovider` four named test files, `TRAINING_MONO_ROOT` and `TRAINING_PORTAL_ROOT` set to the exact detached source checkouts; package paths on PYTHONPATH.
- Actual final **13 PASS / 0 FAIL / 0 SKIP / 1 pytest warning**, exit0, recorded in `docs/evidence/TRAINING_MONO_RUNNER_PRODUCER_NATIVE_20261018.json` with raw log SHA256 and `.log` file; not a synthetic claimed test status.
- First actual run of the eleven previously committed #118 tests: 11 PASS in 7.50 seconds. Two new Training-owned tests plus those eleven: 13 PASS in 8.45 seconds. Final frozen exact-source reproduction is separately recorded.
- Measured Lappy RAM free 11,140,596 KiB and RTX3050 free 3962 MiB of 4096 MiB at approximately 18:07 EDT, volatile and not reserved; no GPU/model load.

## New negative: Runner local admission precedes or survives later real Mono task changes

`tests/test_mono_runner_post_claim_task_drift_v1.py` adds two actual source/cold-process SQLite cases:
1. Open canonical Mono task A, construct an unmarked WorkUnit with caller-supplied producer/source/issuer/model/nonce fields, locally admit it into Runner, then start an actual separate CPython child that CHECKPOINTs A after admission. Cold reopen the same Runner SQLite database: unresolved local admission remains and another local claim rejects. Mono A's task event digest differs from the WorkUnit's old delegated digest and Mono task remains OPEN. **UNKNOWN_NO_REDISPATCH**, not DONE.
2. Repeat the admitted untrusted nominal nonce/WorkUnit in another independent Runner SQLite root after cold checkpoint A. That second root also admits locally, despite A's changed task head and another root's unresolved local attempt. This demonstrates absent cross-root collision/producer binding, **not** that production effects or model inference occurred.

Existing native controls in #118 stayed green: exact literal `UNAPPROVED_MONO_PRODUCER_ASSOCIATION_V1` rejected without changing persisted Runner state; ordinary unmarked legitimate local claims accepted, even if their caller text claims forged source/head/issuer/model/route; task-local A checkpoint prior to local claim does not authenticate a stale WorkUnit; unrelated task B advances GLOBAL without changing A's own event; cold exit before local claim leaves no attempt, cold exit after claim leaves an unresolved durable attempt; two physical Runner roots can admit the same nominal nonce.

> **HOSTILE REVIEW:** Accepting an ordinary unmarked WorkUnit is intended Runner local behavior. Calling that an external issuer security breach conflates local dispatch identity with canonical Mono producer authority.
>
> **ACCEPTED:** This research disproves an implicit claim of authenticated cross-store source origin, *not* Runner's legitimate local WorkUnit leasing/collision controls. Do not blanket-ban unmarked local work or invent backend effects.

## Canonical ABI / producer and model admission

Integration's current authoritative owner return (private Bus #368 comment #6252688998) classified currently selected Mono→Runner authenticated producer and receiver-only issuer/nonce authority as **ABSENT**. This measured native source test does not alter that verdict. PORTAL's #61/#62 known literal unapproved marker guard is valid but does not authenticate an arbitrary recomputed, unmarked fingerprint; VN1 physically source-verified that narrow limit separately at PORTAL #62, not this Training Draft.

Current task-head ABI: `opened_head` immutable TASK_OPENED event digest; `expected_task_head` latest event digest scoped to that task; `TaskState.journal_head_digest` GLOBAL chain integrity. A real producer binding must verify Mono source literal/packet and latest A event under the SAME actual authority/claim transaction that selects route and consumes a genuinely receiver-owned nonce, not rely on WorkUnit payload/fingerprint, TaskDelegationRef by itself or ordinary TASK_CHECKPOINT.

Receiver-owned issuer and host-generation/session, real loaded-model process+artifact manifest/bytes and transport result, cross-root nonce owner and a selected atomic producer association are NOT implemented/observed. Therefore provider result, model qualification, protected effects and canonical memory promotion remain **HOLD**. After an unresolved local Runner claim and no independently accepted real result, preserve UNKNOWN_NO_REDISPATCH and never infer task DONE from lease/attempt persistence.

Current guest remains 431 selected stage files (Mono316/embedded PORTAL89/Pre-Active18/Volition8). #109 Mono-only no-provider TASK_OPENED/HELD cold source evidence is a possible smaller offline design, not an installed graphical Linux guest. The #111 96-repository mechanism matrix does not authorize whole-repo guest imports. Rezon automatic provenance admission remains REJECT; memory writes remain separately governed.

## Review requests and authority limits

Integration VN2: inspect exact current Mono#77 producer callsites/actual selected OS38 receipt binding and return a source/host-authenticated producer owner if any; otherwise continue ABSENT/HOLD. PORTAL: maintain collision and unresolved-attempt no-effect fences; future producer gate must be separately qualified and cannot be the known literal marker alone. VN1: independently challenge post-claim cold A checkpoint, cross-root identical nonce, stale delegation head, unmarked caller substitution, before/after crash and actual source/issuer authentication; distinguish owner tests from reviewer-authored new fixtures and record exact host/head/log hash. No reviewer consent or independent approval is implied by this report.

No protected effect, model/GPU load, paid compute, training, credential/provider/trust mutation, canonical memory, QEMU, deployment, scheduling, or main merge.
