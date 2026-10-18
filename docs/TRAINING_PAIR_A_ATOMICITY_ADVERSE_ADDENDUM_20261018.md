# Training–Integration atomicity adversarial addendum

2026-10-18, Pair A Training Lappy / Integration VN2. SOURCE ONLY.

## Partner evidence actually received

- Integration directly acknowledged the task-head ABI in private Bus #368 comment 6250973911: the opener is first TASK_OPENED digest; expected_task_head means latest event of this task, not GLOBAL journal; task source and own-task advancement remain required. Integration is reviewing current owner task/guest ABI separately.
- Integration reported a NEW actual two-thread SQLite source RED in comment 6251047445 against exact Training #113 51bd9749c2bf77b9b2b3f9d94f55a56e3ef29896, preflight blob 0196b0877aeb9adfaf078d705e636fd31d2c414a, Mono #77 a351d36274595cde26bc60df4c44c3b83a6d1f0a. Once the preflight reads A's one-event snapshot, a second thread commits an A CHECKPOINT; preflight resumes and wrongly accepts stale A, though actual task has advanced. Integration's independently authored script SHA256 is f85303e82db2dad593cedeaa2f736c11d8db2ebf95da8de97c860cc2f7a035f5 (case insensitive as published).
- This is INTEGRATION-OWNED source adversarial evidence, not a real provider result or a VN1 third-party certification. No Integration production source was edited for Training.

## Additional Training-owned executable reproduction

Frozen source test tests/test_preflight_atomicity_interleaving_v1.py records three cases on isolated real Mono SQLite state:
1. Static eligible synthetic preflight returns positive; the canonical task remains OPEN.
2. A checkpoint committed before preflight begins correctly rejects the old receipt.
3. Deterministic real writer thread: reader snapshots A's actual one-event history and blocks; writer commits TASK_CHECKPOINT(A), signals completion; reader resumes with its stale tuple. Old A synthetic preflight still returns ACCEPTED while actual A now has TWO events and remains OPEN. Nonce has been added to in-process set but no model, issuer, task closeout or effect occurred.

These tests intentionally assert the unsafe behavior, to ensure no future research claims preflight atomicity based only on individual task-head reads.

Final Lappy Windows CPython 3.14.5 selection: tests/test_checkpoint_poison_v1.py, tests/test_receiver_nonce_lock_v1.py, tests/test_preflight_atomicity_interleaving_v1.py, tests/test_receiver_preflight_task_head_scope_v1.py, tests/test_receiver_bound_preflight_v1.py — **27 PASS / 0 FAIL / 0 SKIP**, one pytest-asyncio warning, using the exact immutable Mono TaskExecutionLedger Git blob b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de. The previous 12 PASS subset is still historically accurate, superseded for the expanded selection.

> HOSTILE REVIEWER: A final second read would make this race rarer but would still leave a commit gap after that read. A source-only local mutex is insufficient if other task writers do not obey it.
>
> ACCEPTED. The future provider claim must use an existing qualified authority's single atomic task/source/head + receiver-issuer/host/process + nonce collision-domain transaction and readback. Without a selected receiver authority, all current source-positive fixture verdicts are NOT provider admission.

The original #113 positive repair remains required: unrelated task B may advance GLOBAL chain without invalidating unchanged task A. The concurrent A modification is different and MUST invalidate actual A before any authorized provider output or memory effect.

Production receiver authentication, durable cross-root nonce, single-route/process loaded model proof and real Pre-Active result remain HOLD. No TASK_CHECKPOINT promotion, trust DB, model run, memory write, provider/credential change, scheduling, guest deployment or main merge.
