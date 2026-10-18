# Training L2 — Action Lock Is Not an Implicit Canonical Task Fence

2026-10-18. Training Lappy, paired with Integration VN2; PORTAL coordinator. SOURCE TEST ONLY.

## Exact source binding

Training #114 parent 7ad7088993249c00f718896c371eaa68c9013eb2. Mono #77 a351d36274595cde26bc60df4c44c3b83a6d1f0a, task ledger blob b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de. Current VeraOS #38 b388061414b3231ff7cf0caea913683e8b451b0d, architecture #385 472ab3230e75b5e082ca47dec87308aad9d042ff. The old #113 false-HOLD task/global head repair, Integration independent A-preflight TOCTOU falsification, and #114 checkpoint nonce-marker poisoning RED are retained.

## New tested source fact

TaskExecutionLedger.action_lock() returns a PortableFileLock backed by a DIFFERENT SQLite file with suffix .action-lock.sqlite3. Ordinary canonical TaskExecutionLedger.checkpoint() appends through the task EVENTS database using its own BEGIN IMMEDIATE; it does not automatically acquire that separate action lock.

The evaluator opens A, holds action_lock() in the first process, then launches a new cold CPython process which calls an ordinary checkpoint(A, expected_task_head=A opener). While the action lock is STILL held, the holder reads TWO A events, including the newly committed TASK_CHECKPOINT. Thus action_lock by itself does NOT fence every independent canonical task writer. A subsequent stale A checkpoint correctly fails via existing task-head CAS. An unrelated task B advances global journal head without advancing A's own event head; global chain verification still passes. Every synthetic task is OPEN and no model, provider, memory or result was admitted.

Code: evaluate_action_lock_fence_v1.py.
Tests: tests/test_action_lock_fence_v1.py.
Machine evidence: docs/evidence/TRAINING_ACTION_LOCK_NOT_TASK_FENCE_20261018.json.
Frozen five-case SHA256 cbde3fb25b8849df26240d568b7fbe24a32c8c641b31307e810cac0a6e0124e6.

Lappy Windows CPython3.14.5; actual distinct Python process and disposable Mono SQLite. New tests + existing #113/#114 regression **32 PASS, 0 FAIL, 0 SKIP** with one pytest-asyncio warning. Measured complete pytest run 16.46s. A standalone evaluator observation required 3.20s; it is NOT an inference benchmark. No running model, production ledger, trust keys, protected memory, guest or external effects were touched.

> HOSTILE REVIEW: A selected receiver operation could voluntarily acquire action_lock and CAS under an actually authorized common transaction. This experiment only proves ordinary task writers need not acquire the advisory action lock.
>
> ACCEPTED. Do not generalize the result to every future atomic trusted receiver protocol. The exact candidate that assumes action_lock alone serializes arbitrary task writers must be REJECTED. Task checkpoint marker provenance, cross-root nonce scope, trusted issuer and model process still unresolved.

## Canonical provider interface requirements

- Mono immutable original task literal/source/scope, TASK_OPENED event digest and independently current latest TASK EVENT head are typed separately from GLOBAL journal chain head. The original #113 repair must not regress.
- Receiver issuer/host generation/provider session and actual loaded model/process bytes/config/PID/start/endpoint must be observed under qualified host authority, not self-described in a response or a public hash.
- A single owner must atomically bind authenticated receiver claim, task/source/latest event, one route and durable nonce uniqueness across tasks/processes and chosen host collision domain. Ordinary TASK_CHECKPOINT is NOT a receiver-only nonce claim; the separate file action lock is NOT an implicit task-events transaction. If no existing selected canonical/host claim owner can be identified, HOLD instead of introducing another Training trust database.
- Real selected Pre-Active DONE+COMPLETED/run/response evidence and Mono closeout CAS remain separately required. Postcommit ambiguous crash is UNKNOWN/HOLD/NO_REDISPATCH until verified result. Memory proposal under separate authority, privacy, consent, retention and scope; no generated content promotion.

## Cognitive/guest BOM and current ceiling

Minimum source-only offline task: Mono task/source/HELD/currentness/cold recovery. #109 Windows Mono-only source evidence does not prove a Linux guest. Current VeraOS selected source composition remains Mono316 + PORTAL embedded89 + Pre-Active18 + Volition8 =431. Do not remove components without Integration-owned target import closure. The complete #111 source matrix covers 96 repositories and eleven cognitive roles; 147 proposed consumer links are not guest hard imports. Cricket/Voss bounded critique, Volition proposals, Rezon advisory (automatic origin REJECT), semantic/learning donors, Mosaic model placement and Ears/Vision observation remain optional. Qwen and GLM24B distinct, unqualified inference candidates.

Integration #38 effective Buildroot Kconfig, PT_INTERP source checks and current rootfs symlink confinement source test do not prove an actual compiled target rootfs, native Python/ELF/SONAME load, normal guest PID1, Cog DRM pixels/input or two cold boots.

## Partner and reviewer

Integration VN2 must identify current authorized receiver-only claim event/host owner and challenge how all canonical task writers and selected provider transactions can establish a common atomic currentness/nonce domain; if absent explicitly HOLD. Report its response as Integration evidence, not Training self-certification. VN1 independently challenge exact test head against separate processes/hosts, altered task head, wrong issuer, crash and root scope; no independent VN1 verdict received for this new increment. PORTAL may review source evidence but cannot promote it into issuer, memory or deployment authority.

Decision: task-scoped CAS SOURCE ACCEPT; general action_lock as mandatory task writer/nonce fence REJECT; real provider, model admission, canonical memory, and graphical VeraOS HOLD. No scheduling, model run, training spend, credentials, protected effects, deployment or merge.
