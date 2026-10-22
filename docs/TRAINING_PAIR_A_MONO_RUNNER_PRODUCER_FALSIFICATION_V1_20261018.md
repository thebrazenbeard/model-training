# Training Pair A — Mono-to-Runner producer boundary V1

2026-10-18. Training Lappy / Integration VN2, PORTAL coordinator. SOURCE RESEARCH ONLY.

## Bound immutable sources

Training parent #117 a5f52c6fddb7c6daea8c29e489e060e23948d6b7.
Mono #77 8fd64c97998a8d24943439a96409f2114646905e, QualifiedVeraRuntime source blob 48e00014b89adeb392102ef45849fbd49fc9dc29, TaskExecutionLedger source blob b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de.
VeraOS #38 637ff4f4f9f7737d5075a3cc8fa9c74ee4201cb8.
PORTAL #62 6ef585f36194a538a01b751d0802a9e876815f0a, WorkUnit source blob 7c76c2b1f16b5e8aaf9522cc5d97adde53b0e721, direct Runner admit blob 5ca923567f1d4af6bfd997d7584d7c6255c8eed2, test-only proposal blob 0742d2e0dbfbfc09d49d9d7aafd38cba66e59ad2.

Integration current authoritative source return in private Bus #368 comment 6252688998: selected authenticated Mono task-to-Runner producer and receiver-only nonce issuer are ABSENT / UNVERIFIED. A string claiming issuer or model process is not evidence. PORTAL #61 direct claim guard rejects only literal UNAPPROVED_MONO_PRODUCER_ASSOCIATION_V1 marker, not arbitrary unmarked WorkUnit payload.

## Owned source tests and their evidence level

Training branch training/l2-mono-runner-untrusted-producer-20261018 stacks on #117 and contains eleven authored tests across three files.

- Six native tests: forged caller identity/model/source in an UNMARKED Runner WorkUnit; changed A task head; unrelated B global head; explicit UNAPPROVED marker negative with exact store nonmutation; duplicate nominal nonce across independent Runner SQLite roots; changed caller issuer/session/generation/route with a recomputed WorkUnit fingerprint.
- Two actual child-process crash fixture tests: prior to local Runner claim there is no attempt; after the local claim the persisted attempt must remain unresolved and not redispatched. Neither means a real model/provider result.
- Three further tests: forged Mono task/source/head/nonce fields as mere caller payload; removal of the literal UNAPPROVED tag and recomputation of public fingerprint, preserving the normal local Runner path; actual cold Mono child writing checkpoint A between Runner root initialization and the local Runner admit. These are expected counterexamples to a qualifying cross-store producer, not a demonstrated protected effect.

New native tests have NOT been executed: Executor list_devices and Lappy process tools return internal failures. The source roots are explicitly pinned by TRAINING_MONO_ROOT and TRAINING_PORTAL_ROOT. Tests SKIP if roots are absent/mismatched; a skipped suite is not PASS. The machine status in docs/evidence/TRAINING_MONO_RUNNER_PRODUCER_BOUNDARY_20261018.json is NOT_EXECUTED. Historical Training #117 31 PASS concerns earlier task-CAS evidence and does not validate these source tests. PORTAL #61/#62 native regressions and VN1 host-independent review are also outstanding.

## Typed authority contract, not an installed implementation

CanonicalTaskSourceV1: Mono owns original text/packet/source/scope, active delegation and latest TASK event digest under actual compare-and-swap. Immutable TASK_OPENED head differs from current TASK event head; both differ from GLOBAL journal head. Unrelated checkpoint B must not stale A, but actual checkpoint A does.

DelegatedWorkSubjectV1: Mono validates repository/ref/assignee, binding event and allowed effects. It does NOT establish physical issuer custody.

ReceiverObservedIssuerV1: absent/unverified. A future authorized receiver must independently attest issuer/revocation, host generation, process/session/PID/start, process-loaded model bytes/config, one selected route and response byte evidence. Never read trust from untrusted caller dict or model-generated text.

ReceiverConsumeOnceV1: absent/unverified. An authorized existing Mono/host owner must atomically bind source/task/current head/issuer/route to nonce consumption across all participating tasks/processes/roots. General TASK_CHECKPOINT is not trusted receiver-claim history; independent SQLite roots do not coordinate simply because both have a lock. After an ambiguous durable claim but unknown result, HOLD_UNKNOWN_NO_REDISPATCH. No new Training trust database.

RunnerLocalAdmissionV1: PORTAL locally verifies immutable WorkUnit/budget/lease/collision/attempt. The exact literal UNAPPROVED proposal remains a negative guard; unmarked local claim is legitimate local control, not proof of authenticated canonical Mono producer. No blanket rejection of local unmarked WorkUnits is justified.

ResultAndMemory: a real selected Pre-Active DONE plus COMPLETED run and response provenance is required when that path executes, followed by Mono canonical CAS/readback. Canonical memory remains separately consent/source/privacy/correction governed; generated assertions do not self-admit.

> HOSTILE REVIEWER: A successful unmarked Runner local claim is expected, not itself a security breach or external provider execution.
> ACCEPTED. The falsifier only demonstrates that the local claim cannot be promoted into a trusted cross-store producer without independent issuer and atomic task authority.

## Cognitive and guest boundary

Accept source-level Mono no-provider TASK_OPENED/HELD/cold recovery. Preserve current VeraOS #38 selected 431-source composition: Mono316, PORTAL advisory89, Pre-Active18, Volition8. Optional Cricket, Rezon, semantic, identity, learning, memory, Mosaic/model, Ears/Vision candidates require measured task utility and separate admission. Rezon automatic source-origin acceptance stays rejected and #106 missing predecessor remains HOLD. Integration OS38 latest symlink fix and eight Linux source-only tests are Integration owner evidence, not a built/booted graphical VeraOS guest. No Qwen/GLM loaded-model source, GPU or training was run.

Partner route: Integration must identify a selected real Mono→Runner association callsite or explicitly continue ABSENT/HOLD; PORTAL must preserve local collision fences and unknown no-effect holds, not promote an untagged WorkUnit into authenticated producer. VN1 owns independent physical reviewer verdict. No provider/trust/credential/memory/protected effect, deployment, scheduling, merge or paid compute.
