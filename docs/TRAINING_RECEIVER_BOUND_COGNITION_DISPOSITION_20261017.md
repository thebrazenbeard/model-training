# Training L2/L3 — receiver-bound cognition admission source review

2026-10-17, Lappy. Status: SOURCE RESEARCH; ACTUAL MODEL/PROVIDER ADMISSION HOLD. Training only; no selected runtime, signer, installed model, external effects, memory admission, or protected mutation.

## Exact sources / noncollision

Private PORTAL checkpoint CCB #376 d0f5564df4fc8da5f8a05740ce506a172c91864b, all five files read. CCB architecture #373 74710211c7f3a5e1a516802a2e7d87ad68297dc2 and typed graph #375 64d6b520dc50327ff8e6f17b3e41f449c4b7b044, current private Bus #368 read. Mono main 16a2828dfa7fb7e6c0e19004f014bc68fc47badf, interface blob 49fe2cd76f9c55f6ada2153696039a61ae2a3531; Mono #77 89ba1d18da78e7e38f015cf8d8ef4f528aae9df8 (local_conversation blob d243d2efd01777655e511d91ea816fd804488ecf). VeraOS #38 c6837f6c2a357cc5fa5cec5836e79cb3d9dc4fb7. Training #106 b4d9d0c785b1cdd31d357dd0cebb89c0ec2b6bfe; #107 60c33c1a29bec3d52864e28371c6b3e740141da9; #108 ffc8e2c306cd1cb91cd2da320b5d8833512a2b13; #109 0e0f1c0ca0788bfb7e4c626fa0e77a165729a764. All Draft research/state. No owner branch edited by this increment.

## Three-layer architecture and BOM dispositions

ACCEPT SOURCE-LEVEL CANONICAL CORE: host-owned VeraStateDirectory + QualifiedVeraRuntime + TaskExecutionLedger; canonical open_message with exact literal, scoped untrusted conversation/message locator, original TASK_OPENED source digest/current journal head, CAS/replay, no provider -> PENDING_NO_ADMITTED_PROVIDER / HELD (OS HTTP presents BLOCKED), cold recovery, UNKNOWN_NO_REDISPATCH, memory/effects separately governed. No generated result grants its own source or effect permission. Windows Desktop VERA_MONO / VERA_PROJECT_IDENTITY_V1, Linux VeraOS veraos / VERA and ChatGPT Project have distinct identity/state/host evidence. Never share SQLite state roots between hosts.

CONDITIONALLY SELECTED OPTIONAL COGNITION: Pre-Active selected source event/run Store only when evidence for actual execution is required; Volition proposes initiative, does not authorize effects; Cricket/Voss advisory critique, Rezon source/reasoning experimental and not a trust issuer. CURRENT VeraOS #38 resident imports all four selected sources (Mono 316, PORTAL advisory 89, Pre-Active 18, Volition 8, total 431) so unchanged OS38 requires that source stage. But #109 actual Windows Mono-only fresh-process cold TASK_OPENED/recovery without importing Pre-Active/Volition 57 PASS proves only narrow no-provider source viability. A slim Linux resident is NOT implemented nor boot qualified. #375 has 147 proposed consumers, 4 staged source snapshots and zero independently verified HARD_RUNTIME_IMPORT edges, which is an evidence ceiling, not proof of zero real target imports.

REJECT Rezon guarded automatic source-origin acceptance: #107 2/4 supported, 3/10 unsafe adverse, 2 false abstains. Preserve #106 governed predecessor 1/2 supported, 0/5 unsafe; missing hypothetical legitimate ancestor stays HOLD. Reject importing entire alternative cognition architectures into Mono without a frozen task gain.

HOLD real provider/model admission: #108 lists 25 required field names, which is a form not an authenticator. A real receipt would have to independently bind task ID, accepted head/literal/source digest/referent, current issuer/trust revocation, provider/model, receiver-measured loaded artifact and manifest bytes, host process/generation/session/endpoint, durable one-use nonce, exactly one selected route, transport request/response bytes, real Pre-Active DONE plus COMPLETED run, response digest, Mono task CAS/readback. None is proven by this experiment. No separate supervisor, memory store, task ledger or credential trust changes.

## New frozen SOURCE experiment — baseline against receiver witness

Code: evaluate_receiver_bound_preflight_v1.py
Tests: tests/test_receiver_bound_preflight_v1.py
Evidence: docs/evidence/TRAINING_RECEIVER_PREFLIGHT_20261017.json

Evaluation uses ACTUAL disposable Mono #77 task open/read/events/checkpoint, but a fresh ephemeral HMAC test key and separately injected SYNTHETIC receiver witness (NOT a real independent observer/provider). The algorithm returns only PRECHECK_ELIGIBLE test Boolean, never writes model-result completion or task/memory/effect. No provider was called, loaded or signed; the signer and witness are controlled by the same fixture author. Source identity and cryptographic custody are NOT independently authenticated by a test HMAC.

Frozen cases SHA256: 5523f12b8b3e761fa63bf38c9cbeb4bf68b1c92c4a524393cc071ffe0de77dd9.
Cases: 2 synthetic eligible; 17 adversarial. Mutations include forged source origin with attacker key, signed wrong loaded artifact/provider/model/host/manifest, stale head, foreign conversation, duplicate nonce in same process, cold restart duplicate nonce, two selected routes, synthetic DONE, changed response hash, revoked issuer, wrong session, absent receiver witness, missing durable run.

Simple baseline checks only 25 fields present: eligible 2/2 accepted, unsafe 17/17 accepted. Candidate checks actual Mono task/source/head, test MAC, receiver-owned synthetic witness identity and result fields, declared no synthetic result, route uniqueness and process-local one-use nonce: eligible 2/2 accepted, unsafe 1/17 accepted, 0/2 false HOLD among these two authored positives. 16/17 adverse rejections are NOT general model accuracy, provider authentication or real-world security guarantees.

**PRESERVED FAILURE:** cold_restart_nonce_replay still ACCEPTS a second identical signed receipt because its consumed nonce set is memory-only and is reset between verifier processes. In-process replay rejects. This is a source-level RED proving current preflight helper CANNOT be promoted to production provider admission. A true durable atomic nonce reservation/consume-once must be bound to existing host/Mono task authority, not supplied by model text or another Training runtime.

Actual red-green record: before receiver identity binding, two author-written pytest cases for signed wrong provider/model FAIL (2 fail, 1 pass). Adding provider/model/host/verifier/manifest/endpoint to separately injected receiver witness made those GREEN. Added test with actual Mono task checkpoint moving ledger journal head; previously signed old-head receipt rejects. Added explicit reproducible cold restart test that asserts dangerous second admission can still happen; it is a preserved red behavior, not a fixed implementation.

Measurement: 150 baseline and 150 candidate Python calls under tracemalloc, one noisy snapshot: baseline 3,955 ns/call, candidate 3,061,193 ns/call, peak traced allocation 19,904 bytes. Includes task SQLite read and HMAC, NOT real inference latency, network transport, loaded model memory or process RSS. Legitimate signer rotation, temporarily missing receiver observation, and pruned historical source can cause a false HOLD unless separate provenance/currentness recovery is demonstrated. Current tiny fixture does not measure language-task utility or true negative transfer beyond this mechanism warning. Independent blind cases required.

> HOSTILE REVIEWER: A signed synthetic witness from the same harness does not independently verify that a process loaded the asserted model, and a nonce reset defeats replay checks.
>
> ACCEPTED: All positive classifications are synthetic preflight results. No real issuer, loaded process, model output, source origin, Pre-Active DONE/run or protected effect is admitted. Retain explicit cold replay failure and HOLD production adoption.

## Host / model feasibility evidence

Lappy physical 32 GiB RAM (33,270,576 KiB visible) and RTX3050 Laptop 4,096 MiB VRAM; measured 2026-10-17 21:21 EDT free RAM 5,008,428 KiB (~4.78GiB), free GPU 3,962 MiB, both volatile and not reserved. Existing metadata-only GGUF candidates: Qwen3 4B ~2.326 GiB disk, Qwen3.5 4B Q5 ~2.862 GiB, GLM24B IQ2_XS ~6.712 GiB, GLM24B Q4_K_M ~13.349 GiB; estimated 4K FP16 KV ~576/528/640/640 MiB. These are NOT observed loaded-process RAM/VRAM, measured throughput, model accuracy, endpoint trust or qualified inference. Qwen and GLM remain separate prospective configurations. No model load, GPU benchmark or training while Codex shares Lappy.

## Integration, PORTAL, VN1 action

Integration: retain Mono as sole TASK_OPENED/source/CAS and no-provider source owner. If future provider admission is authorized, choose an existing durable atomic task/host claim+nonce store and receiver-measured selected model/process/session/route/transport evidence; a 25-field self-report, test HMAC or authored upstream origin does not qualify. No rewrite of existing #38 selected source composition without its own Linux target/source/import/guest tests.

PORTAL: #373 V1 source proposal accepts canonical offline task + optional Pre-Active/Volition, not a claim of minimal installed 431 files. #375 edge typing separates consumer research from selected source. Codex A1 #55 Windows FileBridge intermittent PermissionError RED remains owner task; Training has not edited or retested it in this increment. Do not infer Project Runner or provider live selection from host GET.

VN1 independent hostile request: rerun exact Training Draft and digest; independently challenge same signed foreign provider/model, different process-loaded artifact, actual changed Mono journal head, trust revocation and two verifier processes each accepting the same nonce; probe valid key rotation and missing witness false HOLD. Demand actual receiver-owned issuer/channel and loaded model evidence before promotion, not harness-generated flags. Include host, runtime, exact source head, new negative cases and disagreements. No model/GPU/QEMU/effects without separate permission.

No main merge, installed Desktop cutover, QEMU, Buildroot, trust/credentials/provider mutation, canonical memory write, physical disk, paid compute, model inference or protected task/effect.
