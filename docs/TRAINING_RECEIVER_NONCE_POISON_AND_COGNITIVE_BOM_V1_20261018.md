# Training–Integration receiver nonce and cognitive BOM V1

2026-10-18 | Training Lappy + Integration VN2 | Source research only; provider and memory admission HOLD.

## Current owners

Training parent #113 51bd9749c2bf77b9b2b3f9d94f55a56e3ef29896; Mono #77 a351d36274595cde26bc60df4c44c3b83a6d1f0a; exact exercised Mono TaskExecutionLedger Git blob b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de; VeraOS #38 4346971cedab05a771a3e6e70fa56a613cc47e06; PORTAL #385 7ae82abe5a80a5d3f15d07052ad4eb0e3280b125. All selected owner PRs were Draft at source read. PORTAL pairs Integration(VN2) with Training(Lappy), Codex(Lappy) with independent VN1. Comments do not constitute independent sign-off.

Current task/head ABI established by the Mono and VeraOS owners: opened_head = immutable TASK_OPENED digest; expected_task_head = latest event digest scoped to that task; TaskState.journal_head_digest = GLOBAL event-chain head. Training #113 fixed a separate synthetic provider-preflight false HOLD caused by comparing these two head types (17 focused tests passed). Owner Mono/VeraOS A → B → replay A correction is separate; VN1 independently reported disposable cold source/host PASS for those exact owner heads, not provider admission.

## New falsifier and executable results

Existing Training #112 research reserve() serializes a global scan plus a general Mono task CHECKPOINT using Mono's action_lock. A public scope key is SHA256(receiver_id, host_generation, provider_session, nonce). The checkpoint payload contains a freely set next_frontier and lifecycle_evidence_digest. It is a source-only task record, not an authentic model/issuer claim.

NEW source evaluator: evaluate_checkpoint_poison_v1.py. Test: tests/test_checkpoint_poison_v1.py. Machine result: docs/evidence/TRAINING_CHECKPOINT_NONCE_POISON_20261018.json. Eight frozen scenario names SHA256 963fea4466057e3d13f0f89c286c9820c16ad2ca44c7cc06d6656a26724e953e.

An ordinary task checkpoint writer can insert the same public marker in a DIFFERENT task. The next eligible synthetic first claim is falsely rejected with DUPLICATE_SCOPE_NONCE, despite no authenticated receiver event. The marker passes real Mono task journal chain validation; the alleged victim task stays OPEN. This is task-checkpoint origin POISONING, not unauthorized provider execution. A marker with a different nonce or receiver scope does not block; clean first claim accepts. Another physical SQLite task root accepts the same nominal nonce: neither root has shared trusted uniqueness. Real separate Python child process os._exit(17) BEFORE marker commit leaves the later claim eligible. Exit AFTER the checkpoint commit makes the next claim reject, demonstrating persistence of a marker, NOT authentication of its origin. Caller-modified host generation remains accepted as another scope, preserving #112 spoofing RED.

Observed Lappy Windows Python 3.14.5 tests: 12 passed / 0 failed / 0 skipped, 1 pytest warning, on disposable SQLite. The 12 comprise six new falsifier tests plus six existing receiver-scope tests. Executed task ledger Git blob is identical to Mono current #77; local_conversation opening remained functionally compatible but the entire latest owner package was not independently executed. Resource snapshot at 10:17 EDT was only 855,660 KiB FREE RAM from 32 GiB installed, RTX3050 3962MiB free from 4096MiB; no model or GPU work.

> HOSTILE REVIEWER: A qualified receiver would not grant ordinary task writers permission to write trusted nonce events. This test attacks the deliberately untrusted checkpoint representation, not all hypothetical receiver-verified Mono protocols.
>
> ACCEPTED. Reject this specific checkpoint-as-nonce candidate. A lock serializes writes but does not establish the issuer of a marker, consent, provenance or canonical task effect authority.

## Required provider interface — proposal, NOT implemented

1. Mono canonical core owns immutable task literal, source/packet digest, referent/privacy scope, TASK EVENT head/currentness CAS and separate GLOBAL journal chain integrity. TASK_OPENED/HELD and cold recovery function with NO provider; no generated completion or memory writing.
2. An authorized receiver must obtain host identity, host generation, session and issuer trust/revocation from independent receiver-owned observation, not caller fields. It must bind actual process identity/PID/start/endpoint and process-loaded model artifact bytes/config. Installed GGUF header, ELF static inspection, signed-looking text or fake ADMITTED flag do not qualify.
3. Future consume_once_and_bind_result is a proposed single existing-authority transaction, NOT Training runtime code or another trust database. It must verify receiver ownership, task/source/head, issuer, selected route and process witness, atomically consume nonce unique in explicit receiver/host-generation/provider-session domain ACROSS TASKS AND PROCESSES, record a trusted receiver-only claim event and read back. Never count general TASK_CHECKPOINT as a consumed nonce. If separate physical hosts or separate task databases lack one authenticated collision-domain writer, HOLD rather than infer shared state.
4. Precommit crash: verify original task/source and nonconsumption before retry; postcommit ambiguous result: HOLD / UNKNOWN_NO_REDISPATCH until actual receiver result is reconciled. Distinguish claim durable from inference completion. A model request cannot self-certify response acceptance. Bind real Pre-Active DONE and COMPLETED run plus single selected route and response bytes before Mono canonical result CAS when that execution path is selected.
5. Generated memory proposal is not memory admission. Separate vera_memory authority requires source/referent/scope, consent/privacy/retention, provenance/correction and explicit permission; no automatic Volition, Cricket, Rezon, provider or training result promotion.

The selected/qualified existing host or Mono receiver-only event authority has NOT been located/verified in this Training experiment. Integration must explicitly return an authenticated owner/source/runtime witness or maintain HOLD. No issuer/host trust or protected effect was changed.

## Component-level cognitive BOM and VeraOS gate

| Component | Minimum offline canonical task | Current VeraOS #38 stage | Additional qualified option |
| --- | --- | --- | --- |
| Mono task/currentness/HELD/cold recovery | Required source path | Selected Mono 316 source files | One task/effect owner only |
| PORTAL | Full host Runner not needed for no-provider task | Selected embedded 89 PORTAL source files | External Windows host service separate |
| Pre-Active | Not intrinsically required for basic TASK_OPENED/HELD | Selected 18 source files | Execution event/run/result only when actual route selected |
| Volition | No boot authority | Selected 8 source files | Proposal/ABSTAIN only |
| Cricket/Voss, Rezon | Not required for offline task HOLD | No distinct mandatory guest engines | Advisory critique/reasoning; Rezon AUTO provenance REJECT |
| Semantic Atlas/Semiotics/SPM/SQL Connectome, memory donors | Scope-bounded source retrieval as needed | No new memory supervisor | Read-only provenance and separately governed proposal |
| Mosaic, model-training, Qwen, GLM24B, model providers | Not required for no-provider offline core | No admitted model by source staging | Loaded process/provenance/quality/host resources qualification HOLD |
| Ears/Vision, Noema/LGCM/HC/BT2/MESO experiments | Not boot required | No wholesale sibling imports | Task A/B evidence, explicit sensory consent, cost and adverse transfer |

The established Training #111 full versioned 96-repo / eleven-role capability matrix controls source classification. 147 proposed consumer links are NOT proven hard guest dependencies. Current selected guest staging remains 316+89+18+8 = 431 files. Mono-only #109 Windows cold TASK_OPENED/HELD source pass is not a built minimal Linux guest. Integration VN2 has verified Buildroot effective Kconfig and PT_INTERP static ELF scanner against host/synthetic source, NOT actual target Python/loader/SONAME closure, installed rootfs, Cog DRM pixels/input, normal guest PID1 or two cold graphical boots.

## Partner requests and remaining HOLD

Integration VN2: identify the currently selected existing canonical host/Mono claim owner if any. Test whether a receiver-authorized event can be distinguished from an arbitrary ordinary CHECKPOINT, whether uniqueness is atomic with task/source/head and survives crash or another ledger root, and whether selected OS38 431-source package closure actually includes required trusted host APIs. Return exact source/host evidence and dissent through private Bus #368; a Bus request does not constitute partner consent or executed tests.

VN1 independent falsification: run the frozen eight cases on exact new Training Draft and current Mono77; independently attack cross-task marker, copied root, real cold process exits before/after marker write, spoofed generation/session and two processes with a deterministic barrier. Include legitimate false-HOLD, issuer source and reviewer host/runtime/log hashes. VN1 alone owns independent acceptance.

Decisions: ACCEPT canonical Mono source-only offline HOLD and previously verified #113 typed task-head correction; REJECT general TASK_CHECKPOINT nonce authority; defer all optional cognition by measured task usefulness; HOLD authenticated receiver/provider, loaded-model artifact, cross-host atomic nonce, canonical memory, qualified graphical VeraOS. No new DB, scheduler, model, GPU inference, paid training, protected memory/effect, deployment or merge.
