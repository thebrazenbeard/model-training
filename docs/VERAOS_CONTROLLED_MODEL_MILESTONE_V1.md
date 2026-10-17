# Next VeraOS cognition milestone: controlled local-model trial

Status: Training source-only development proposal; NO route selected/installed.
Scope: one operator-authorized, no-tools, no-memory, single-task local text
experiment via existing VeraOS PR31/32 native admission/execute APIs and
Pre-Active store. Normal resident remains HOLD until VeraOS/PORTAL owners
separately qualify/authorize provider and observe installed consumption.

## What actually worked

PR32 diagnostic selected /sbin/vera-virtio-probe, not /sbin/vera-init.
It reported Qwen3 4B Instruct Q4_K_M inference and a durable COMPLETED
Pre-Active result, zero protected tools. A Windows Ollama host served the
model through a network-host relay (host.docker.internal:11434), private
Docker AF_UNIX socket/QEMU VirtIO serial, and guest local
127.0.0.1:18766/v1. Guest RAM was 768 MB; this is NOT Ollama host RAM/VRAM.
Exact model-serving physical host and loaded raw model-file identity are
not independently established by the report. Ollama /api/tags digest
is inventory, not actual hashed neural weights. Diagnostic PASS is fake.

## Minimum NEXT supervised development experiment, not production PKI

1. Source-frozen, reviewed VeraOS resident with explicit operator-owned
   one-task no-effect scope and empty tools/memory/identity authority.
   No implicit route activation from Training/Integration source receipt.
2. Keep known Qwen3 4B as the text-only development control. Find the actual
   Windows Ollama serving host; obtain receiver-owned live session/generation,
   nonce, model artifact and RAM readback on THAT host, bracketed by a second
   generation/session check. Do not substitute QEMU/VN1 telemetry.
3. Verify actual serving model artifact bytes (or precise Ollama blob/manifest
   chain) separately from the reported Ollama inventory digest. Pin exact
   local target model, endpoint, API health and bounded response policy.
4. Reuse Vera Mono/Training task semantic/dissent receipts and Integration's
   durable task-journal checks. Preserve independent logical contributors;
   add receiver-owned one-use task/nonce consumption before use. Model text
   never determines permission, replay, identity, effect or task acceptance.
5. With exact authorization, make ONE local text request in normal resident,
   preserve request ID, actual endpoint/model and loaded artifact link,
   durable Pre-Active run/result, replay/failure controls and observations.
   A held/failed run is useful evidence, not a synthetic success.

Development sandbox does not need production PKI in advance. Locally
observed host/artifact/session, explicit no-effect permissions, and actual
application readback are sufficient for a supervised experiment. The
source fixture below CANNOT authorize performing it.

## Runnable source test adapter

controlled_resident_trial_v1.evaluate_controlled_trial accepts a
ControlledTrialTarget, two ServingSession readbacks around a bounded
ArtifactResourceSample, synthetic artifact bytes, existing Training
DevelopmentAssessment and actual two-participant cognition receipt/join,
plus fixture-only one-use nonce set. Coherent fixture => status
ACCEPT_SYNTHETIC_DEV_PREFLIGHT. Cross-host/session, stale sample, forged
bytes/identity/semantic source or replay => HOLD_SOURCE_PREFLIGHT.
ALL outcomes have normal_resident_admitted=False,
model_identity_authenticated=False, inference_executed=False, and
effect_authority=False. Caller-supplied real-host evidence always HOLD.
A production receiver must replace synthetic bytes and in-memory nonce
set with trusted host readback and durable task-journal admission CAS.

## Production gates (separate)

Eventually require trusted issuer signatures/trust policy, verified
base/tokenizer/adapter/decoding/runtime lineage, actual model artifact
attestation, durable replay/rollback protection, independent held-out
capability evaluation, authorized route selection and ongoing drift
review; see VERAOS_NORMAL_RESIDENT_MODEL_EVIDENCE_V1.md. No such gates
are required to run this synthetic source test.

## Model choice and cognition capability

KEEP existing PR32 Qwen3 4B as control; it is not the trained Vera model.
Separate Qwen3.5 local typed-source development evaluation achieved 13/25
correct with codebook, including 0/5 fully correct authority typed values.
Use deterministic source gates for permission and receipt semantics.

AUGMENT: a Vera-trained base-plus-LoRA candidate exists in the authorized
local model tree. It has a separate checkpoint, quantization, adapter,
tokenizer and host resource subject. Do not transplant it into PR32 Qwen3
Ollama tag or publish private artifact paths/IDs in this public repo.

REPLACE only after privately pinned combined model + resource fit, held-out
referent/correction/scope/authority/recovery results and a supervised
same-task VeraOS no-effect result outperform the control. Bootcamp's
identity capsule/proxy tests are not neural weight training.

No model ran, training happened, or route changed for this work.


## Explicit two-stage OS admission handoff (Training source adapter)

The tested prepare_resident_admission_handoff(target, decision) returns
SYNTHETIC_WIRING_ELIGIBLE ONLY for a coherent synthetic source preflight;
all other input produces HOLD_UNVERIFIED_REAL_ADMISSION. Both responses
contain a VERA_OS_COGNITION_PROVIDER_V1-shaped descriptor with
qualification.status=HOLD, admission.status=HELD, all authority flags false.
It is accepted by the exact VeraOS cognition_provider.validate_provider
shape checker but is not eligible in provider_route_eligible.
validate_resident_admission_handoff rejects fabricated PASS, fake admission,
changed artifact subject or route/status escalation; this is source
consistency checking, NOT authentication of model, issuer or source.

The genuinely verified resident admission belongs INSIDE VeraOS. The
authorized OS host verifier must independently bind enrolled Windows
serving-host generation/session and one-time nonce to the exact Ollama
model manifest+model blob bytes, current CPU/RAM availability, endpoint
identity, task continuity and authorized PORTAL local-text route. Only then
may VeraOS produce a SEPARATE trusted-qualified provider with actual
signature/issuer/trust-policy evidence appropriate to the local controlled
trial and recheck the task before native_cognition.admit_local_text_route.
Training NEVER emits PASS. For synthetic fake-model wiring, keep any
testing override exclusively in an OS test harness, not an installed
normal-resident default or user-reachable provider config.

Deferrable until production: long-lived multi-issuer PKI/revocation,
trained-Vera model replacement, global Intranel sender attestation,
Executor protected-effect gateway, continuous behavioral qualification.
NOT deferrable for an actual bounded resident trial: explicit operator
authority, host-owned artifact and nonce checks, same-session resources,
limited no-tool route, and observed durable Pre-Active request/result.
