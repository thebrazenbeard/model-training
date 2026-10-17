# Training L2/L3 Qwen resident source trust attachment V1

SOURCE ONLY: no installed runtime, no inference, no provider grant.

## Exact composition

Mono main 16a2828dfa7fb7e6c0e19004f014bc68fc47badf
qualified_runtime.py blob 48e00014b89adeb392102ef45849fbd49fc9dc29.
QualifiedVeraRuntime.from_state_directory() composes outbound_trust and
tasks from the same VeraStateDirectory. provider_authority_verifiers
must be current in OutboundTrustRegistry role PROVIDER, exact ID/key.
Host supplies real verification key; registry does not store key
material. Mono PR76 36ae11802808bc1b1392a361716109d1022ee6af
TaskExecutionLedger provides existing atomic expected_task_head CAS.
Integration PR6 df968e87a3af955d52d69c86c7a5db41e5795c6b
establishes the exact HTTP handler(task,request_id) ABI and original
canonical message/source binding, with no automatic dispatch.

OS37 59a4d8f0d26c00104cf1ea6d87249d94fc5a58ad keeps ordinary
CLI no-provider HELD. Its in-process Python conversation_template and
arbitrary host_admission_verifier callback returning literal
SYNTHETIC_SOURCE_FIXTURE_ONLY can still reach a network-capable
native_cognition OpenAI-compatible adapter. This string is not
verifier authority. PORTAL owns removing test path from shipping
resident.main, isolating synthetic work in a non-network test harness.

## Training source API ? actual Mono trust + task consumed by PR37

resident_os_composition_gate_v1.py:
- ScopedResidentTurn(model_request,request_id,task_text,source_digest,
  expected_task_head): exact Qwen control host, generation/session,
  endpoint, manifest/blob hashes, loaded resource/freshness, nonce,
  signed whole task/head/request ID and COGNITION_TURN only, no tools.
- QualifiedRuntimeModelGate(runtime,os_owned_host_verifier): requires
  EXACT vera_core.QualifiedVeraRuntime, obtains outbound trust and
  existing task ledger FROM runtime. Verifier is injected once by
  trusted OS startup, never from HTTP or model descriptor.
  Host verifier port: verify_signature(message,signature,key_id),
  reobserve(request) including independently attested loaded serving
  PID/blob/manifest/session/resource, consume_once(request) with
  durable atomic host-wide nonce semantics.
- reserve_source_trial(turn,signed,now_epoch_s): checks real Mono
  outbound PROVIDER currentness+revocation, signatures and exact host,
  fresh loaded process artifact/endpoint/nonce, canonical task
  literal/source/head, not closed/corrected/dependent/delegated.
  Calls host nonce consume then original TaskExecutionLedger.checkpoint
  with expected_task_head CAS. Wrong task/head, signature, host,
  stale sample, session, provider, revoke, replay or permission HOLD.
- ScopedSourceLease result is never a real-model admission:
  real_model_admitted=false, network_dispatch_authorized=false,
  identity/canonical/effect authority=false. This is intentional.
- make_held_resident_conversation_handler(gate,turn,signed,now_epoch_s)
  directly supplies PR37 desktop.open_loopback_server's exact
  conversation_handler(task,request_id) signature. It refuses changed
  HTTP text/id; even verified source reservation returns BLOCKED
  (HTTP202), not any model text. This adapter has no outbound client,
  network adapter, model factory, tools or callback PASS.

HOSTILE REVIEW ACCEPTED: any Python arbitrary callback returning
true with fabricated loaded-process fields is still fake if that
callback is not an OS-installed, independently trusted verifier.
The source contract must NOT be interpreted as a real route permit.
Only PORTAL's OS-owned installed provider verifier + native adapter
may eventually authorize actual Qwen development admission. Avoid a
new issuer hierarchy or provider registry.

## Tests and remaining evidence

Actual pinned Mono QualifiedVeraRuntime source instantiated in
disposable VeraStateDirectory; real OutboundTrustRegistry PROVIDER
registration/currentness/revocation and TaskExecutionLedger checkpoint.
Tests reject fake signature, changed literal/source/head, revoke,
missing process-loaded observation (inventory alone), host/session/
generation substitution, unauthorized browser action, closed tasks,
and same-task replay over freshly reopened ledger/verifier.
The host nonce fixture is in-memory, NOT production durable.
Exact pinned OS37 desktop source HTTP POST verified it consumes our
two-argument handler, returns BLOCKED for valid and changed text,
and creates one committed source checkpoint with no model call.

Remaining real trial requirements:
1. PORTAL installed OS image build/source hash, real PID1 startup,
   remove synthetic network-capable callback and select OS-owned
   provider verification before native admit_local_text_route.
2. Genuine Lappy host verifier public key already authorized in
   Mono OutboundTrustRegistry with current PROVIDER role; OS key
   installation/provisioning has not been performed.
3. VN1 actual Lappy enrolled session/generation, Ollama loaded
   PID/process selected model bytes/manifest and endpoint, current
   RAM/VRAM and response causality. Stored Qwen SHA and /api/tags
   inventory do NOT prove loaded or actively served model; /api/ps
   previously empty.
4. Durable host-wide nonce uniqueness across sessions, tasks and
   restarts; recheck revocation right before actual dispatch.
   Host nonce consumed before Mono CAS may burn a nonce in a race,
   which is safe HOLD. Source checkpoint does not ensure exactly
   once model execution after crash.
5. Integration original conversation_id/message_id and exact
   accepted task source/semantic digest/head, cancellation and
   correction recheck, followed by PreActive event/run/message/journal
   readback and independently authenticated model response.
6. Patrick exact permission before any new real inference, boot,
   model activation, installation, paid compute or protected effect.

Keep Qwen3 4B Windows Lappy Ollama as development reference; QEMU
guest via isolated private VirtIO relay. Trained Vera/donor weights
remain separate and not prerequisites. No effects or model load.
