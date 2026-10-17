# VeraOS Qwen development cognition: real-result binding V1

SOURCE ONLY ? not installed model admission, actual inference, or task closeout.
Code: resident_model_turn_readback_v1.py.
Tests: test_resident_model_turn_readback_v1.py;
test_resident_model_turn_actual_store_optional.py;
test_veraos36_callback_hostile_optional.py.

## VeraOS #36 synthetic callback: concrete falsifier

Source exact 79b2562cb2b30fafae712e64a1d260bcaea2baaf,
resident.py blob c4433b6c22cea77cc8225a9e21d9506e5b9470ae.
Unmodified ResidentHandles AST executed on Lappy with stubbed native
functions: a self-declared PASS provider and arbitrary in-process
callback returning SYNTHETIC_SOURCE_FIXTURE_ONLY reached native
admission and execute boundary. No model, endpoint or network invoked.
The ordinary CLI still does not inject this path. The finding is a
deployment shortcut HAZARD, not evidence of installed exploitation.

PORTAL-owned correction: remove string-based synthetic admission from
production resident.main. Preserve the fixture only in a noninstalled
test harness with a fake, non-network model adapter. For a real turn
compose independently trusted host verifier in the installed VeraOS
resident, using Vera Mono OutboundTrustRegistry PROVIDER role
currentness/revocation and independently provisioned verifier key,
registered Lappy generation+connectedAt+session, receiver-generated
one-use nonce, loaded Ollama manifest/blob association, bounded
request/endpoint/task and an atomic durable task-head/nonce CAS.
Only a separate OS-owned trusted decision may enter normal
native_cognition.admit_local_text_route; test callback strings,
claimed PASS and caller-provided status are NOT admission.

## Runnable result-readback adapter

After a separately authorized and genuinely admitted single turn,
the OS owner can adopt the source algorithm:

    verify_resident_model_turn(store, expected, result, transport)

ExpectedTextTurn binds exact stable request ID, task ID and text,
declared semantic digest, provider/model, guest loopback endpoint,
serving host/generation/session, and separately sourced model blob
and manifest digests. The semantic digest is explicitly DECLARED;
Integration's true current Mono task journal gate must verify it.
ModelTransportEvidence is caller INPUT, not signed attestation.

The algorithm checks ACTUAL pinned Pre-Active SQLite event dedup_key,
task payload, run.source_event_id, completed state, bounded final text,
no failure, empty tools/capabilities, last assistant message and
VERAOS_COGNITION_ROUTE_SELECTED journal in one committed read-only
snapshot. It rejects uncommitted state, changed task/ID, missing or
failed run, changed or empty text, forged provider/model/host/session,
route or blob mismatch, and a self-declared verified_by_os=true.
Positive = SYNTHETIC_STORE_LINK_VERIFIED with display_text and
attribution UNVERIFIED_MODEL_SOURCE, never proven real Qwen consumption
or canonical task completion. Holds suppress display_text.

Tested both with synthetic SQLite and ACTUAL pinned Pre-Active
Store source 1f23a809d7274df506e03e2d9052525538c3bcf6
(store blob 4612d4c621e01ce4ee69ad5702c4ca37fa7e3402):
enqueued source event, run, assistant message and route journal;
successfully reopened from disk, then terminal corruption HELD.
NO engine, provider or model invoked.

Crucial remaining falsifier: lower-level Pre-Active Store.update_run
can rewrite terminal status and final text without a journal entry.
The adapter catches mismatches, not coordinated tampering of
all mutable fields. True causal loaded model/response attestation
needs a trusted OS-owned signed response receipt or append-only
authenticated transcript bound to active host session and task head.

## Placement and scope

KEEP Qwen3 4B Q4_K_M on Windows Lappy via Ollama and the
private host relay / AF_UNIX / VirtIO serial to network-isolated QEMU
guest local 127.0.0.1:18766/v1. Do NOT treat VN1, Docker relay,
and QEMU guest as serving-host-equivalent. Guest's 768MiB RAM
is not the 2.5GB neural weights. Verified stored on Lappy:
Ollama manifest SHA256
0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0;
model layer SHA256
85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9,
2497280480 bytes. Inventory size 2497293803 bytes is
a separate manifest/inventory subject. Read-only GET /api/tags
identified Qwen, but GET /api/ps returned NO currently loaded models
during this investigation. No new inference was run.

First authorized normal graphical text trial: verify actual
loaded model and host session, one-use request and fresh host
resource/route, exact response readback, and display a bounded
model-sourced contribution without attributing it to a trained
Vera model or independent mind. Existing native Pre-Active route
already uses empty ToolRegistry, context max_chars=4096, max_steps=2,
zero autonomous turns. No model decides identity, correction,
instruction priority or effect authority. Failure/UNKNOWN remains
visible as failure rather than manufactured COMPLETED.

Integration #6 owns the live task currentness, original source
referents, semantic fidelity, correction/scope and dissent provenance,
and existing Mono task-head CAS. Training only verifies a source
readback; a Pre-Active COMPLETED is not Mono canonical task acceptance.
A separately qualified Vera base+LoRA is future comparative candidate,
not a prerequisite for the working Qwen development control.

No installation, QEMU/model run, paid compute, credentials,
trust mutation, main merge or canonical-memory writes.
