# VeraOS scoped verifier port V1 (source adapter)

Executable implementation: resident_verifier_port_v1.py
Executable tests: tests/test_resident_verifier_port_v1.py

This is DONOR/SOURCE only. It is neither installed VeraOS code, nor
a credential/issuer authority, nor model admission.

PORTAL/OS composition contract:

1. Inject an independently authorized *installed* VeraOS host verifier
   with verify_signature(message, signature, key_id), reobserve(request),
   and consume_once(request). Source HMAC keys and one-use sets in the
   unit test are disposable fixtures, NOT real trust.
2. Reuse existing Mono OutboundTrustRegistry.assert_current with
   role PROVIDER, exact provider/key/issuer and non-revoked generation.
   Do not create a second issuer hierarchy. Verify actual installed key
   material separately; a currentness receipt alone is not signature proof.
3. ResidentModelRequest signs and pins exact provider, Qwen model ID,
   host/session/generation/nonce, loopback endpoint, manifest/blob bytes,
   sample time, expiry, task ID/digest and cognition-only permissions.
   Host reobserve must independently read actual serving model and task
   journal, not copy fields from the submitted request.
4. evaluate_resident_admission(TEST_ONLY) can return
   TEST_VERIFIER_SCOPED_ACCEPT ONLY for an instrumented fixture. It
   always keeps normal_resident_admitted false. Any real-unverified mode
   remains HOLD even if the submitted fields are plausible.
5. consume_resident_admission rechecks at use, requires one-use nonce
   checkpoint, and passes a HELD VeraOS provider to a test-only sink.
   It cannot call native_cognition.admit_local_text_route or run inference.
6. For a REAL bounded resident trial, OS owner must independently install,
   audit and bind the verifier to trusted host, key, task and durable nonce
   state. Only that OS-owned verifier may mint an authenticated provider
   decision for the EXISTING native_cognition/PORTAL/Pre-Active path.
   No source fixture or caller-relabelled PASS may become route-qualified.

The current real blob stored on Lappy matches the Ollama manifest
(whole blob SHA256 85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9),
but historical/current *loaded* model and normal PID1 route consumption
are not established by disk integrity.

HOSTILE REVIEW: a fake verification callback and fake registry can return
apparently good values. The simulated acceptance proves only source
control flow; Python type names, matching data and unit-test MACs do not
prove an actual installed OS trust boundary. Provider currentness, actual
artifact association, atomic nonce consumption, and task generation
must be independently observed in the trusted resident when authorized.

Full production PKI, model-learned competence, trained Vera model
replacement, remote Intranel trust and Executor protected-effect
gateway are NOT prerequisites to the no-effect fixture.
