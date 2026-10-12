# Vera neural model → P.O.R.T.A.L. → VeraOS target

**Status:** Source-level training and integration contract, not an installed route, a trained-weights receipt, or permission to launch training.

## The product being trained

Train a **real standalone Vera neural model** that can run directly through **KoboldCPP** after reproducible adapter export, merge/quantization and GGUF validation. Preserve the standalone product even if VeraOS is offline or P.O.R.T.A.L. is replaced.

This is **not** the zero-cost identity bootcamp's Trainer/Student/Examiner proxy and capsule. That public bootcamp has no neural-weight mutation and its proxy PASS is never evidence that the standalone neural model was trained or qualified. Do not feed private transcripts, secrets, or personally sensitive records into the public training workbench.

The primary weights-level objectives are semantic/pragmatic fidelity, Vera-consistent independent judgment, epistemic discipline, corrective state interpretation, long-context reasoning, tool/delegation comprehension, and robust KoboldCPP inference. The precise priorities and independent held-out evaluation families are in `architecture/VERA_MODEL_PORTAL_VERAOS_TRAINING_TARGET_V1.json`. Numeric acceptance thresholds must be preregistered against a frozen evaluation panel rather than invented after seeing candidate results.

Do **not** train mutable runtime configuration, secrets, credentials, live permissions, P.O.R.T.A.L. endpoint values, operating-system policy, current autobiography, or right-to-act claims into model weights. A model should **understand** an externally supplied authority/currentness envelope; the runtime must **enforce** it.

## Intended operating topology

1. `vera_model_training` builds and independently evaluates weights/adapter; verified GGUF remains a usable **standalone KoboldCPP model**.
2. Candidate export records the exact base revision, recipe, dataset manifest, adapter weights, adapter config, GGUF digest, quantization and context declaration. `training/verify_vera_model_handoff.py` verifies the **local candidate bytes only**.
3. A separate evaluator must independently qualify the exact exported model, including direct KoboldCPP inference and held-out semantic, pragmatic, relational, tool-planning, and authority-boundary regression suites.
4. **Pre-Active** owns the intended persistent, exact **active ModelTarget** binding. The live model/endpoint/provenance probe must match that target. Its mere port, process, name or file existence is insufficient.
5. **P.O.R.T.A.L.** owns orchestration and accepts the **qualified selected target** for human and autonomous cognition; it must record the actual request → route → candidate artifact → response chain and fail closed when the selected target is unavailable or mismatched. No implicit substitute based only on `vera` naming or an Ollama model tag.
6. **Vera Mono** owns persistent Vera identity and governed runtime state through `vera_core.QualifiedVeraRuntime`. Cognition outputs are not canonical memory, consent, trusted effect authorization or independent audit.
7. **VeraOS** supplies the operating environment and intended system-facing interface. The existing P.O.R.T.A.L. Desktop is a potential client/UI component in that environment, not the permanent identity root. VeraOS must remain provider-agnostic and boot even with no cognition model selected.

The architecture is **not** `model = operating system`; it is a standalone model serving cognition inside a persistent governed Vera system.

## Current source gap and acceptance tests

At the 2026-10-12 main-branch read, P.O.R.T.A.L. `portal/desktop_cognition.py` still auto-discovers Ollama tags and Pre-Active `/v1/models` and uses names and shallow adapter metadata to rank candidate routes. This is **source evidence of discovery, not proof of a persisted active ModelTarget or exact artifact consumption**. No running trained Vera model or actual P.O.R.T.A.L./VeraOS inference is claimed here.

Before route qualification, test:

- **Weights:** exact dataset/training/base/adaptation lineage and independently evaluated held-out behavior; no unexamined transfer of a bootcamp capsule grade.
- **Export:** SHA-256 and GGUF header validation; concrete KoboldCPP inference passes after merge/quantization.
- **Admission:** independent issuer signs or securely binds the exact candidate manifest/artifact, and live Pre-Active target matches the exact artifact/runtime. A JSON field claiming PASS is not an independent signature.
- **Routing:** spoofed `vera-local:latest`, a valid-looking adapter SHA for the wrong base, stale target/endpoint, missing provider, and unauthorized fallback must not cause acceptance. Test simultaneous human/Volition requests against the **same** selected model.
- **Consumption:** inspect real P.O.R.T.A.L. receipt and resident runtime acceptance bound to one request/route/model digest; do not infer consumption from discovery.
- **Interface:** an observed VeraOS/UI → IPC → P.O.R.T.A.L./Pre-Active → selected model → governed Vera response → UI roundtrip. The UI never mints protected-effect permits.
- **Recovery:** restart the provider and runtime, revalidate the target, preserve externally governed memory/state, and invalidate stale route bindings.

Run the source-level candidate tests with `python -m unittest discover -s tests -p test_vera_model_portal_handoff.py -v`. The candidate guard's successful status is `ARTIFACT_BYTES_VERIFIED_NOT_MODEL_QUALIFIED`. No provider is started, no weights are trained or changed, and no model is installed by this document or guard.

> **HOSTILE REVIEWER:** Why optimize weights for P.O.R.T.A.L. API details instead of training a strong general Vera model?

**Accepted.** Embedding transient host commands and permission semantics in weights would overfit implementation details and turn generation into counterfeit authority. The surviving plan trains transferable cognition and scores its ability to interpret typed host state; exact model-route admission and external effects stay outside weights.

No direct `main` mutation, merger, credential change, private-data disclosure, release, installation, or paid compute is authorized by this source-only plan.
