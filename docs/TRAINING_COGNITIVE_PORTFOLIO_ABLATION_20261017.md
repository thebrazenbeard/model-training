# Training L2/L3: source-bound cognitive portfolio ablation

Date 2026-10-17. Owner model-training Draft #106. Scope SOURCE ONLY.

## Fresh provenance

- Private 96-repository whole-portfolio review: chat-communication-bus Draft
  #371 at 951e7af1a97694faaf6bc18d51b35f70982e31fc. Read the catalog
  (96 heads/trees, 76 representative source reads), convergence document
  and immutable PORTAL checkpoint 543370fa7a57f751842c6d72b4484814c5509063.
- Bus #368 fresh-read (83 comments at this cut). Four overlapping chats:
  PORTAL / Integration / Training / VN1. Only Training #106 mutated.
- Mono main 16a2828dfa7fb7e6c0e19004f014bc68fc47badf,
  QualifiedVeraRuntime blob 48e00014b89adeb392102ef45849fbd49fc9dc29.
  Integration-owned Mono #77 current 959211d0ad237fa62006840f075c76c01deafb28,
  local_conversation blob d243d2efd01777655e511d91ea816fd804488ecf.
- VeraOS Integration-owned #38 a84dcfe037d35b162e3922ba770fce30d9f400cb,
  resident source blob 76c6e60b828a1ce386985f48f40e6387f5756662.
  PORTAL #37 df04f2cc01bb23e28543c3cea40cd462ba960fd2.
- Training #106 source before this experiment a48c86de47f2e71ed567e71f9de570de7c9c714c.
  Prior Mono77 readback fix (canonical task ID / DONE / one route) PRESERVED.

## Single governed cognitive pipeline

1. UI exact text + explicit conversation/message locator (not authenticated user).
2. Mono #77 open_message binds literal input to canonical task ID,
   source digest and current journal head, checks corrections/cancellation.
3. Governed context selection from current Mono state, identity and
   memory provenance; no retrieved memory becomes canonical truth.
4. Optional salience/Volition PROPOSAL under existing policy; neither
   drive nor initiative grants tools, external effects or user consent.
5. Rezon reasoning or planning only through a tested explicit input/return,
   not another task ledger or independent supervisor.
6. Model selection is advisory; host-owned verifier must attest exact
   provider, process-loaded bytes, host generation/session, task/head,
   endpoint, current nonce, response transport and revocation.
7. Pre-Active durable model execution/assistant response, bounded text.
8. Advisory Cricket/Voss review; critiques are not provider signatures,
   independent observers, or automatically correct.
9. Training's source readback requires Mono task-ID dedup, DONE,
   COMPLETED, single route + exact result text; Mono CAS/cold recovery.
10. Memory influence/learning proposal enters existing governed Mono
    admission separately. No silent canonical-memory or weight update.

## Measured RED -> GREEN, no model execution

Original cognitive_continuity_eval_v1.guard_prediction required
exactly ONE active source in scope. When two distinct active source
event IDs agreed on the identical literal, it incorrectly returned HOLD
for an exact sourced answer. Kept task/source digest and correction
guards; allow answer if all active in-scope records agree, cited event
is active, and claimed text equals the cited literal. This improves
evidence-reference usability, NOT likelihood of objective truth.
Identical sources may originate from the same upstream claim. No
independent corroboration or new permission is inferred.

Source regression tests/test_cognitive_agreeing_sources_v1.py:
- Initial prechange 11 source tests: 3 FAILED (two supported
  agreeing events, frozen score); remaining eight negative controls
  passed. Red was reproduced against prechange Training code.
- Postchange tests cover agreeing original and second citation,
  out-of-scope distractor, conflicts, duplicated IDs, fabricated
  citation, stale task head, superseded source, valid exact correction
  and contradictory correction. All tests pass locally.
- Existing earlier five-case false-HOLD developer counterexample
  changed from expecting HOLD to correctly expecting ANSWER.
- Frozen runnable ablation evaluate_source_agreement_ablation_v1.py,
  SHA of canonical six-case input/labels:
  f5f363fe736c6d3384ec7c4832fda78709e31a93e6d9545b047b84014c3a0fc6.
  Previous unique-event policy: 1/3 supported answers preserved, 2/3
  false HOLD, 3/3 unsupported cases held, 0 unsafe acceptance.
  Candidate agreement policy: 3/3 preserved, 0/3 false HOLD,
  3/3 unsupported held, 0 unsafe acceptance.
- Local microbenchmark 6,000 synthetic guard calls per policy:
  old 16,087 ns/call, new 12,731 ns/call; Python tracemalloc peak
  1,024 vs 960 bytes. Timing/memory are noisy interpreter-level
  footprints, not Qwen inference latency, VRAM or independent effect.
  Do not promote to performance advantage without repeat/multi-host.
- This is a *developer-designed* test, not a held-out real model,
  not evidence of learning, improved general reasoning or true factual
  corroboration. Existing Qwen and GLM24B remain unqualified.

## Ranked source-mechanism adoption matrix

Rank | Source / observed executable mechanism | Adopt? / falsification
--- | --- | ---
0 | Mono QualifiedVeraRuntime, task CAS, identity/correction, governed memory; Pre-Active durable result | Already selected SOURCE baseline; close source/guest/host qualification. Test interruption and exact task result first.
1 | Rezon @ e14fc692 replay.py blob a8b8cf5a, typed ReplayCase/ReplaySource/Disposition; source-absorbed into Mono | Test replay operator vs exact Mono source-guard + one model, frozen task and correction pairs. No new state authority.
2 | Cricket @ 0bae95bc reviewer.py blob 6f7f8992, rule + semantic + optional critic review | Advisory only. Score detected false claims AND false critiques vs baseline; critic output never signer/independent fact.
3 | Volition @ 64e61c5 engine.py blob d271183d, typed Policy, protection veto and bounded proposals; Mono already has initiative | Use existing initiative kernel only for user-scope eligible proposals; test false proposals before acceptance, no effect grant.
4 | Mosaic @ a2d4dc9 residency.py blob e58282f9, SpecialistResidency/ResidencyEnvelope/evaluate_swap_sequence | Research until measured per-provider gain, swap delay, RAM/VRAM/energy and interruption prove value. Single Qwen baseline preferred.
5 | Noema @ d6251fe svf0_evaluator.py blob c237800c, frozen-seed prediction metrics; LGCM @ 9b9081b evaluation.py blob ec10d3f6, prequential numerical MSE | Independent research methods. Not neural language skill or live learning; measure a separately governed proposal only.
6 | MESO-CRCT @ d821406 appraisal.py blob 19d6fe4e; HC Brain @ f7ffd4a cognitive_loop.py blob 5a4c98c6; God Brain @ a4ad8fb8 cognitive_loop.py SAME blob 5a4c98c6 | Duplicated/synthetic research, not two independent cognitive engines or observed wants. Do not install.
7 | Transcendence @ 4da8646, BT2 @ 0f35778, semanticatlas @ 2cd42a9, Semiotics @ 857d912 engine blob 1094d5bc, SPM @ 0dfe59a state_runtime blob 188b90e8, SQL Connectome @ b609fce | Source-indexed research/semantic tools; evaluate context-specific referent/SQL test improvements only, no semantic authority hierarchy.
8 | Roots @ 24ed94d, DeepMemoryStorage @ 3952068, private Redworm @ c87f93d historical provenance | Read-only, privacy-aware memory evidence candidates; no source becomes autobiographical memory or task authority.
Separate | Vera Model GLM24B @ 682511b and Qwen3 4B development reference; three HF donors | No provider selected by file presence, manifest, synthetic PASS or local HTTP. No weight execution/training in this test.

Full 96-repo portfolio review is an entrypoint source inventory, not a
deployed consumption receipt, and the above rankings are engineering
priorities rather than per-donor demonstrated behavioral superiority.
The exact sampled same-blob HC/God Brain identity is proven at SOURCE level.

## Next falsifiable experiment and cost of overholding

Separately authorize actual model execution before any Qwen/GLM run.
Preregister a new, blind task-level dataset of supported/unsupported,
contradicted/corrected/context-switched records. Freeze same precomputed
model responses, compare no guard, original one-source guard, and current
source-consensus guard. Report correct answer with exact source/event,
false-HOLD, false-accept, correction latency, task interference and
peak memory. Measure throughput and provider latency on the serving
host separately. Include source duplication and circular citations as
adversarial controls. Existing 6-case corpus is neither blind nor
sufficient for model-selection or production admission.

## Exact real-model HOLD

Current pre-existing Training resident_os_composition_gate_v1,
resident_verifier_port_v1 and resident_model_turn_readback_v1 are
source-only contracts; their positive synthetic receipts do not grant
real dispatch. Training #106 protects task/head, nonce, endpoint,
provider/model ID, signed metadata, host generation/session and
Pre-Active DONE + unique route, but production loaded process/byte
linkage, independently authenticated session/key/current revocation,
durable global nonce and response transport provenance are still
missing. Integration-owned OS38 selected Buildroot Mono/PORTAL pins and
PORTAL OS37 terminal/health changes still require exact owner
reconciliation. VN1 must attest actual Lappy serving selected loaded
model bytes, rather than the /api/tags inventory or stored hash.
No model inference, QEMU trial, provider activation, train, key
provisioning, deployment, canonical memory write or protected effect.
