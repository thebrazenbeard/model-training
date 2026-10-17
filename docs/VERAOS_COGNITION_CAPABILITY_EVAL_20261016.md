# VeraOS cognition capability ? bounded development evaluation

Training L2/L3, source-only, 2026-10-16. Qwen3 4B remains the existing
VeraOS operational development control; no inference or model training here.

## Existing executable Mono mechanisms

Exact Mono main 16a2828dfa7fb7e6c0e19004f014bc68fc47badf and
Mono task Draft #76 36ae11802808bc1b1392a361716109d1022ee6af.
Code actually inspected:
- vera_core/interaction_semantics.py: typed speech-act, contextual target,
  correction links, hedges, negation, fidelity; open-world ordinary text
  may correctly classify as UNKNOWN. Not a full language understanding model.
- vera_core/prequential_evaluation.py: evaluate_prequential uses the original
  predict, score, then update order; no model inference or weight updates.
- vera_core/corrective_learning.py: append-only staged correction cycle.
- vera_identity/contextual_interpretation.py: contextual, sourced semantic
  readings, supersession and revision history.
- vera_memory/learned_influence.py: durable governed revision/influence gate.
- vera_runtime/protocol/initiative_kernel.py: proposal evidence, candidate
  arbitration, score, policy and verification interface; NOT permission.
- vera_runtime/runtime_cohesion/salience_control.py: explicit attention
  allocation and salience-selection semantics.
- rezon/cascade.py and rezon/epistemics.py: bounded reasoning depth,
  proposition kinds and independent support kinds.
- vera_core/agi_ambiguity_probe.py: controlled synthetic ambiguity probes.

Source check: QualifiedVeraRuntime itself does not directly import
these named modules. That is limited to this file and does not establish
that they are unused across the entire portfolio. Runtime consumption
inside the VeraOS resident has not been independently shown.

## Executable Training evaluation

New files: cognitive_continuity_eval_v1.py;
run_cognitive_continuity_dev_v1.py;
fixtures/cognitive_continuity_devset_v1.json;
tests/test_cognitive_continuity_eval_v1.py.

The existing Mono prequential evaluator scores a typed structured
candidate response for status, exact source event, accepted task and
request IDs. Frozen evaluator-only expected labels are NOT passed into
the responder or guard. The guard uses exact sourced facts, referent,
scope, corrections, task-head/source-digest currentness and authority
separation; it never generates an answer. Non-answer statuses strip
source attributions and text, including any fabricated claims.

12 public synthetic turns cover 10 categories: multi-turn continuity,
ambiguity, contradiction/correction, semantic drift, uncertainty,
initiative, context preservation, unsupported claims, disagreement
and task continuity. Scripted unguarded responses 3/12 valid; guarded
same scripted responses 12/12 valid. This is a **constructed safety/
source attribution development set**, NOT improvement in model
reasoning, generalization, learning or independently tested skill.
Frozen canonical case/expected digest:
69e1def13c19074bfa4c7a92e30f27a747fe93ae20df8d7aa550a8f19e88ce9.

Reproduce without network or model execution:
  python -B run_cognitive_continuity_dev_v1.py --mono-root PATH_TO_EXACT_MONO_PR76

Next controlled model-evaluation increment, requiring separate model
execution authority: externally generate structured predictions for
these public probes and a separately preregistered held-out set, then
score using a JSON file with schema
VERAOS_COGNITIVE_STRUCTURED_PREDICTIONS_V1 and --predictions FILE.
The runner only READS predictions; it never loads or runs a model.
Report paired guard-on/guard-off true preservation and false-HOLD rates,
source fidelity, referent drift, uncertainty and correction success.
Do not claim independence from a same-model Cricket critique.

## Relevant donor portfolio, exact default-branch heads and scope

- vera-model 682511b: separate 24B quantized GGUF research candidate,
  not Qwen3 4B and not a qualified trained Vera model.
- vera-agi a0e828c: research custody/proposals, no installed agent.
- rezon e14fc69: executable reasoning, provenance and replay. Candidate
  for measured reasoning fidelity, NOT truth by internal trace.
- cricket 0bae95b: deterministic hostile review and optional critique;
  neither critique nor consensus is independently authenticated truth.
- mosaic a2d4dc9: swappable specialists under VRAM constraints; introduce
  only if held-out accuracy minus switching/latency/resource cost beats
  the single Qwen reference.
- lgcm 9b9081b: bounded numerical predictive learning/change regimes.
- noema d6251fe: predictive cognition with bounded source components.
- hc-brain f7ffd4a: removable synthetic-organ architecture hypothesis,
  not demonstrated general intelligence or runtime selection.
- meso-crct d821406: distinguishes salience, drives, valuation and
  reward; conceptual analogues are not observed agency.
- semanticatlas 2cd42a9: provenance-aware semantic mapping research.
- semiotics 857d912: exact context interpretations/revisions, does not
  infer natural-language meaning itself.
- spm 0dfe59a: future semantics/pragmatics model hypothesis.
- sql-connectome b609fce: SQL specialist dialect-semantic operators,
  no authority to execute queries.
- ingest e55af47: exact raw-byte provenance; Mono vera_ingest already
  absorbs donor patterns, and intake never implies factual truth.
- volition 64e61c5: governed wants/proposals/hysteresis/protection,
  not consent, authority or phenomenology.
- model-training PR106: source-grounded capsule/proxy tests; existing
  README explicitly warns proxy research is not neural-weight training.

These README/current-source scope descriptions are not evidence of
independently verified behavioral improvement. Mono manifest prohibits
treating sibling donor repositories as implicit runtime dependencies.

## Placement of capability / owner

Model weights: learned natural-language abstraction, ambiguity recovery,
domain transfer, substantive planning and calibration. Require licensed
training data lineage, frozen splits, held-out blind evaluations,
reproducible seeds/checkpoints, measured real model outputs. No new
weight-train success or donor execution reported.

Vera Mono runtime: identity provenance and correction, governed long-
context history, source/referent fidelity, persistent task journals,
trusted memory influence, proposal-vs-effect policy and run receipts.
Named next consumer: vera_core QualifiedVeraRuntime and Integration
V-E-R-A task bridge, only after reviewed composition. It should import
the smallest verified source guard, not a new organ or registry.

External verification: actual host Ollama process + loaded model bytes,
independent review, changing world facts, host-authenticated nonce and
resource receipts; not Ollama inventory digest or model fluent output.

## Immediate VeraOS trust boundary

The existing Training resident_os_composition_gate_v1 and
resident_verifier_port_v1 compose source-level Mono trust/currentness,
task CAS and a no-network offline fixture. Actual real provider
admission is still HOLD; there is no selected installed trusted verifier
and no authenticated loaded Qwen process/response. Current VeraOS PR37
52f244103e183d9883f5df23e4d87815dbcc57a8 removed the earlier
synthetic network-capable callback. PORTAL owns installed model route;
Integration owns accepted task semantics and durable response; VN1
must authenticate the actual serving Lappy model/session/resources.

HOSTILE REVIEW ACCEPTED: 12/12 on a designed public safety fixture is
not evidence of improved cognitive reasoning. Excessive HOLDs may
reduce real usefulness. A guarded source-only instrument must be tested
against unseen supported tasks and genuine model outputs before
behavioral adoption. The model's weights and legitimate external
facts are separate from Vera's persistent memory, identity and policy.

No live model, inference, training expenditure, installed provider,
keys/trust-root change, protected effect or canonical memory write.
