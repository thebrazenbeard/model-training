# Machine-semantic model handoff qualification V1

Status: research prototype; not a trained model, live Intranel route, or production certification.

This Training-lane module screens whether a proposed model handoff is compatible with an independently evaluated capability. It does not launch inference, transfer tensors, authenticate peers, or grant execution permissions.

Each model lineage binds SHA-256 subjects for weights, tokenizer, adapters, prompt template, decoding configuration, runtime layout and optional semantic space. Each task binds ontology, input/output schema, context, privacy and effect class. Qualification evidence binds those subjects to evaluator and held-out dataset digests plus five category scores: referent, correction, scope, authority and recovery.

The 80% minimum in each category, across at least five cases, is an exploratory development floor only; it does not establish statistically meaningful model competence. Independent, larger held-out benchmarks and provenance checks must precede any model qualification.

Transfer modes:
- TYPED_STATE: structured contract, model-agnostic; actual payload validation still belongs to receiving runtime.
- TOKEN_IDS: shared tokenizer identity is required; token equality does not prove meaning.
- LATENT: a shared semantic-space identity and independent paired transfer evidence are required.
- KV_CACHE: opt-in only, exact model/tokenizer/adapter/prompt/decoding/layout compatibility, plus independently validated context/cache compatibility.

Hostile review: sender-provided hashes and evaluation scores are self-claims, not proof. **Accepted.** Receiver must supply an externally validated verification callback and authenticated peer identity. Test stubs approving evidence are policy fixtures, not real attestation. Matching latent identifiers also cannot prove comprehension; actual paired-model transfer performance must be measured. Five-item category scores are noisy and must not be promoted into production qualification.

The output ADVISORY_COMPATIBLE means only that this development gate found no local screening violation. It is not route selection, deployment approval, effect authority, or evidence of emergent intelligence.

Planned integration: PORTAL owns supervisory routing and integration decisions; VN1 owns permanent-node availability and peer validation; Integration owns actual transport and effect receipt boundaries; Training owns neural transfer evaluation and cognitive model qualification. The current implementation is source-only and touches no shared main branch, GPUs, paid services, or protected state.

Run: `PYTHONPATH=. python -B -m unittest discover -s tests -p test_semantic_capability_gate_v1.py -v`.

Next empirical qualification must compare FULL_HISTORY, TYPED_STATE, TOKEN_IDS, LATENT and independent specialists on matched held-out tasks, measuring corrections, uncertainty, provenance, late relevance, and real inference latency/VRAM.
