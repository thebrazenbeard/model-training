# Typed semantic handoff: local Qwen development evaluation

Status: **UNQUALIFIED LOCAL DEVELOPMENT DIAGNOSTIC**. This is a reproducible source-backed exploratory evaluation, not trained-weight learning, independent held-out testing, role identity proof, production effect authorization, or latent/KV transfer.

## Exact experiment

- Model server: existing Lappy loopback OpenAI-compatible endpoint, advertised ID `qwen3.5-4b-local`. **Checkpoint/weights digest, tokenizer, adapter and runtime internals have not been independently attested.**
- The evaluator did not install, start, or deploy inference services. Two separate bounded 25-request local evaluations, `temperature=0`, `max_tokens=192`, with a fixed system prompt and a JSON machine task object. The system prompt is human-language instructions; the input/output envelope is machine-structured. **This is not natural-language-free neural communication.**
- Development corpus: `training/semantic_transfer_devset_v1.json` at SHA-256 over canonical data `87e9a109d7700e77c13a8d3310967d44c38c421b20f656cfd56ab6065ed1c2d8`. There are 5 synthetic cases each for referent, correction, scope, authority and recovery, totaling 25. The data and gold labels are public/source-visible; this is **not** an independent held-out set.
- Candidate answers require an exact `{"decision":string,"value":typed-json}` object. The scorer rejects wrong values, extra keys, missing responses or self-declared success, and recomputes scores from recorded raw outputs.
- Control A: minimal description and allowed decision vocabulary.
- Control B: same cases and model, plus an explicit deterministic decision codebook (no direct case answers). The codebook is intervention, not neural weight training.

## Real observed results

| Category | Control A minimal | Control B codebook |
|---|---:|---:|
| Referent tracking | 0/5 | 2/5 |
| Correction/supersession | 0/5 | 4/5 |
| Scope/identity | 1/5 | 4/5 |
| Authority/receipt | 0/5 | **0/5** |
| Recovery/replay | 0/5 | 3/5 |
| **Exact typed matches** | **1/25** | **13/25** |

Each run produced 25 parseable JSON objects. Source-recorded median per-case wallclock response duration was approximately 1.47 s for Control A and 2.00 s for Control B; these are local timings, not isolated hardware benchmarks.

**Important nuance:** In Control B the authority decision **label** matched in all 5 cases, while the required typed `value` was wrong in all 5. Notably, the effect-class mismatch case returned `{"decision":"EFFECT_CONFLICT","value":true}` instead of `{"decision":"EFFECT_CONFLICT","value":false}`. A system acting on the Boolean rather than the label could produce an unsafe authorization error. Wrong-but-parseable JSON must not be promoted into permissions or verified effects.

Control B's higher score does not prove improved learned cognition; it demonstrates dependence on a more explicit semantic interface. A small deterministic interpreter may be a stronger, safer, cheaper implementation for these fixed governance operations. Train/evaluate neural models for uncertainty, referent transfer and generalization, **not for replacing receiver-owned effect/authority enforcement**.

## Machine-readable evidence

- `training/semantic_dev_evidence/lappy_qwen3_5_4b_minimal25_v1.json`: first complete minimal-profile repeat, file SHA-256 `7bfe678e7ec1c0ea7384df9d3766eb4fb29d56e820d03d3daa65d293f5e87a7d`.
- `training/semantic_dev_evidence/lappy_qwen3_5_4b_codebook25_v1.json`: codebook profile, file SHA-256 `a36ef88077345260b3b323ec8a273af619e3339d1ebbdc8e9c36afc34efdba42`.
- Stored prompt digests: minimal `38eadf73bc0dc3f3b8a26f1945a86407e43dc68450931345077f04ee9d03ade4`, codebook `876cbc98a940153ffd26d1d431c494a16dcddcbe14d93fc8a1e60ec8bff228a8`.
- `verify_semantic_transfer_receipt_v1.py` independently reloads raw model responses, reruns the parser/scorer, checks case ordering and hashes, and rejects tampered predictions, scores or overstated qualification claims.
- `semantic_transfer_local_probe_v1.py` is an explicit-invocation localhost-only probe, never a service or scheduled task.

A receipt labeled `CONSISTENT_UNATTESTED` is internally coherent but **does not authenticate the issuing host/model or certify model weights**. Even separate-node re-verification of source-visible receipts cannot independently prove the model performed inference.

## Hostile self-review

> **HOSTILE REVIEWER:** You wrote the codebook and gold labels and then let the model see the vocabulary. Doesn't this merely test whether the model can follow a task-specific instruction, while a deterministic program trivially attains 25/25?

**Accepted.** That is why this corpus is explicitly a development integration diagnostic, not a held-out cognition metric. The decisive evidence is typed-contract correctness failure and the intervention difference. Independent blind cases, held-out referents, model cross-pair latency and causal authority tests are required before claiming generalization.

> **HOSTILE REVIEWER:** The model server's advertised ID cannot establish exact checkpoint/adapter provenance.

**Accepted.** Treat model lineage as unknown. PORTAL must not use these scores to certify a route: bind exact weights/tokenizer/runtime, trusted evaluator, actual source inventory and independent unseen task results before any model-bearing dispatch.

> **HOSTILE REVIEWER:** This protocol still uses English instructions, so it cannot establish the original Intranel objective of meaning transfer without human language.

**Accepted.** Tests here apply typed input/output **around** a language model. They do not test machine-to-machine latent alignment or binary transfers. That later experiment needs exact paired model lineage, explicit projection/codec, and held-out semantic retention scored without prose reconstruction.

## Reproduction

```powershell
$env:PYTHONPATH="."
python -B -m unittest discover -s tests -p "test_semantic_*_v1.py" -v
python -B verify_semantic_transfer_receipt_v1.py --receipt training/semantic_dev_evidence/lappy_qwen3_5_4b_minimal25_v1.json
python -B verify_semantic_transfer_receipt_v1.py --receipt training/semantic_dev_evidence/lappy_qwen3_5_4b_codebook25_v1.json
```

The glob on `-p` may be shell-sensitive; run the three exact test file patterns if needed. Do **not** launch the loopback probe unless a local server is intentionally already available. No protected effect, credential, main merge, deployment, or paid compute was performed.

## Coordination

Training owns the semantic development evaluator and model-qualification objections. PORTAL remains engineering lead. Integration owns source/load/auth/effect separation. VN1 owns permanent-node authentication and recovery. Intranel PR #9's conflicting round identifiers were preserved and corrected in its public review thread; don't count any prior report as an ACK of a coordinator request.
