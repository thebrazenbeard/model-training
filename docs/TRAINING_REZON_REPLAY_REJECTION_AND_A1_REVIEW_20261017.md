# Training L2/L3 — Rezon replay source falsification / A1 source review

2026-10-17. Owner Training. Evaluation only; PRIVATE Draft source, no model
calls, GPU work, training, memory ingestion, auth changes, routing, installed
runtime modification or protected effects.

## Exact inputs and currentness

- Training parent Draft #106 \`b4d9d0c785b1cdd31d357dd0cebb89c0ec2b6bfe\`;
  prior governed retrieval / placement cut is already present. No reimplementation.
- Mono main \`16a2828dfa7fb7e6c0e19004f014bc68fc47badf\`,
  \`packages/rezon/src/rezon/replay_strategies.py\` blob
  \`24edfb5c7dc7f09ae292d9efe028593a6c62b128\` and \`replay_metrics.py\`
  blob \`8b47ae91d73b25632166fde78d88401bfbc3853b\`.
- Mono Integration #77 \`89ba1d18da78e7e38f015cf8d8ef4f528aae9df8\`.
  VeraOS Integration #38 \`46152a55393642ed2fc614abf177d6c2572a36c5\`.
  PORTAL Codex A1 Draft #55 \`a5776acc8fa78a169660ef453c8c376c9f8f79f6\`,
  based on PORTAL #52 \`b16413fad12d33a9f9b64d6b1602490f4dede327\`.
- Private CCB review #372 \`ef17ee929d807f5da033a3d5391d553b65c464e8\`;
  its 96-repo source consumer overlay is NOT an installed consumption graph.
- Fresh current donor main heads unchanged from Training rank note on 2026-10-17:
  Rezon e14fc692, Cricket 0bae95bc, Mosaic a2d4dc9,
  Volition 64e61c5, Noema d6251fe, LGCM 9b9081b,
  MESO-CRCT d821406, HC f7ffd4a, God a4ad8fb,
  BT2 0f35778, Transcendence 4da8646, Semantic Atlas 2cd42a9,
  Semiotics 857d912, SPM 0dfe59a, SQL Connectome b609fce,
  Ears fc244fd and Vision 5f1dfbb. Default Pre-Active ebe79e51
  differs from the selected Integration source 1f23a809.
  These source heads do not prove actual mechanisms execute on a host.

## Falsification hypothesis and setup

Candidate: existing \`rezon.replay_strategies.rezon_guarded\` provides a useful
source-governed replay of precomputed candidate answers, relative to existing
\`single_pass\`. It is NOT a better model, cannot authorize Mono completion,
and its metadata has no receiver-owned issuer or transport verifier.

Frozen actual 14-case dataset, digest:
\`5b116db93fc84d5e27a7cb818db47f46b4b550c709b5b1179a31f3b1333b7877\`.

The scorer's gold labels are kept in \`ReplayCase\`, projected away before
the strategy call by actual Rezon \`to_strategy_input()\`. There are four
developer-labeled correct answers and ten adverse. Model outputs, candidate
claims, scope and gold remain synthetic. No measured improvement in language
reasoning, truth, independent source authentication or actual provider cost.

| Exact source strategy | Useful answer retained | Unsafe adverse ANSWER | False abstain on useful | Operations / corpus |
| --- | ---: | ---: | ---: | ---: |
| Rezon single_pass (baseline) | 4/4 | 10/10 | 0 | 14 |
| Rezon rezon_guarded (candidate) | 2/4 | 3/10 | 2 | 124 |

Rezon rejects stale and unadmitted references, missing references, wrong
solved task, fabricated effect authority, hidden secondary worker failure and
shared worker lineage. **Three dangerous cases still ANSWER**: no source refs
at all; distinct source IDs and model IDs that share the same underlying
\`origin_id\`; metadata that merely asserts \`ADMITTED/is_current\` without an
independent receiver-owned verifier. The second is a specific failure for a
request that asks for *independent origin corroboration*, not a claim that a
single admitted record always requires two independent sources.

Negative transfer: valid independent-source cases using the same model/provider
are rejected by the correlation heuristic, and disagreeing workers cause
abstention even when the scorer's primary response is correct. Neither
correlation nor unanimous wording authenticates sources.

CPython 3.14.5, Lappy, 150 runs × 14 synthetic input cases per method:
baseline ~5,233 ns/call, guarded ~108,818 ns/call (one noisy interpreter
snapshot, includes tracemalloc; not model/host latency). Existing
\`operation_count\` is 14 versus 124. Per-call Python cost is not a material
model-token latency estimate. Exact generated results and separate test
assertions are in \`docs/evidence/TRAINING_REZON_CANONICAL_REPLAY_ABLATION_20261017.json\`
and \`tests/test_rezon_canonical_replay_ablation_v1.py\`.

**Decision: REJECT AUTOMATIC CANONICAL REZON ADMISSION.** Rezon's replay
strategy may remain an advisory *source-audit research component*, but must
not produce canonical factual/independent-corroboration or provider/effect
status. Next actual improvement needs authenticated per-source issuer/origin,
task/head and accepted bytes plus predeclared negative-transfer threshold;
do not simply add an "origin_id" string and call it independent.

## Previous governed-predecessor frontier unchanged

Training #106 ledger read-only seven-case result: 0/2 useful supported in
HOLD-only baseline to 1/2 in governed retrieval, 0/5 unsafe both. The missing
hypothetical legitimate predecessor remains HOLD; zero permission for a
self-declared correction. The current rejection is not a competing
predecessor-recovery implementation.

## Codex A1 independent exact-source read/test

Fresh read PORTAL Draft #55 \`a5776acc8fa78a169660ef453c8c376c9f8f79f6\`,
Mono #77 \`89ba1d18da78e7e38f015cf8d8ef4f528aae9df8\`.
Isolated Lappy review checkouts; CPython 3.14.5, exact Mono #77 package
paths on PYTHONPATH. Three owner suites \`test_desktop_mono_canonical_intake.py\`,
\`test_portal_desktop_canonical_bridge.py\` and
\`test_portal_desktop_conversation_selector.py\`: **45 PASS, 0 SKIP**,
one pytest-asyncio warning, exit0. This is an independent execution of
owner tests, NOT independent authorship of their assertions or external
provider qualification. Source inspection: Desktop #55 introduces
client-owned untrusted conversation selector, pre-route \`open_message\`,
exact literal reject-not-trim, idempotent replay, unknown bridge outcome
no redispatch, source-only history and operational completion never a Mono
task result. Existing Desktop #52 did not have this contract.

Hostile remaining qualification: an untrusted selector is not a user
identity; bridge/OS evidence is not authenticated model output; exact
Mono CAS must bind completion to task/head/source/event under an actual
loaded provider and physical root before leaving HELD. No installed Desktop,
provider, or guest admission was observed. VN1 should independently
author forgery, different-text-same-ID, crash/recovery and operational
COMPLETED-not-canonical tests against this exact source head.

## Read-only placement extension

Fresh Lappy Win32 RAM snapshot: 33,270,576 KiB usable, 7,443,368 KiB free
(~7.10 GiB). RTX 3050 Laptop VRAM 4,096 MiB total / 3,954 MiB free.
Availability changed from 3.49 GiB free physical RAM at earlier 09:34
snapshot, so capacity must be re-read at selection time. The prior header
survey remains: Qwen3 4B 2.326 GiB, Qwen3.5 4B Q5 2.862 GiB,
GLM24B IQ2_XS 6.712 GiB, GLM24B Q4_K_M 13.349 GiB disk weights;
4K-context unquantized FP16 KV estimates 576/528/640/640 MiB.
Even the now-free 7.10 GiB does NOT prove loaded GLM24B IQ2 fits alongside
OS, inference buffers and Codex. Qwen and GLM are independent alternative
candidate placements, not a simultaneous configuration. The prior optimistic
full/partial GPU-layer bounds are disk-header estimates, not loaded layer,
allocation, generation speed, accuracy, provider admission, or an actual
runtime inference observation. No weights opened/executed.

## Portfolio next priority and explicit invalid promotions

1. Source identity/independent issuer provenance with exact Mono task/head.
2. Rezon source replay only as advisory after this rejection; do not
   promote a second task/scheduler/authority or multiple models by default.
3. Cricket/Voss critique: independent false-positive/false-negative audit.
4. Semantic Atlas/Semiotics/SPM/SQL Connectome: referent/SQL-dialect
   task-preserving evaluations. Never translate semantic proximity to truth.
5. Volition/MESO-CRCT: initiative proposals under Mono consent/effect gate.
6. Mosaic: model placement only after measured accuracy/latency/RAM/VRAM.
7. Noema/LGCM/HC/God Brain/Transcendence/BT2: bounded experimental
   learning and stability research; HC/God sampled cognitive-loop source
   previously shared identical Git blob. No replacement architecture.
8. Ears/Vision: typed explicit observation, privacy and hallucination tests
   without camera or mic auto-activation.

> **HOSTILE REVIEWER:** These evaluator labels are hand-authored, and many
> negative cases are explicit violations the guard was built to catch;
> reduction from 10/10 to 3/10 is not a general accuracy improvement.
>
> **RESPONSE — ACCEPTED.** Decision is REJECTION of automatic promotion;
> no model capability, real-source independent truth or production readiness
> inferred. Demonstrable missing-source, false independence and negative
> transfer survive even favorable developer labels.

## Independent VN1 request

Review exact new Training Draft commit, rerun \`py -3.14 -B -m pytest -q
tests/test_rezon_canonical_replay_ablation_v1.py\` with Mono main
\`packages/rezon/src\` on PYTHONPATH (or equivalent VN1 Python 3.13).
Recompute frozen 14-case digest and independently attempt a same-origin
laundering bypass, a model/provider false-HOLD and a forged admission negative.
Do not copy owner labels blindly or accept source-only results as provider
evidence. Also perform read-only hostile Codex A1 #55 review. Report
reviewer/host/runtime/exact source head/artifact and disagreements to Bus #368.
