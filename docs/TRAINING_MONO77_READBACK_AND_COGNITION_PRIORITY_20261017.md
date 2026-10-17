# Training L2/L3 ? Mono #77 / VeraOS #38 readback alignment

Date 2026-10-17. Draft Training #106 SOURCE-only candidate. No model execution,
weight training, installation, trust key provisioning or canonical memory writing.

## Exact source and current boundaries

PORTAL full-chat checkpoint 543370fa7a57f751842c6d72b4484814c5509063,
00_READ_FIRST, 01_REPOSITORY_BASELINE_AND_SOURCE, and 03_FOUR_LANE.
Latest Bus #368, Mono main 16a2828dfa7fb7e6c0e19004f014bc68fc47badf.
Mono #77 31155c9e6f00abf5869da21f886f90717f22b6fb, local_conversation
blob 477beb70c33020ad8506aea624618c5776c9e724, based on #76.
Integration-owned VeraOS #38 85d6cd39fe32e1214b213a63bdac0b1f54e664ac,
resident blob 6c34af80db1acd9e38206cfd108c641f2da7f058.
PORTAL #37 14c57dd89cae905139f5329c0789dbb8f83f6a8d contains
a newer event-DONE terminal check than the #38 base. Do not edit owner PRs.

## Actual pipeline and evidence/authority placement

UI user text, conversation/message locator (NOT user authentication)
 -> Mono #77 open_message: exact task_id, literal source digest, task CAS
 -> Mono governed context/retrieved input and corrected history (unverified
    user input is NEVER autobiographical memory)
 -> optional Rezon reasoning/planning/critique (requires measured ablation)
 -> PORTAL advisory model placement (not admission)
 -> OS-owned Mono PROVIDER currentness + external Lappy loaded model/process,
    exact model/endpoint, host generation/session, fresh sample, one-use nonce
 -> authorized text-only PreActive execution (NOT present in current guest)
 -> PreActive task.requested event, actual run/assistant message, ONE
    VERAOS_COGNITION_ROUTE_SELECTED and exact final-text hash
 -> mono task-scoped CAS/checkpoint and conservative cold recovery
 -> separate vera_memory provenance/influence qualification if authorized.
No source stage promotes itself to effect authority, provider qualification,
model autonomy or trained weights.

## RED -> GREEN boundary correction

Training resident_model_turn_readback_v1 previously checked PreActive dedup
as veraos-text:<client_request_id>, but the current Mono #77 source uses
veraos-text:<canonical_task_id>, to prevent same message ID collisions in
different conversations. Original Training accepted events with status
CLAIMED despite Mono #77 requiring DONE. It also accepted any matching
route record from multiple VERAOS_COGNITION_ROUTE_SELECTED journal rows.

New test_mono77_readback_alignment_v1.py reproduced four separate source
failures before changes: correct canonical task dedup rejected; legacy
request-key dedup incorrectly accepted; CLAIMED event incorrectly accepted;
duplicate matching route selections incorrectly accepted.
Production-bound alignment requires canonical task-id dedup, DONE only,
and exactly ONE route journal record. Preexisting readback tests and
pinned genuine PreActive Store fixture updated to current task-ID semantics.
This positive result is still SYNTHETIC_STORE_LINK_VERIFIED and explicitly
real_model_consumption_verified=false/canonical_task_completed=false/
effect_authority=false. It is NOT proof of serving Qwen.

Critical hostile caveat: Mono #77 runs its own source recovery; this
Training utility is a reviewable adapter, not installed in #38. Its
synthetic transport evidence remains caller-constructed, and it has
no OS-installed trusted loaded-model signature or durable nonce.
DO NOT route a network call based on this adapter's positive output.
The history/source readback cannot prove real loaded model bytes or
independently authenticated response causality.

## Repository donor evaluation / measured priority

Priority 0: already selected Mono canonical tasks and PreActive
source receipts, plus vera_identity and vera_memory correction/intake.
Use and verify them before adding more models or repos. #38 host tests
show actual source consumption of task/open/history and bounded offline
PreActive results, but exact selected Buildroot pins are still historic
Mono/PORTAL revisions; no current normal QEMU package qualification.

Priority 1: Rezon e14fc6921d4f56bb6f6dc306e72b33ef25fd6777,
src/rezon/replay.py blob a8b8cf5abb2c0110fdd361e70d03797a73152c1b.
ReplaySource/ReplayCase/StrategyInput and typed dispositions are concrete
evaluation and provenance mechanisms; prefer a paired ablation with
single-model Qwen plus Mono source guard over installing a new planner.

Priority 2: Cricket 0bae95bc45db55d78001ee9c492681834191ec0b,
src/cricket/reviewer.py blob 6f7f89923130e38322ba2110e16625000e6e080d.
Rule/semantic/optional critic reviewers can catch claims, but advisory
critique is not independent truth; measure false critique and missed
error on frozen evidence, do NOT confer admission via dissent.

Priority 3: Volition 64e61c546b89f5dc0eb78d382637ccccef42aa31,
src/volition/engine.py blob d271183ddfcaf59438960130996c33aac0ff9e82.
The source has typed proposal/goal/arbitration policy; #38 composes
a VolitionEngine but runtime import is not evidence of endogenous
initiative/effect authority. Defer live proactive routing.

Priority 4: Mosaic a2d4dc9812156f41e370e062dad5e2b45a0dab1e,
runtime/residency.py blob e58282f983fe6d596a2868c691a5bc486ab93e81.
Concrete VRAM envelope/swap evaluator; require held-out accuracy,
swap latency, actual available VRAM, memory and energy measurement
versus existing single Qwen, before specialist placement.

Priority 5: LGCM 9b9081b404b0f7ade73d83768758873bb2b988fd,
src/lgcm/evaluation.py blob ec10d3f663ea656d44095ad848abfa54fe5589b4.
Genuine numeric prequential evaluation code uses numpy and
action-conditioned predictive learner, NOT transferred language cognition.

Priority 6: Noema d6251fecb083be1a6bf1d6a300ef0a4ffcac6186,
src/noema/svf0_evaluator.py blob c237800c92e4c86712555d8d714cebf32afc2c6c.
Frozen-seed persistent predictive research with evaluator/selection gates,
not authorized live learning, trained Vera weights or resident inference.

Separate unqualified research: Vera Model
682511b6b5945c874cdc550a9705b5371a151581
runtime/local_model.py blob 0f80bff506b61172c007d71f81e18a889b3b3909
is a 7,207,034,784-byte pinned IQ2_XS GLM24B GGUF candidate with local
llama.cpp test client, not the Qwen3 4B resident, not a trained Vera
model, not proof of active loaded/served GLM. Stored Hugging Face donors
unchanged: Open-Reasoner research; GPT-X3 no custom-code execution;
uncensored-stage1-hacker license/provenance HOLD.

## Next falsifiable blind work

Prior frozen Training 12-case synthetic source-attribution benchmark:
3/12 scripted baseline, 12/12 deterministic guard; these are NOT
inference-quality gains. Separate five-case counterexample shows guard
incorrectly HOLDS two agreeing sources (false HOLD against a supported
claim). Before deploying that guard, preregister fresh supported AND
unsupported unseen examples. Freeze Qwen real outputs under separately
approved inference authority; paired source guard ON/OFF with identical
outputs. Measure answer accuracy, false HOLD, incorrect attribution,
correction recovery, ambiguity, task swapping, run retry/cold recovery,
latency and peak memory. Test privacy/memory admission independently.
Never train or modify weights under this source-only work.

Remaining VeraOS blockers: #38 Buildroot selects obsolete Mono/PORTAL
heads, #37 DONE fix needs owner-coordinated convergence, deterministic
SQLite close, bounded VirtIO parser #32, QEMU resource headroom;
no currently authenticated Lappy serving-process model/nonce/provenance.
These belong to Integration/PORTAL/VN1/OS owner, not Training branch.
