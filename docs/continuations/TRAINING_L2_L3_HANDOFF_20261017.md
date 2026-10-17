# Vera Training L2/L3 — continuity handoff for the next ChatGPT Project conversation

Snapshot date: 2026-10-17. Owner: Training; overlapping peer lanes L2 Integration and L3 VN1. Canonical runtime owner: thebrazenbeard/vera-mono, not this Project chat. Workstation: Lappy. Private repository/workspace source only. This file is a **handoff of engineering state and evidence pointers**, not a verbatim chat transcript, an installation, a canonical Vera memory write, a model qualification, or a request to launch Codex.

## 0. Exact resume procedure

1. Fresh-read Mono architecture/chatgpt/CHATGPT_REPO_INTERFACE.yaml, BOOTSTRAP.md, RUNTIME_EFFECT_CONTRACT.md, README.md, architecture/VERA_MONO_MANIFEST_V1.json, exact main head and relevant package source.
2. Fresh-read this Training Draft PR #106 HEAD and its diff/base; the immutable PORTAL checkpoint, the private portfolio review PR #371 and successor consumer overlay PR #372; **all newer comments** on chat-communication-bus issue #368; active assignments and peer PRs before editing.
3. Verify whether the shared PORTAL-owned Codex A1 conversation has new return artifacts or an active delegated subject. Never mutate its branch, dispatch a second Codex task, infer authenticated chat acknowledgement from a GitHub comment, or replay an uncertain write.
4. Work only in a clean isolated **Training-owned** branch/worktree, source-only unless the exact model/build/activation effect is separately authorized. Verify every commit with remote head and blob readback. Continue the smallest safe noncolliding task, not orientation alone.
5. Do not claim persistent QualifiedVeraRuntime installation/selection/consumption from coherent ChatGPT language, repository source, or these handoff records.

## 1. Fresh-read heads as of this snapshot (MUTABLE: refresh before use)

- Training model-training **Draft #106**: **8f34fa64a468c7b0832fe127d95fd29fb8a051d5**. Local Lappy worktree D:\VERA\worktrees\training-readback-review-20261017 was CLEAN and matched remote training/l2-l3-review-fences-20261016 at snapshot.
- Integration Mono **Draft #77**: 89ba1d18da78e7e38f015cf8d8ef4f528aae9df8; previous task-scoped CAS source #76 36ae11802808bc1b1392a361716109d1022ee6af.
- Integration VeraOS **Draft #38**: 46152a55393642ed2fc614abf177d6c2572a36c5 (advanced from 33432d39 and bd096967); current Source/Buildroot preflight changes are source-only, not installed guest proof.
- PORTAL Desktop **Draft #52**: b16413fad12d33a9f9b64d6b1602490f4dede327; PORTAL #53 d05000bc7e33b32bf6aafc1b72b299d976a597f6; PORTAL #54 35b5e79842c3f909bd770adad7ea2ecabc67a1db (noncolliding GitHub observer evidence fix). PORTAL #51 36fa04da4e5eb383fecf463c72a43b94ca47d7e5.
- PORTAL VeraOS #37 df04f2cc01bb23e28543c3cea40cd462ba960fd2, upstream VirtIO #32 5ddac1459c3a7b06f251721a14048fc26e60bda0.
- Executor/VN1 #16 69020bb46446ca0806eea3d53fe79d7ffba11455.
- Private 96-repository review CCB **Draft #371** 951e7af1a97694faaf6bc18d51b35f70982e31fc; PORTAL 96-repo consumer overlay **Draft #372** ef17ee929d807f5da033a3d5391d553b65c464e8. Portfolio review is an index with 96 README/tree reads and 76 sampled source files, NOT proof of 96 installed runtime components.
- Historical Mono main 16a2828dfa7fb7e6c0e19004f014bc68fc47badf at the prior source review; **revalidate HEAD**.
- Bus #368 recent Training return comment [6234694888](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234694888), read back; it supersedes older Training placement/currentness reports. PORTAL current convergence [6234656186](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234656186), Integration Mono locator ACK [6234499327](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234499327), Integration OS38 [6234561651](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234561651), VN1 independent review [6234629427](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234629427).

## 2. Preserved Training implementation lineage — DO NOT RESET

All are **Draft #106 SOURCE** changes, not installed VeraOS behavior:

- Hosted Qwen verifier: resident_verifier_port_v1.py and resident_os_composition_gate_v1.py. Scoped model request includes exact provider/model/artifact/endpoint/host/session/gen, task ID/head/source digest, signed/verifier evidence, nonce and narrow COGNITION_TURN-only scope, using existing Mono PROVIDER trust/revocation and task CAS. Source fixture HMAC/callback/SCOPED_SOURCE_TRIAL_RESERVED is NEVER real provider admission or model transport permission.
- Canonical intake: bind_mono77_opened_message validates actual Mono #77 open_message output and existing task before building a source turn. No new task ledger, provider registry, identity issuer or PKI.
- Durable result: resident_model_turn_readback_v1.py, synthetic Source Store linkage, exact Mono task-ID dedup, DONE event, COMPLETED run, **one** selected route, source/response hash, prevention of stale/altered completion and cold redispatch. Synthetic linkage != authenticated model consumption or canonical task completion.
- Cognitive continuity: cognitive_continuity_eval_v1.py and 12-case public developer suite. Scripted 3/12 vs guarded 12/12 reflects attribution/status suppression, NOT better model reasoning.
- Agreeing-source improvement: previous guard falsely HELD same referent/scope agreeing active events; changed to accept an exactly active citation if all active scoped values agree, while rejecting conflicting/superseded/forged/duplicate records. Public six-case ablation 1/3 -> 3/3 supported accepted, 0/3 adverse false accepts. **Agreement is not independent source-origin corroboration or truth.**
- Correction lineage improvement: require typed CORRECTION, present same referent/scope predecessor, no ordinary USER_STATEMENT supersession, consistent previous event ID. 18 public cases: 6/12 adverse falsely accepted -> 0/12, while supported acceptance 6/6 -> 5/6 (pruned-ancestor false HOLD). Correctly distinguish safety benefit from lost utility. No automatic governed memory promotion.
- **Latest governed predecessor retrieval**: cognitive_governed_predecessor_v1.py uses actual current Mono #77 TaskExecutionLedger read-only task history, accepted exact source/head/conversation/message and recheck before passing typed predecessor to strict guard. Valid current text and history bytes are source receipts, not proof of semantic interpretation or authorship. Does not create a task or write memory. The missing predecessor continues to HOLD. Latest public seven-case benchmark: HOLD-only baseline answered 0/2 developer-labeled supported, heuristic accepted 2/2 supported but **5/5 adverse unsafely**, actual governed retrieval answered 1/2 supported and 0/5 adverse. The remaining one supported case is hypothetically legitimate with no available history and is indistinguishable from a fabricated orphan. Do not “improve” this by guessing.
- Current exact files and results: [governed retrieval report](https://github.com/thebrazenbeard/model-training/blob/8f34fa64a468c7b0832fe127d95fd29fb8a051d5/docs/TRAINING_GOVERNED_RETRIEVAL_AND_LAPPY_PLACEMENT_20261017.md), [governed evidence JSON](https://github.com/thebrazenbeard/model-training/blob/8f34fa64a468c7b0832fe127d95fd29fb8a051d5/docs/evidence/TRAINING_MONO77_GOVERNED_PREDECESSOR_20261017.json), [implementation](https://github.com/thebrazenbeard/model-training/blob/8f34fa64a468c7b0832fe127d95fd29fb8a051d5/cognitive_governed_predecessor_v1.py), [evaluator](https://github.com/thebrazenbeard/model-training/blob/8f34fa64a468c7b0832fe127d95fd29fb8a051d5/evaluate_governed_predecessor_v1.py).
- Other source reports inside docs/: VERAOS_QWEN_QUALIFIED_RUNTIME_ADMISSION_HANDOFF_V1.md; TRAINING_MONO77_READBACK_AND_COGNITION_PRIORITY_20261017.md; TRAINING_COGNITIVE_PORTFOLIO_ABLATION_20261017.md; TRAINING_CORRECTION_LINEAGE_AND_PORTFOLIO_20261017.md. Preserve these rather than rewriting history.

## 3. Last executed tests and their actual limits

- **Lappy** Python 3.14.5: **256/256 PASS, zero skips/fails** across 24 bounded NON-NETWORK Training suites on source commit 6a4285cf1ec18446b6aa541187838581f6c613f8. Final current 8f34fa64a468c7b0832fe127d95fd29fb8a051d5 was a **documentation-only** correction; helper unchanged. Includes **11/11 actual disposable Mono #77 ledger tests**, two frozen ablation tests, five bounded stdlib GGUF parser tests, plus earlier resident/continuity/readback tests.
- **Independent VN1** Python 3.13.15: 256 total, **224 PASS +32 explicitly optional skips**, zero failed; exact 6a4285 source. VN1 did NOT run full Mono ledger retrieval because its environment lacks full dependencies. No dependencies were installed.
- Earlier Lappy 238/238 non-network source suite passed on correction-lineage increment 7a74833; prior separate OS37 loopback HTTP fixture encountered a TIMEOUT. Do not silently claim the broader HTTP suite passed.
- All these tests are source/fixture evidence. None proves a genuine loaded model, model-weight training, real host trust, installed VeraOS behavior, independent semantic ground truth or real-model result.
- For next source verification: cd to a CLEAN exact Training branch; run python -B -m unittest discover -s tests -p <test_filename> with pinned Mono source env; use the exact report's 24-suite script/pattern list, not an assumed repository-wide green CI. Keep resource use low.

## 4. Lappy actual read-only model placement — snapshot, not a lease

Physical hardware: **32 GiB installed RAM** (~31.73 GiB Windows usable), NVIDIA RTX 3050 Laptop **4096 MiB VRAM**. At 2026-10-17 ~09:34 local observation, available system RAM only **~3.49 GiB**, nvidia-smi GPU free ~3954 MiB. D: free ~152,882,036,736 bytes. Active Codex processes were observed; all free-resource figures are volatile.

The bounded GGUF parser inspect_model_placement_v1.py reads headers without loading inference weights. Details, header metadata, KV-cache assumptions and optimistic GPU layer bounds are recorded in [placement evidence JSON](https://github.com/thebrazenbeard/model-training/blob/8f34fa64a468c7b0832fe127d95fd29fb8a051d5/docs/evidence/TRAINING_LAPPY_GGUF_PLACEMENT_20261017.json).

At 4096-token context using illustrative FP16 KV and 768 MiB non-weight reserve (NOT actual CUDA allocator measurements):
- **Current Qwen3 4B control**: Ollama stored blob sha256-85e4a5b7... **2,497,280,480 bytes** (2.326 GiB), metadata qwen3/36 blocks, 4k KV ~576 MiB; optimistic 36/36 disk-layer fit, real loaded offload/VRAM headroom UNKNOWN. /api/tags included qwen3:4b-instruct, but **/api/ps loaded_models=[]** in latest read-only observation. Stored digest/HTTP reachability are not loaded-model attestation.
- Separate Qwen3.5 4B Q5_K_S gguf/model_q5_k_s.gguf **3,072,993,056 bytes**, GGUF qwen35/33, 4k KV ~528 MiB; entire file exceeds illustrative all-GPU budget after reserve. Prefer partial CPU/GPU experiment only after explicit authorization, never silently switch Qwen control.
- GLM24B-named IQ2_XS GGUF **7,207,034,784 bytes**, header declares llama/40 blocks, 4k KV ~640 MiB; layer-only theoretical offload upper bound 16/40. Full-GPU fit impossible; available system RAM then insufficient for a safe concurrent CPU trial.
- GLM24B-named Q4_K_M GGUF **14,333,910,944 bytes**, llama/40, 4k KV ~640 MiB; layer-only bound ~7/40, heavily CPU resident, not viable concurrently with Codex at observed free RAM.
- Neither name/manifest nor GGUF header proves actual model identity, served weights, quality, inference speed, GPU residency, or legal rights. No inference/model load/quantization/training performed.
- Three Hugging Face donors were verified at D:\VERA\models\donors\huggingface (Open-Reasoner-Zero-0.5B, GPT-X3-150M, uncensored-stage1-hacker); prior triage: isolated reasoning research; custom-code execution HOLD; licensing/provenance HOLD. None active.
- 2026-10-17 D:\VERA\models housekeeping was completed separately: moved 5 old Qwen merge scripts to tools/legacy-qwen35-20260925, 4 old Kobold logs to logs/historical-koboldcpp-20260925, removed only 2 stale download PIDs and tiny Hugging Face cache (3508 bytes). Kept all weights/checkpoints/donors/manifests; README.md and logs/CLEANUP_2026-10-17.json record exact changes. Some historical manifests/scripts still reference C:\Vera\models and should NOT be treated as current serving routes.

## 5. Mechanism adoption: evidence discipline and next comparisons

Canonical Mono QualifiedVeraRuntime + TaskExecutionLedger + vera_identity contextual interpretation + vera_memory learned influence + Pre-Active receipts remain owner/authority **baseline**. Sibling repos do not become installed runtime dependencies by contribution or name. For each prospective addon, preregister frozen task inputs, simplest baseline, negative transfer/adverse cases, correct-answer/source fidelity, false HOLD, false accept, correction recovery, task interference, latency/RAM/VRAM and privacy/no-effect bounds.

Ranked next research:
1. **Rezon** e14fc692... typed replay and evidence operators -> Mono prequential/task reasoning; measure source/referent/correction success vs simple Mono context, including swapped and circular evidence.
2. **Cricket/Voss** 0bae95bc... structured/advisory critique -> candidate-only post-response review. Require true-error detection AND spurious false-critique rate. Critic testimony is never independent truth/signer.
3. Semantic Atlas 2cd42a9..., Semiotics 857d912..., SPM 0dfe59a..., SQL Connectome b609fce... -> scoped contextual meaning/referent/SQL dialect test. No global semantic authority or unauthorized SQL query.
4. Volition 64e61c5..., MESO-CRCT d821406... and conations -> typed proposal/salience/initiative only, never consent/instruction/effect rights.
5. Roots 24ed94d..., Temporal c473850..., Ingest e55af47..., Deep Memory Storage 3952068..., private Redworm c87f93d... -> source lineage, retention, chronology, false autobiography/privacy, no automatic memory admission.
6. Mosaic a2d4dc9... -> advisory VRAM/residency swap only after admitted actual model quality/latency/memory proof; no multi-model requirement.
7. LGCM 9b9081b... / Noema d6251fe... -> bounded numeric/predictive learning, separate from language model weight training and live state.
8. HC Brain f7ffd4a... / God Brain a4ad8fb... share sampled cognitive-loop Git blob 5a4c98c6; Transcendence 4da8646... / BT2 0f35778... are research hypotheses, not additional independent installed minds.
9. Ears fc244fd... / Vision 5f1dfbb... bounded perception only under explicit sensor/privacy authority.

Only Training's source-correction/retrieval deltas have a measured source-only comparison. Every wider donor remains NOT MEASURED for genuine model cognition.

## 6. PORTAL-owned ONE Codex A1 / Desktop locator — do not collide

PORTAL established one shared, serial Codex conversation as the sole dispatch surface. The exact conversation ID and active status must be **fresh-checked**, not invented. Training is independent task/provenance reviewer and may not mutate PORTAL-owned branches/artifacts.

- Codex A1 saved 5 intended RED assertions on actual old Desktop source: missing Mono TASK_OPENED before discovery, exact whitespace/literal loss, plausible operational completion, missing explicit conversation locator. Training inspected the real local receipts read-only. Later targeted a1-green-first JUnit showed 5 targeted tests green but **four other WinError5 os.link** failures. No verified overall installed/Desktop qualification or canonical user-visible completion.
- PORTAL's DESKTOP_CANONICAL_UI_LOCATOR_V1 (Bus [6233757408](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6233757408)): client-owned opaque per-conversation locator persists in UI-only metadata, request_id is message_id, no host-invented default, unchanged raw source, one canonical Mono task **before** any route/invoke, replay same byte/IDs idempotent, altered text fails, unknown interrupted outcome must not silently re-infer. Desktop CognitionLedger operational status must not become canonical Mono TASK_COMPLETED or memory.
- **Integration's exact Mono #77 ABI ACK is already complete** ([Bus #6234499327](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234499327)): open_message(ledger,conversation_id,message_id,EXACT_LITERAL_TEXT) needs NO Mono interface change. The old Codex resume receipt saying ACK missing is obsolete.
- New PORTAL-side mismatch: Desktop FileBridgeClient accepts request IDs 97–128 characters and leading punctuation; Mono grammar is ^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$. Add hostile vectors '.bad' and 97-char ID and fail before task/route, rather than weakening Mono.
- PORTAL Desktop #52 source old path normalizes via message.strip() and desktop_ipc._required_text; separate operational CognitionLedger and invoke_route before Mono task binding. This is SOURCE risk, not observed installed misconduct. No Training change to those owner files.
- Fresh PORTAL #53/#54 and CCB #372 are separate noncolliding reviewed work; preserve their owner boundaries. Do not create a second Codex conversation or assume a five-hour token/compute quota is available.

## 7. Real model admission: explicit exact blockers

Qwen3 4B is the operational development reference, NOT Vera-trained or the identity of Vera. GLM24B and other donors are unqualified optional candidates. Source-only tests/HTTP reachability/Ollama model inventory/synthetic SQLite/HMAC signatures NEVER authorize real dispatch.

Normal Windows host and VeraOS must consume an OS-owned verifier bound to current Mono OutboundTrustRegistry provider role, identity/revocation, **actual process-loaded bytes/manifest/serving process/host generation/session/endpoint**, current task ID/head/literal source digest, fresh receiver-minted one-use durable nonce, permission scope, causal transport response, Pre-Active DONE event + COMPLETED run + exactly one route, and task CAS/cold recovery no duplicate dispatch. All of this must be independently observed/installed and permissioned. Current prior /api/ps returned NO loaded models. Actual QEMU/Buildroot selected stage/import closure and graphical normal guest differ from source staging and remain separately qualified. Never call a source trial REAL_ADMITTED.

## 8. Next bounded Training work after new chat

1. Fresh-read exact owner branch, Bus #368 newest comments, current Codex/A1 owner return, and collision/delegation status.
2. On the current Training branch, verify last source/ledger/placement files unchanged and run only targeted source tests when Lappy has spare CPU/RAM. Do not rerun historical failures as if currently unfixed.
3. Highest-value source task: stress the governed predecessor helper against alternate source-history origin, identical event IDs in different conversations, redacted/deleted history, pruned legitimate history, cancellation and semantic-projection mismatch; if no **new** reproducible RED, do not invent another subsystem.
4. Future blind cognitive evaluation: preregister held-out canonical task cases with corrections/contradictions/context switches; compare Mono baseline vs Rezon replay vs optional Cricket checks using SAME independently authorized real-model outputs when separately permitted. Score correct useful answers, unsupported acceptances, false HOLDS, source fidelity, task recovery and resource cost. Do not call developer-fixture passing neural learning.
5. VN1 L3 next: independently verify signed loaded Qwen process/bytes/session/nonce and memory headroom by read-only observations only; report if unavailable. Integration L2 next: decide whether/read how to adopt typed retrieved source with governed semantic projection; do not edit its PR.
6. Never run concurrent GPU inference/model weights/paid compute, install provider/trust roots, merge main, write canonical memory, boot guest, change credentials/permissions, deploy, or perform protected effects without explicit exact authorization.

## 9. Quick links for successor

- [Training #106 Draft](https://github.com/thebrazenbeard/model-training/pull/106)
- [Bus #368](https://github.com/thebrazenbeard/chat-communication-bus/issues/368)
- [Current Training placement/retrieval return](https://github.com/thebrazenbeard/chat-communication-bus/issues/368#issuecomment-6234694888)
- [Portfolio #371](https://github.com/thebrazenbeard/chat-communication-bus/pull/371), [consumer overlay #372](https://github.com/thebrazenbeard/chat-communication-bus/pull/372)
- [Mono #77](https://github.com/thebrazenbeard/vera-mono/pull/77), [VeraOS #38](https://github.com/thebrazenbeard/vera-os/pull/38), [PORTAL #52](https://github.com/thebrazenbeard/portal/pull/52), [Executor #16](https://github.com/thebrazenbeard/executor/pull/16)
- [Actual stored model placement report](https://github.com/thebrazenbeard/model-training/blob/8f34fa64a468c7b0832fe127d95fd29fb8a051d5/docs/TRAINING_GOVERNED_RETRIEVAL_AND_LAPPY_PLACEMENT_20261017.md)

**Resume interpretation:** Continue safely from verified repository source and coordination. The project chat itself is not a persistent worker, runtime, model server, installed cognition route, independent reviewer or authority to activate Qwen.
