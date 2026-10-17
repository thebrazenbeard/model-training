# Training L2/L3: governed correction recovery and model placement

2026-10-17. Draft #106, SOURCE ONLY. No provider loading, model inference,
training, installation, trust provisioning, canonical memory admission, QEMU,
external publication, production effect or Codex dispatch.

## Exact currentness, authority and control

Current at this cut: Training #106 prechange
7a74833c4f3fe297ddb3f2f82777da1438d2a8b0;
Mono #77 89ba1d18da78e7e38f015cf8d8ef4f528aae9df8;
Integration VeraOS #38 33432d39255670c2fe258da2cb998bfa0a4cdf4a;
PORTAL #52 b16413fad12d33a9f9b64d6b1602490f4dede327;
private 96-repo review #371 951e7af1a97694faaf6bc18d51b35f70982e31fc.
Pre-Active selected donor 1f23a809d7274df506e03e2d9052525538c3bcf6,
Volition selected dbc628d376515a0a523b1eecdf62129cca5d6b08.
Separate default donor heads are source provenance, not resident imports.

PORTAL's single Codex A1 is a different active subject. Only Training
branch files written. Exact Codex saved five RED tests and locator V1 policy
reviewed read-only; no Codex branch changes or Codex dispatch.

## Existing safe cognitive pipeline, not an additional supervisor

Exact Desktop/UI literal and explicit conversation/message IDs -> Mono
TaskExecutionLedger open_message (exact subject/source digest/head) ->
governed selected context and source history -> optional typed salience
and Rezon reasoning proposals -> advisory model placement ->
OS-owned authenticated host/provider loaded artifact, generation/session,
endpoint, task/head/current trust and nonce -> accepted transport output
and Pre-Active DONE + completed run + ONE route -> Mono compare-and-swap
task result/cold recovery -> separately permissioned governed memory or
learning proposal. No status, desire or callback generates authority.

## NEW correction-recovery RED -> GREEN

The earlier guard at Training 7a74833 correctly HOLDS a correction whose
typed predecessor is not supplied. It cannot tell a fabricated orphan from
a legitimate pruned predecessor. The source-only improvement is a NEW
read-only helper cognitive_governed_predecessor_v1.recover_pruned_predecessor,
which accepts EXACT current vera_core.task_execution.TaskExecutionLedger,
not an arbitrary caller/remote callback. It validates the current real
canonical task ID, head/source digest, current user's literal text,
nonclosed/nonunresolved state, exact current source event receipt and
same-conversation predecessor. It reads historical Mono task opener,
verifies its exact accepted predecessor text and event digest, constructs
a narrowly scoped source-record copy, then reuses the existing strict
guard. It rechecks current journal head before return. It does not call
open_message, write a task, infer a semantic value, publish a user fact,
import memory, query a model or authorize transport.

Real disposable Mono #77 task tests first saw the HOLD-only baseline
failing 2 expected utility assertions, then implementation and hostile
tests GREEN. 11 genuine read-only ledger tests, no skips on Lappy.
A frozen seven-case source-only comparator evaluates (1) HOLD-only,
(2) unsafe self-declared heuristic acceptance, (3) canonical retrieval.
Developer labels: TWO assumed supported (one genuinely present in
disposable ledger, one HYPOTHETICALLY legitimate but absent) and FIVE
adversarial. Baseline answers 0/2 useful, unsafe 0/5; heuristic 2/2
and unsafe 5/5; canonical retrieval 1/2 useful, unsafe 0/5.
The remaining 1/2 false HOLD is an unavailable predecessor:
indistinguishable from a fabricated orphan at this layer. No inference,
independent factual/source-origin corroboration, model learning or
production installation measured.
Evidence digest and per-case outcomes recorded under docs/evidence/
TRAINING_MONO77_GOVERNED_PREDECESSOR_20261017.json.
Negative tests: missing/foreign prior, switched conversation, stale
head, changed accepted digest, false current citation, altered literal,
no ledger write and different-conversation same ID.

**Hostile review ? accepted:** The helper projects the referent/scope
provided in a typed observation onto a verified historical USER INPUT;
Mono proves only exact bytes, not that this text really refers to the
same subject, that the claimant is authentic or that a correction is
true. It must remain a Training source utility until a Mono owner
controls semantic projection and authorizes route consumption. Blind
model evaluation and independently authenticated source-origin evidence
are separate requirements.

## Lappy READ-ONLY capacity and exact offline GGUF metadata

Lappy observed 2026-10-17 09:34:11 local: 32 GiB installed (31.73
GiB usable), 3.49 GiB free physical RAM at observation; RTX 3050 Laptop
4096 MiB physical VRAM with 3954 MiB free via nvidia-smi; D: drive
152882036736 bytes free. Multiple Codex processes observed. This is
a host snapshot, not guaranteed available reserved RAM/VRAM. Do NOT
run models concurrently with the Codex slot on these measurements.

Model GGUF headers were parsed by inspect_model_placement_v1.py with
bounded struct reads, no weight load, custom-code import or inference.
GPU budgeting illustration reserves 768 MiB for allocator/compute
buffer and uses full-model FP16 KV at stated token count, NOT a measured
engine's allocation. Layer figures ignore nonlayer tensors and exact
backend/pipeline allocation; they are OPTIMISTIC UPPER BOUNDS for
layer-only on-disk weights, NOT measured offload counts.

| Verified local artifact | File bytes / disk GiB | Header architecture / blocks | FP16 KV MiB at 2k / 4k / 8k | Optimistic disk-layer GPU bounds at 4k and 768MiB reserve | Placement verdict |
| --- | ---: | --- | --- | --- | --- |
| Ollama stored sha256-85e4a5b7... Qwen3-4B Instruct-2507 | 2497280480 / 2.326 | qwen3 / 36 | 288 / 576 / 1152 | 36 of 36; full on-disk size within illustrative budget | GPU candidate at 2k-4k, not proven allocator fit; CPU fallback only if free RAM recovers; no parallel Qwen |
| gguf/model_q5_k_s.gguf Qwen3.5 4B | 3072993056 / 2.862 | qwen35 / 33 | 264 / 528 / 1056 | 33 of 33 layer-only; full file EXCEEDS budget after reserve at 4k | partial GPU/offload or CPU fallback; distinct from Qwen3 reference; do not auto-select |
| GLM24B IQ2_XS candidate GGUF | 7207034784 / 6.712 | GGUF declares llama / 40, GLM in model name | 320 / 640 / 1280 | 16 of 40 layer-only | mixed CPU/GPU only in principle; FULL VRAM impossible; insufficient currently free RAM |
| GLM24B Q4_K_M candidate GGUF | 14333910944 / 13.349 | GGUF declares llama / 40, GLM in model name | 320 / 640 / 1280 | 7 of 40 layer-only | heavily CPU resident; FULL VRAM impossible; not a Codex-concurrent option |

The GLM label is repository/model nomenclature; metadata reports
general.architecture=llama, not authenticated runtime model identity.
Large context lengths advertised by GGUF are NOT resource-admitted
context lengths. No actual GPU-offload layers, speed, peak RAM or
latency measured; runtime overhead/KV quantization/graph allocations
vary by engine. The GPU snapshots do not authorize inference.
At this free-RAM moment, even CPU fallback for full GLM is inappropriate.
Concurrently loaded GLM+Qwen or multiple Qwen sessions should be HOLD.
Wait for separate scoped load authority, an uncontended resource window,
verified engine load/profiling and admitted current provider identity.

The 3 offline HF donors under donors/huggingface remain research-only;
their past SHA/Git/LFS inspections are provenance, NOT runtime selection.

## Ranked portfolio mechanism falsifiers, NOT new organs

Every row has a task and simplest Mono baseline; no non-Training donor
received an empirical behavior comparison this turn. Resource and
privacy ratings are hypotheses to measure, not observed runtime costs.

| Candidate exact head | Task, simple baseline, proposed A/B, positive and adverse tests | Target Mono API, cost, privacy and decision |
| --- | --- | --- |
| Rezon e14fc6921d4f56bb6f6dc306e72b33ef25fd6777 | Task contradiction, baseline current Mono source+simple replay vs typed ReplayCase strategy; positive correction recovery; negative source swap/circular proof | prequential_evaluation/task-scoped reasoning; low CPU source, model cost unknown; no private source export; FIRST next independent model-output ablation |
| Cricket 0bae95bc45db55d78001ee9c492681834191ec0b / Voss | Catch unsupported answer vs same source guard; compare optional rule/critic pass; negatives false criticism, fabricated dissent | advisory review only; extra latency/possibly compute, keep private context; DEFER until measured |
| SemanticAtlas 2cd42a91830b7bd3801896393588c1c1a144b59a + Semiotics 857d912975af07fdd5a1e0e3cc39bda82e5287e0 + SPM 0dfe59a3b708d113c2b420f9cd508e776556f90b + SQL Connectome b609fcec50fe5a135ca1d8139f5563f5631582b2 | Referent/ambiguity and SQL provenance task, baseline typed Mono interpretation, targeted semantic mapping; negatives dialect mismatch/cross-domain or unknown referent | vera_identity.contextual_interpretation, authorized SQL tasks only; context/database/privacy cost; DEFER wholesale |
| Volition 64e61c546b89f5dc0eb78d382637ccccef42aa31 + MESO-CRCT d8214069cf41c8a92b84805ea2b10bc2a908d227 + Conations | Safe initiative suggestion, baseline Mono initiative/salience, optional appraisal; negatives goal drift, unrequested action, pressure to "want" | vera_runtime.protocol.initiative_kernel; cheap source probes; proposals not consent/effect; DEFER |
| Mosaic a2d4dc9812156f41e370e062dad5e2b45a0dab1e | Single Qwen reference placement vs specialist swap, context-change accuracy+latency; adverse VRAM eviction, concurrent process pressure | optional placement advisory; current 4GiB GPU, resource-heavy; HOLD until measurements and authority |
| LGCM 9b9081b404b0f7ade73d83768758873bb2b988fd + Noema d6251fecb083be1a6bf1d6a300ef0a4ffcac6186 | Frozen-seed prequential numerical prediction vs Mono baseline; negative out-of-distribution and catastrophic forgetting | prequential_evaluation only; independent research, not language model weights or memory; DEFER |
| HC Brain f7ffd4a7bfe5c360a139cfe3925bae22419afa16 + God Brain a4ad8fb8a581fed91db35664164db4c1fcb5de8d + Transcendence 4da8646b618cb1d5c62d2d2465151adcb5137251 + BT2 0f35778fb3705bf642533f543b7080b7282114bd | One-task continuity baseline vs imported cognitive loop; negative competing task/memory owner and state duplication | current sampled HC/God cognitive_loop same blob 5a4c98c6; no second supervisor, unmeasured memory/CPU costs; REJECT runtime import |
| Roots 24ed94d69ec07c5bba638de9ed2b8ecfad3cc7ce + Ingest e55af474323f2115c5181b443760b8427090562f + Temporal c473850285eed69c1dc3e479fa59f379bbdf5c6d + Redworm c87f93dc40f3009807b7964ec845048f6af06d49 | Task-specific chronology baseline Mono exact stored source vs donor retrieval; negative forged autobiographical memory, historical replay and deleted/cross-person records | vera_ingest and vera_memory provenance only; potentially private, retention/interference budget; no auto memory admission |
| Ears fc244fd18cd9bdabe40c03c6245342215a143307 + Vision 5f1dfbb717193e97fb9c4c1096122aea2f522430 | Typed sensor event task, baseline manually supplied observed text vs bounded DSP/image source; negatives identity attribution, hallucinated object, privacy leak, stale sensor | vera_ingest source-typed perception; potentially sensitive/raw media, CPU+privacy cost; no microphone/camera used, DEFER until permission |
| Separate Qwen3/GLM24B/HF candidates | Exact same heldout task set for quality and run cost once approved; negative provider substitution/replayed nonce and memory-state mutation | OS-owned provider verifier, Mono task head, PreActive DONE/COMPLETED unique route; no candidate currently admitted |

## Actual Codex A1 RED and locator V1 (read-only review)

Read saved artifact under Lappy Documents/Codex/.../outputs:
test_desktop_mono_canonical_intake.py has FIVE original assertion REDs:
(1) TASK_OPENED absent before route discovery; (2) raw Unicode padded
literal silently trimmed by handler; (3) same by view-model;
(4) synthetic operational success with no Mono task and transport call;
(5) view-model supplies no conversation_id.
Original a1-resume-red-junit.xml lists 5 failure/0 errors.
Later a1-green-first-junit.xml reports those FIVE targeted tests
apparently green, but whole 15-test suite FOUR ERRORs on WinError5
FileBridge os.link (not a clean acceptance). Do not infer installed
Desktop success or a merged Codex implementation from saved artifact.

PORTAL owner locator V1 Bus comment #6233757408:
client-created opaque per-conversation ID, explicit request_id as Mono
message ID, no host-invented locator on missing input, persistent
client-owned selector only, input literal forwarded unchanged or
explicitly rejected under Mono #77 strip constraint, no-provider HELD
with ZERO model dispatch, no legacy operational-history migration.
The 09:33 saved a1-resume-gate-receipt.md said Mono ABI acknowledgment
was missing. This is NOW SUPERSEDED by Integration's explicit source ABI
ACK at Bus #368 comment 6234499327 / Mono #77 comment 6234501080.
No Mono #77 API change is required. Source ABI compatibility is NOT
Codex GREEN implementation or installed desktop qualification. New
PORTAL-only adverse case from that ACK: FileBridge request ID grammar
allows leading punctuation and 97-128-character IDs that Mono rejects.
Require Mono's narrow [A-Za-z0-9][A-Za-z0-9._:-]{0,95} ID syntax
BEFORE any request ledger/model work, do not widen Mono's contract.
Integration VeraOS #38 advanced to exact
46152a55393642ed2fc614abf177d6c2572a36c5 with additional
static target-file/ELF dependency presence tests, still no guest boot.
Training opinion: YES to literal identity and pre-route task open;
preserve message ID on uncertain retry, return OUTCOME_UNKNOWN,
do not treat CognitionLedger or a synthetic route as canonical
completion/independent evidence. No PORTAL/Codex edits by Training.

## Provider contract / evidence ceiling

Expected future Windows and VeraOS route: accepted Mono task ID,
exact journal head/source digest + literal, current HOST/PROVIDER
role trust/revocation, exact loaded-process model bytes/manifest and
serving endpoint + enrolled host generation/session, receiver
challenge/nonce durable one-time consume, causal transport response
byte/digest, Pre-Active DONE event / COMPLETED run / single selected
route, and task CAS result readback/cold no redispatch. Distinct
provider/admission from user provenance; learned state/memory separate.
Current Training source-only gate/readback remains unchanged. No
actual loaded-model attestation, invocation or installed resident
verified; Qwen3, Qwen3.5, GLM24B all real provider HOLD.

Evidence files: docs/evidence/TRAINING_MONO77_GOVERNED_PREDECESSOR_20261017.json
and docs/evidence/TRAINING_LAPPY_GGUF_PLACEMENT_20261017.json.
Reproduce source tests with VERA_MONO77_SOURCE_ROOT pointing to exact
read-only Mono #77 checkout, VERA_MODELS_ROOT to D:\VERA\models.
