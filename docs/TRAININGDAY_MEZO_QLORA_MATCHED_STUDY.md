# R16 successor: matched MeZO-versus-QLoRA study contract

Status: **source-only, hard HOLD**. No trained Vera weights, physical training,
model download, rights admission, evaluator enrolment, host lease or private DEV
access is created by this module. All synthetic test outcomes are fixtures.

Run offline: `python -m unittest tests.test_trainingday_matched_method_comparison`.

## The two comparisons are different questions

1. **LoRA-restricted training-recipe comparison:** MeZO and QLoRA/AdamW use
   the **same declared LoRA targets**, Qwen3.5-4B source revision, tokenizer,
   TRAIN commitments and seed list. This narrows one confound but does **not**
   causally isolate the gradient estimator: MeZO's zeroth-order estimator,
   QLoRA's AdamW update rule, and possibly different NF4/backbone precision
   remain mixed. The plan names itself `RECIPE_COMPARISON_ONLY` and sets
   `causal_gradient_estimator_isolated=false`. Isolating the estimator would
   require an additional study with matching optimizer update rule, executed
   backbone precision, batch loss, target tensors and hardware kernels.
2. **Practical deployability comparison (separate):** Original full-parameter
   MeZO versus QLoRA under the same per-device hardware and wall-clock cap.
   Explicitly acknowledge differences in trainable capacity and memory.
   Never pool these two studies or describe either as a fully controlled
   head-to-head on algorithm alone.

Each study seed gets **seven** arms: MeZO / QLoRA crossed with correct,
corrupted and zero-update label modes, **plus an untouched official base
reference with no adapter**. Three seeds produce 21 independently named arms.
An initial shape-only version of this contract omitted the base from its
validated schedule; the independent hostile review caught that gap.

Require at least three predeclared seeds and a fixed bounded wall-clock
**ceiling**; the actual elapsed time may legitimately differ and is a measured
efficiency outcome, not evidence of a matched compute expenditure. The
method-to-method comparisons now enforce equal counts of training examples,
tokens and the same declared example-order SHA-256 within each seed and
label condition. Source receipts record optimizer steps, forward/backward
calls, VRAM and host RAM with positive memory telemetry. The validator checks
consistency, not truthful measurement, causal equivalence of hardware kernels,
or host custody. For each trained arm, require a fresh-process checkpoint
reload and interrupted-run resume. Zero-update arms must perform no optimizer
steps; they still require fresh snapshot reload. The untouched base requires
`adapter_sha256=null`, no optimizer updates and no training exposure; a
PEFT zero-step arm is **not** an untouched base.
All true training paths must pass the separate protected physical launch
authority; these declarations cannot make them runnable.

### Initial adapter identity and the zero-update causal control

The V4 study retains V3's requirement for one `initial_adapter_commitments` record **for every
seed**. Its `initial_adapter_sha256` is frozen into *all six PEFT arms of
that seed* (both methods × all three label conditions), while the untouched
`OFFICIAL_BASE` reference has **no adapter** and a null initial commitment.
The zero-update control's **final** `adapter_sha256` must equal its exact
predeclared initial commitment; merely claiming zero optimizer steps is not
enough. One adversarial V2 receipt substituted an unrelated but well-shaped
final adapter SHA for `MEZO_LORA:zero:11` and was wrongly shape-accepted;
V3 rejects that exact falsifier and mismatched method/base initial hashes.

To compare methods across different quantization/serialization wrappers, an
upstream qualified artifact verifier must derive a **canonical tensor-only
digest of the LoRA initialization**, with fixed target names, order, dtype,
shapes and raw tensor bytes. Do not hash wrapper-specific optimizer metadata
and then call two different representations comparable. This canonicalizer
and its host-owned readback are **not** implemented in this source protocol;
the current digest remains supplied, untrusted evidence, and the study plan
explicitly records `initial_adapter_bytes_independently_attested=false`.
Distinct seeds do not automatically demand distinct initialization hashes:
seeds may vary data order with identical initial tensors. A changed training
arm's adapter hash alone never proves reasoning improvement or proper updates.

### Actual ordered TRAIN content and label-custody commitment

The V4 plan **requires** a distinct `schedule_commitments` entry for every
seed, with exactly three SHA-256 digests: `train_order_sha256`,
`correct_labels_sha256`, and `corrupted_labels_sha256`. The pure helper
`commit_training_schedule(seed, ordered_rows)` generates these from the
full ordered `(row_id, input)` sequence and, separately, the ordered
`(row_id, assigned_label)` pairs for each correct/corrupted condition. It
canonicalizes UTF-8 JSON with sorted object keys, no insignificant spaces,
domain labels, and the seed. Each input row must carry distinct `row_id`,
`input`, `correct_label`, and `corrupted_label` strings. The model adapter
must build these *from the actual admitted TRAIN rows after the corruption
transform* and independently re-hash the data at read time: a made-up digest
is not authentication. Never log or commit raw private training examples,
private evaluation cases, or the corruption seed's secret material.

Every trained arm must return the **precommitted** order and assigned-label
digests; a pair of method receipts that colludes on a different digest is
rejected against the frozen plan. No cross-seed corrupted-label schedule reuse
is permitted. For public R202 as a future input, canonicalization of actual
row payloads (including chat template, tokenizer and special tokens) would
need an explicit adapter and verified data lineage; none has been built here.

The `candidate_sequence_length=128` field is a planning candidate, **not**
a certified runtime context limit. A source-only R202 check with the pinned
official Qwen3.5-4B tokenizer found **757 of 1,536** simple
`prompt + newline + answer` rows above 64 tokens (max 91); none exceeded
128 *without template overhead*. A subsequent corrected **official chat-
template** tokenization measured min 44, median 77, p95 99, max 104 tokens;
**1,024 of 1,536** examples exceed 64 tokens and **none** exceed 128.
The earlier observation `len(BatchEncoding)=2` was a misuse of the API; the
corrected counts use `len(input_ids)` with the actual pinned tokenizer. Thus
seq64 would truncate most R202 examples, while the tested template fits
128 token positions. This does **not** qualify actual activation memory,
device support, throughput, or unrelated admitted curricula. Physical launch
qualification must recheck all examples using its actual formatting and reject any truncated labels;
the receipt's `truncated_training_rows=0` is only untrusted telemetry.

### Wall-clock cap versus measured cost

The two methods receive the same **maximum duration**, not equivalent actual
compute or guaranteed equal numbers of FLOPs. The V4 plan predeclares a
0–50% `actual_time_tolerance_percent` (default 20%), then measures the
relative difference against the larger elapsed time for each seed/label pair.
Over-tolerance pairs return
`TIME_DISPARITY_REQUIRES_SEPARATE_ANALYSIS`, never an equal-cost claim.
Unequal forward/backward counts are reported separately; those counts
have different semantics for zeroth-order MeZO and backpropagation QLoRA,
so even equal counts or equal seconds cannot prove matched compute.
`measured_compute_equivalence_established` remains **false** by construction.
Independent accelerator time/energy/FLOP profiling and separate quality-vs-
cost plots are required for an actual resource-efficiency comparison.

### Source-only optimizer-step falsifiers (V4)

The standard supervised QLoRA/AdamW recipe must execute a forward loss graph
for each backward call. V3 wrongly accepted a synthetic training arm claiming
10 optimizer steps, 10 backwards, but **only 1 forward**. V4 rejects
`QLORA_ADAMW.forward_calls < backward_calls` for trained arms, while accepting
legitimate gradient accumulation (for example, 20 forward/backward microbatches
over 10 optimizer steps) and true zero-update controls. A deliberately
different recipe that backward-propagates multiple losses through a single
retained graph needs its *own explicit accounting contract*; it is not this
standard QLoRA procedure.

Original MeZO uses two loss evaluations at `theta + epsilon * z` and
`theta - epsilon * z` on the **same minibatch**, with replayed perturbation
direction and restored unperturbed parameters before update. For `n`
independent directions, its finite-difference estimator takes **2n forward
evaluations** per optimizer step. The optional bounded
`mezo_directions_per_step` source field enforces at least `2 * n * steps`
forward-call telemetry when explicitly declared; when absent, only the basic
two-per-step minimum is enforced. Neither case proves actual paired minibatch
identity, perturbation replay, truthful GPU telemetry, or restoration:
`mezo_perturbation_pairing_independently_verified=false` always. Forward calls
and token exposure are not interchangeable compute or energy units.

The pinned model ID `Qwen/Qwen3.5-4B` is the official **post-trained**
checkpoint. `Qwen/Qwen3.5-4B-Base` is a different pretraining-only model.
`OFFICIAL_BASE` in this study means the untouched, no-adapter version of the
pinned **post-trained** checkpoint, not the distinct `-Base` release.

The private evaluator must hold an independently protected test set unseen by
trainers, disjoint from TRAIN in both surface and equivalent problem families.
The public study plan records only a hash commitment to that protected set;
the model-training repo must never contain unsealed DEV questions or answers.
Store paired per-case identity/outcome records solely with the evaluator.
The pure `summarize_paired_binary_outcomes` accepts 16-hex-digit case IDs and
booleans and emits *aggregate-only* exact two-sided McNemar counts. A private
evaluator must use independently protected keyed pseudonymization: the public
test fixture's sequential IDs are not private identities.
It is **not** independent evaluation custody or proof of authenticated scores.

## Exact source bridge to existing rights and evaluator work

Reviewed as of November 4, 2026: `model-training` Draft
[#296](https://github.com/thebrazenbeard/model-training/pull/296) source head
`effd5f10ae354d5bc7a1eb1ea4c977e30f0792e5` contains
`trainingday.calc_eval_commitment.matched_commitment`. Its source-only return
contract has schema
`VERA_TRAININGDAY_R201_CALC_MATCHED_PRIVATE_DEV_COMMITMENT_V1`,
`dev_count=64` and `private_dev_sha256` that may be copied as this plan's
`private_dev_commitment_sha256` *only after an independently custodied,
authenticated evaluator supplies it*. Its matched cubic/small-|x| DEV
generator checks public R201 TRAIN exact source and derivative-semantic overlap,
but independently sealed nonce entropy and evaluator custody remain false.
Neither the private nonce nor any DEV prompt, answer, or per-case result
belongs in a public plan, source repository or trainer process. The private
evaluator alone must construct keyed anonymous case IDs and release only
aggregates for a separately authenticated comparison receipt.

The same draft has `trainingday.r201_owner_claim_candidate` with exact R201
subject checks. It can report `signature_valid_for_supplied_key=True` while
`owner_key_enrollment_authenticated=False`,
`corpus_rights_admitted=False` and
`protected_evaluation_authorized=False`. These fields are not interchangeable.
To attach this matched study, integration must independently establish an
admitted source-rights/exclusions claim, legitimate protected evaluator
custody, trusted Qwen/adapters, host-owned process lease and an independent
Q6 monotonic witness. **No such bridge is implemented or qualified.**

Another separate Draft [#332](https://github.com/thebrazenbeard/model-training/pull/332)
offers public, TRAIN-only R202 material; its 1,536 generated examples are
neither a protected DEV set nor a substitute for a separately admitted R201
lineage. Cross-curriculum use would require an explicit new rights and split
contract and a distinct study plan rather than silently rebinding this one.

## Why the October 30 R16 result is not a learning qualification

The historical four-arm report says base 3/64, zero-update 3/64, correct-label
eight-step 5/64 and corrupted-label eight-step 3/64. Those are **marginals**;
without the paired identities and correctness bits for each case, exact
McNemar statistics cannot be computed. Even the most favorable nesting of
three baseline hits within five trained hits yields two discordant cases and
a two-sided exact p-value of 0.5, not persuasive evidence of generalization.
The observed 2/64 change could reflect noise, exposure effects, or real but
small task-specific benefit. Adequate independent heldout coverage, statistical
power analysis and multiple seeds are required before claiming improvement.

The physically re-inspected October 30 initial R16 training receipts say
`R16_ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED` and corpus status
`ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE` even though eight physical
optimizer steps were recorded. A *later* separate-process adapter reload was
reported in a different Bus event. Those are time-separated evidence classes;
the first receipt itself does not prove the later event, and neither status
grants a new R201 curriculum or rights authorization. For a new study, preserve
original event times and immutable artifact identities, then reproduce a
fresh-process reload independently before using those weights as controls.

The public receipt validator now rebuilds the complete canonical seven-arm-per-
seed schedule, rejecting a rehashed plan that dropped a method/base arm or
changed any held authority flags to true. It checks declared cross-method
exposure and order parity, plus telemetry *shape*. It cannot tell whether a run actually
occurred, whether telemetry is forged, whether a host broker held the GPU, or
whether independent evaluator custody survived a rollback. The Q6 independent
monotonic authority, authenticated base/adapter bytes, admitted data rights and
separate protected DEV custody remain **unqualified**. Green unit tests do not
change those gates.

Primary research: [MeZO (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html),
[QLoRA (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf).
