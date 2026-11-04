# R16 successor: matched MeZO-versus-QLoRA study contract

Status: **source-only, hard HOLD**. No trained Vera weights, physical training,
model download, rights admission, evaluator enrolment, host lease or private DEV
access is created by this module. All synthetic test outcomes are fixtures.

Run offline: `python -m unittest tests.test_trainingday_matched_method_comparison`.

## The two comparisons are different questions

1. **Causal optimizer comparison:** MeZO applied only to the **same pinned
   trainable LoRA targets** as QLoRA/AdamW, on the exact same authenticated
   Qwen3.5-4B base, tokenizer, training split and seed list. This isolates the
   algorithmic choice more effectively than full-parameter MeZO versus
   adapter-only QLoRA. Still record forward/backward counts and actual token
   exposure; matching a time cap alone cannot equate compute or examples.
2. **Practical deployability comparison (separate):** Original full-parameter
   MeZO versus QLoRA under the same per-device hardware and wall-clock cap.
   Explicitly acknowledge differences in trainable capacity and memory.
   Never pool these two studies or describe either as a fully controlled
   head-to-head on algorithm alone.

Each causal study seed gets **seven** arms: MeZO / QLoRA crossed with correct,
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
