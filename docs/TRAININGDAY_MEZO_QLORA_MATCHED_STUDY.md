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

Each causal study seed gets six arms: MeZO / QLoRA crossed with correct,
corrupted and zero-update label modes. Preserve one independent base-model
reference. Require at least three predeclared seeds and a fixed bounded
wall-clock budget, **plus** recorded optimizer steps, unique example exposure,
token exposure, forward calls, backward calls, VRAM, host RAM and source-
identical model/adapter target commitments. For each trained arm, require a
fresh-process checkpoint reload and interrupted-run resume. Zero-update arms
must perform no optimizer steps; they still require a fresh snapshot reload.
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

The public receipt validator checks schedule completeness, unchanged plan
commitments and telemetry *shape*. It cannot tell whether a run actually
occurred, whether telemetry is forged, whether a host broker held the GPU, or
whether independent evaluator custody survived a rollback. The Q6 independent
monotonic authority, authenticated base/adapter bytes, admitted data rights and
separate protected DEV custody remain **unqualified**. Green unit tests do not
change those gates.

Primary research: [MeZO (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html),
[QLoRA (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf).
