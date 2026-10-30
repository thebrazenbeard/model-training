# TrainingDay R16 — checkpointable LoRA pilot (review gate)

Status: **SOURCE_TEST only; no R16 GPU optimizer step or resume equivalence verified.**

This adds an opt-in R16 path to `train_neural.py`, preserving the existing
R15 behavior and receipt format unless `--r16-checkpointable` is supplied.
It targets the same pinned official Qwen3.5-4B NF4 base, **not** a GPT-OSS
checkpoint or Vera's deployed local model.

## Frozen experiment

- Labyrinth v2 manifest SHA-256:
  `90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836`
- Producer Git head:
  `1d38e4f25b5663bf8a84ff5e139875d4152e3680`
- 0-step reference or 8/16/24/32-step correct/corrupted-label arms, with eight
  selected examples per optimizer step, all bounded to 1,024 cases.
- Seed, source selection, base binding, answer formatting, hyperparameters,
  and initialization digest are bound into an immutable per-run contract.
- Trainer is configured for `save_strategy=steps`, `save_steps=8`, and
  `trainer.train(resume_from_checkpoint=...)` when resuming.
- No automatic promotion, merge, deployment, final-bank access, model-to-model
  weight averaging, or unadmitted code-source training.

## New run

Run the R15 command line with these changes, using a *new* run ID:

```text
--r16-checkpointable --mode correct --steps 16 --cases 128
--expected-initial-digest <exact-zero-control-digest>
```

The zero-update arm uses `--mode zero --steps 0` and is not resumable.
Corrupted and correct arms must use the identical corpus/binding/seed/schedule
and separate run IDs/output folders. Existing R15 source/tests do **not**
establish that longer R16 training improves generated reasoning.

## Resume after interruption

1. Confirm the last `trainer-work/checkpoint-N` write finished. Checkpoint
   files are treated as executable/serialized inputs; never accept a downloaded
   or untrusted checkpoint even if an attacker provides a matching digest.
2. Inventory its trainer state, LoRA adapter, optimizer, scheduler, RNG state,
   all file digests and exact step using:

   ```text
   python -m trainingday.checkpoint_custody
     --checkpoint <absolute-run>/trainer-work/checkpoint-8
     --output-receipt <separate-trusted-directory>/r16-custody.json
   ```

   The output receipt **must be outside** the run directory. Pin its SHA-256
   independently before permitting any Trainer deserialization.
3. Repeat all original run arguments and append:

   ```text
   --r16-checkpointable
   --resume-checkpoint <absolute-run>/trainer-work/checkpoint-8
   --resume-custody <separate-trusted-directory>/r16-custody.json
   --resume-custody-sha256 <externally-recorded-receipt-sha256>
   ```

The runner rejects changed inputs/hyperparameters, missing or altered
optimizer/scheduler/RNG/adapter files, stale step counters, symlinked paths,
in-run custody receipts, or a checkpoint at/after the target step.

## Physical qualification required before use for long campaigns

1. Run isolated R16 zero and 8-step correct/corrupt GPU controls on Lappy.
   Check actual loss labels in the constructed SFTTrainer batch, not only TRL
   source or a standalone collator call.
2. Inspect real `checkpoint-8` file inventory and independently reload the
   final adapter after writing.
3. Compare uninterrupted 16 steps to an independently interrupted 8+8 resume
   from the **same initialization/corpus/seed**; check full FP32 LoRA tensor
   digests, optimizer state, RNG and exact output. If nondeterminism prevents
   bitwise equality, define and freeze a justified tolerance before outcomes.
4. Evaluate frozen/zero/correct/corrupt on a newly reserved, unchanged case
   bank, with retention anchors and a matched no-KL/KL study when implemented.
5. Keep R16 developmental and do not replace `vera-local:latest` until
   generated accuracy/retention and operational gates actually pass.

The source tests cover admission and custody *logic*, not real GPU checkpoint
format, optimizer-state restoration, tensor equivalence, or performance gain.
