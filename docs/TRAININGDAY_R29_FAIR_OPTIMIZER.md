# TrainingDay R29: public scalar-CE optimizer fairness witness

**Evidence:** source inspection + synthetic CPU toy mathematics. **Authorization:** none.

This deliberately narrow artifact addresses a gap that cannot be resolved by
another descriptive claim that MeZO and QLoRA are “comparable.” Matched source
loss expressions and equal proposed seconds are necessary checks, not physical
proof of equal corpus, tokenization, training exposure, optimizer dynamics, or
observed behavior. It does **not** recover or rehabilitate historical R18.

## Exact source contracts examined

- `trainingday/r18_mezo.py`: `step_rows()` selects cases by modulo cycling;
  `paired_batch_loss()` aggregates effective-batch completion-only causal LM
  CE in proportion to actual supervised target-token counts. Each MeZO
  finite-difference step runs two perturbed forwards, no backward pass.
- `trainingday/r18_qlora.py`: `_training_loop()` takes one non-repeating sweep
  with effective batch eight, and calls
  `effective_batch_token_loss_scale()` to combine microbatch means in
  proportion to target-token counts; an AdamW8bit optimizer is constructed
  **without** an explicit `weight_decay=` argument. A source inspection
  cannot establish which effective runtime default applies.
- `trainingday/r24_optimizer_sampler_audit.py`: the existing exposure simulator
  is reused rather than duplicated. It counts optimizer updates, unique and
  repeated cases, target/input tokens, forward/backward work, and padding.
  `r24_tokenizer_parity.py` provides the separate tokenizer implementation
  parity check on caller-supplied cases, not a pinned physical tokenizer hash.

## Public objective math

For public toy case `i`, with labels `y_{i,t}` at supervised causal positions
and a *single frozen* logit snapshot `z_{i,t}`, the scalar completion CE is

\[
L = \frac{1}{\sum_i T_i}\sum_i\sum_{t=1}^{T_i}
  \left[\log\!\sum_k e^{z_{i,t,k}} - z_{i,t,y_{i,t}}\right].
\]

MeZO evaluates this full effective-batch scalar at each of two perturbations.
QLoRA computes mean CE in each microbatch `j` then accumulates
`sum_j (T_j / T_all) * L_j` before AdamW8bit update. An unweighted mean of
microbatch means is generally unequal when supervised completion lengths vary.
This proves only the mathematical *functional objective* for identical frozen
logits and labels; MeZO's stochastic finite difference and QLoRA's backprop
are not equivalent gradient estimators or optimization trajectories.

`scalar_objective_witness()` rejects nonfinite logits, invalid target IDs,
non-causal position-zero labels, interior completion masks, duplicate IDs,
unequal vocabularies, and partial microbatch partitions. It returns SHA-256
commitments for ordered `(case_id, input_ids, labels)` and ordered frozen
logits; compares first/last IDs and commitments in every arm; and refuses a
parity result when labels or logits differ. It returns no prompt, completion,
token IDs, protected examples, or model weights.

## Source schedule + timing + controls

`audit_fair_optimizer_pair()` combines the above with R24's bounded source
exposure simulation. It checks declared monotonic active optimizer time
against step sums, declared wall time against budget, discrepancy between wall
and monotonic clocks, and nonzero Modern Standby exposure. Declared controls
must include zero-update pre/post tensor hashes that match, distinct correct-
and corrupted-label commitments, and nonzero updates in each labeled arm.
Controls and clocks are **caller declarations**, not signed host receipts.

Always-present non-authorizing blocks include
`RUNTIME_EFFECTIVE_ADAMW8BIT_DECAY_UNATTESTED`,
`CONTROL_RECEIPTS_NOT_INDEPENDENTLY_ATTESTED`, and
`PHYSICAL_EQUAL_TIME_NOT_ATTESTED`. An explicit `weight_decay=0.0` in source
would remove one source syntax mismatch only; it does not verify effective
bitsandbytes behavior or make two optimizers dynamically equivalent.

Historical R18 MeZO was planned for **531 seconds** but elapsed
**19,813.562 seconds**, with **19,698.314 seconds** of Modern Standby.
`TIME_BUDGET_OVERRUN_INVALID` is permanent for that pair. Neither a synthetic
toy substitute nor an R18 resumed checkpoint can convert it into fair
equal-time evidence.

## Verification and next physical gate

Run without GPUs, hosted compute, credentials, or private corpora:

```sh
python -m unittest -v tests.test_r29_optimizer_fairness
python -m unittest -q tests.test_r24_optimizer_sampler_audit tests.test_r24_tokenizer_parity tests.test_r25_budget_lease_policy tests.test_r29_optimizer_fairness
```

To establish a *real* fair pair, a separate lease-bound runner must supply:

1. Exact model/base/adapter/tokenizer hashes and authenticated admitted corpus
   selection; independently replay and verify target IDs, ordered case IDs,
   target-token loss masks, and first/last presentations.
2. Identical preregistered wall budget with monotonic/wall/Standby and completed
   optimizer-step receipts; no overrun or hidden sleep/Standby; comparable
   active dose measured separately as examples/unique cases/forward and
   backward useful and padded tokens. Single-sweep exhaustion is not allowed
   to masquerade as modulo-cycle exposure parity.
3. Effective runtime `bitsandbytes` AdamW8bit decay, version, step semantics,
   and optimizer state receipt, plus MeZO hyperparameters/directions.
4. Frozen zero-update, correct-label and corrupted-label controls; adapter
   pre/post tensor hashes and independently verified fresh-process reloads.
5. Predeclared untouched evaluation under independent bank custody before
   claiming behavioral improvement. Neither equal clock nor matching toy CE
   proves model benefit by itself.

**No training, deployment, PR, merge, protected data, or host effect is
authorized by this artifact.**
