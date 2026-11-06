# R203 — original counterfactual stateful tool-use TRAIN candidate

**SOURCE ONLY / UNSIGNED / UNADMITTED / HARD HOLD — 2026-11-04.**

R203 is an explicitly **separate** proposed TRAIN corpus. It does not modify
the frozen 1,536-row R202 TRAIN SHA, R202 admission-request subject, R201
claims, private DEV, host state or any model weights. No rights signature,
protected evaluator, GPU lease or NAS witness has been issued for R203.

## Why this narrow prototype exists

The existing R202 tool-selection example is numerically correct but shallow:
a six-phrase lexical classifier chooses the tool class correctly in **256/256
public TRAIN cases**, without executing tools. A source reviewer also found
that a training-field-only heuristic predicts **187/256** R202 uncertainty
labels. Balanced class totals are not enough to measure transfer.

This **new original generator** makes each case a three-way contrast. The
same initial state, policy, tool catalog, and sentence structure appear in
all three versions. Only two update *numbers* can change:

1. SUM: after adding `A_delta` to sensor `A`, `A+B < floor`; call
   `sum_ints(updated_A,B)` and return its computed result.
2. SORT: alter only `A_delta`, so the sum reaches the floor; the rotated
   queue remains disordered, so call `sort_ints(updated_Q)`.
3. NO TOOL: keep the SORT sensor update, alter only `Q_rotate_left`; the
   queue becomes ascending, so return `no_tool` with empty args/null result.

All responses are exact compact JSON with `tool`, `args`, `result`. The source
uses deterministic SHA-seeded original numbers and three original prose
surfaces, public content-derived IDs and hash-scrambled output row ordering.
No Calc/Labyrinth source, private conversations, outside corpora or protected
DEV records are copied.

| Frozen source commitment | Value |
| --- | --- |
| Schema | `VERA_TRAININGDAY_R203_COUNTERFACTUAL_TOOLSTATE_V1` |
| Corpus | `R203_NEW_ORIGINAL_COUNTERFACTUAL_TOOLSTATE_TRAIN` |
| Contrast groups | 64 |
| Public TRAIN | 192 rows, exactly 64 each SUM/SORT/NO TOOL |
| TRAIN JSONL SHA256 | `ab04ddf9eb88aa0c809404aada24b5784c3c7de11fb3cd9cb62b0b5d4752d389` |
| DEV/evaluator | **Not generated or admitted** |

## Independent local source oracle and adversarial baselines

`tests/test_trainingday_r203_counterfactual_toolstate.py` parses each
public prompt using its own regular expression, applies the specified
state updates and independently computes the appropriate tool arguments and
result. It checks all 192 rows, all 64 minimal contrast triples, exact
frozen JSONL bytes, and the permanent hold flags. It also verifies that
SUM/SORT differs by one sensor-update integer and SORT/NO_TOOL by one
rotation integer. Source tests are **not** evaluator-custody proof.

| Source-only shortcut predictor | Correct / 192 | Accuracy |
| --- | ---: | ---: |
| Constant majority class | 64 | 33.3% |
| Former R202 keyword-only tool rule | 64 | 33.3% |
| Most common class by surface template | 64 | 33.3% |
| Most common class by raw queue-rotation value | 70 | 36.5% |
| Most common class by sign of `A_delta` | 73 | 38.0% |
| Independent state-update executor | 192 | 100.0% |

These are measurements on **PUBLIC TRAIN**, not protected DEV or model
performance. The exact state rule remains visible; a parser implementing that
rule is an intentional positive *task-specific* algorithm baseline. The
tests do not prove any model learned the rule.

Run the source-only checks without a model:

```bash
python -m trainingday.r203_counterfactual_toolstate
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v tests.test_trainingday_r203_counterfactual_toolstate
```

## Independent transfer evaluation must remain separate

To qualify R203 as **useful learning material**, a protected custodian should
design, commit, generate and score unseen variations **without sharing their
seed or answer key with the trainer**:

- New prose grammars that do not copy these three public surfaces.
- Different update operators (e.g. two adjustments, reverse rotation,
  insert/delete events), queue lengths and integer ranges.
- Counterfactual pairs where every keyword is preserved and only a numeric
  or availability fact changes the correct action.
- Boundaries `A+B == floor`, repeated queue elements, empty input and
  non-actionable tool requests; define exact handling before generation.
- Separate metrics for tool-class accuracy, exact args, tool-result validity,
  and abstention, with an executor and counterfactual-pair consistency rate.
- Zero-step, corrupted-label and base-model controls at matched compute;
  repeated seeds and separate-process reload scoring.

**Hostile limits:** The public source still encodes one fixed precedence
rule, only three tool types and three wording families. A model may memorize
this recipe or the finite 192 public rows. No zero-leakage claim follows from
the hash-derived IDs or reordered rows. Passing source tests cannot elevate
R203 to broad tool-use or reasoning generalization.

> **Stop condition:** R203 is a *candidate training source*, not a training
> grant. The R202 rights claim and private R201 evaluator do not cover it.
> Exact R203 rights/provenance, exclusions, independently held DEV, tokenizer
> fit, host lease, and independent NAS recovery must be qualified on their own.
