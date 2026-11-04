# TRAININGDAY R202 — original multi-domain TRAIN candidate

**Status: SOURCE ONLY / UNAUTHENTICATED HOLD.** This is not a training-data
grant, protected evaluation, authorization to use official Qwen weights, or
evidence that Vera has learned any of these tasks.

## Why R202 exists

R201 supplies 1,024 reproducible Calc-backed cubic-derivative problems.
Derivatives alone cannot teach or test every ability Patrick wants. R202 is a
**separate, original TRAIN-only source candidate** with six deterministic,
machine-checked task families:

| Family | Rows | Narrow capability and independent test |
| --- | ---: | --- |
| `math_logic` | 256 | Divisibility, parity, or inequality after integer operations; recomputed from prompt |
| `language_tracking` | 256 | Follow ownership through two natural-language transfers; test replays each transfer |
| `tool_selection` | 256 | Select typed integer tool, JSON arguments, or abstain when no weather tool exists; test parses intent and arguments |
| `dependency_planning` | 256 | Order a five-task DAG by available shortest or longest duration with alphabetical tie-break; test independently validates legal predecessors and greedy choice |
| `error_correction` | 256 | Find first wrong intermediate arithmetic value under two operator orderings; test recalculates every step |
| `evidence_uncertainty` | 256 | Three-valued decisions with missing, failed and satisfied conditions; test independently applies open-world decision rule |

Total **1,536 TRAIN rows**, not 1,536 independent reasoning demonstrations.
The text and numeric examples are original to the generator in
`trainingday/cognitive_curriculum.py`; no excerpts of `calc`, `labyrinth`,
private chat exports, books, external language examples, or protected DEV data
are copied. Calc and Labyrinth influenced the *choice of task domains*, not
the implementation. Their repository roots have no verified `LICENSE` files
at the inspected `calc@17e12d0f659b42732fb98e5730732aba9ff1b488` and
`labyrinth@b33a517daa47c13f8692a898cd277460dfca968a`. Do not infer
rights to train on their source or redistribute it.

## Reproduce and freeze the public source

Python 3.10+ standard library only. From this repository:

```bash
python -m trainingday.cognitive_curriculum
python -m unittest -v tests.test_trainingday_cognitive_curriculum
```

The CLI prints an integrity/authorization-hold manifest; it does not reveal
private rows, contact an API, download models, or launch training. The
`train_rows()` function returns deterministic `{id,prompt,answer,split}`
objects; `train_bytes()` emits canonical JSONL bytes in frozen family/index
order. This output SHA-256 pins the exact generated TRAIN data:

```text
schema=VERA_TRAININGDAY_R202_COGNITIVE_CURRICULUM_V1
corpus_id=R202_NEW_ORIGINAL_COGNITIVE_TRAIN
train_count=1536
train_sha256=771471a6524410a99b14e8988ef6e0cdd7515a5756c4376a07319b439f58b80a
private_dev_included=false
authorizes_training=false
```

The test suite independently reconstructs each answer using only each row's
public prompt, verifies row uniqueness, structural/answer-class coverage and
an exact corpus commitment, rejects invalid generator indices, and checks
that every authority flag remains false. It is a SOURCE test, not model
evaluation or host attestation.

### Cheap shortcut baselines / source bias checks

Raw deterministic random sampling initially produced a **FAIL for curriculum
adequacy**: 194/256 divisibility answers were NO, while 178/256 access
decisions were REJECT. A constant-majority prediction would score 75.8% and
69.5% on those source rows without solving any task. This does not prove the
model would exploit those shortcuts, but it is a clear, measurable training
bias. R202 now applies a bounded, reproducible per-class rejection sampler:

| Family | Frozen answer counts | Constant-class TRAIN baseline |
| --- | --- | ---: |
| `math_logic` | YES 128 / NO 128 | 50.0% |
| `evidence_uncertainty` | ACCEPT 86 / REJECT 85 / INSUFFICIENT 85 | 33.6% |
| `tool_selection` | Three supported/no-tool intents, each below 42% of rows | Below 42% |

The other families contain structured output targets rather than small
enumerated answer classes; their answer frequencies alone are not a meaningful
measure of shortcut resistance. The generator checks distinct prompt surfaces
and tests structural varieties but does **not** certify semantic independence.
Labels are generated after public deterministic rejection, not by protected
private evaluator sampling. The public class stratification and exact TRAIN
SHA can still be memorized and must not be used as hidden qualification data.

## Practical rights admission (owner and steward may be the same person)

It is unnecessary to demand an unrelated third-party human as rights-holder
for genuinely owner-created material. But an asserted name alone does not
authenticate the owner, permission, or exclusions. Before actual physical
training, the rights process needs:

1. Authenticated rights-holder/steward identity and evidence of authorship or
   a valid license to each exact included source. A single person may fill
   both roles with that authority.
2. A signed purpose binding to `Qwen/Qwen3.5-4B`, the exact checkpoint
   revision, this schema, source commit and corpus SHA-256, with an explicit
   grant for intended noncommercial Vera training.
3. Explicit exclusions (private chats, third-party documents, protected DEV,
   personal or confidential text) and an auditable source manifest.
4. A separate evaluator custodian with private DEV/test seeds and answers,
   immutable pre-experiment commitments, and controlled one-time scoring.
5. Genuine host-owned process/GPU lease, model/tokenizer verification and
   physical-run receipts. Source signatures alone cannot satisfy these.

Existing Draft #265 provides one source-only Ed25519 claim *candidate* for
R201; it does not enroll a real owner key or grant anything. R202 should be
included in that discipline with its **own** frozen source identity, not
declared a replacement for R201 or given a synthetic grant.

## Protected evaluation boundary

Do **not** produce protected DEV rows or a private nonce on a public branch.
A separately operated evaluator should build task-family *structural* holdouts
whose parameter ranges, language patterns and graph topologies differ from
published TRAIN. It should commit to the test set **before** observing model
responses and preserve the secret set off public GitHub. The evaluator may
expose only family-level accuracy, exact-format adherence, failure taxonomy,
and blinded commitments until scoring is complete.

Exact string disjointness alone is not enough: the same math, transfer,
execution graph, or three-valued record can be paraphrased. Compare normalized
semantic structures, template families, task operators, hidden seeds, and
duplicate answers where appropriate. Reuse of public R201 or R202 examples as
hidden grading data invalidates the independence claim.

## Negative findings / limits

- These examples are structurally templated (two or three forms for some
  families). High performance can be template recognition rather than
  generalization, and 256 samples per family do not amount to a broad,
  human-reviewed curriculum. Even balanced classes have trivial structural
  shortcuts; a blinded, distribution-shifted evaluator must challenge them.
- The task set omits long-context discourse, robust code execution, real tool
  side effects, adversarial dialogue, and genuinely open-ended writing.
- Logged error-correction steps are **independent checks against intended
  correct prior values**. Therefore a later displayed entry can be correct
  even when an earlier entry is wrong; these are not a dependent calculation
  trace. Tests independently validate that exactly one displayed entry differs.
- No Qwen tokenizer 64-token fit, contamination-free private DEV, independent
  oracle custody, copyright permission record, owner signature, authenticated
  base checkpoint, actual optimizer training or inference result has been
  established by this source package.
- The original-source claim is about the newly written templates here, not
  legal assurance about third-party contributions or a production grant.

**Promotion rule:** Hold until actual rights, evaluator, model, host and
recovery evidence are independently qualified. Passing eight source unit
tests by itself never promotes the corpus.
