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
| `math_logic` | 256 | Sequence of integer operations and divisibility; arithmetic recomputed from prompt |
| `language_tracking` | 256 | Follow ownership through two natural-language transfers; test replays each transfer |
| `tool_selection` | 256 | Select typed integer tool, JSON arguments, or abstain when no weather tool exists; test parses intent and arguments |
| `dependency_planning` | 256 | Order a five-task DAG by available shortest duration and alphabetical tie-break; test independently validates legal predecessors and greedy choice |
| `error_correction` | 256 | Find first wrong intermediate arithmetic value and correct it; test recalculates every step |
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
train_sha256=e95563a1672d531b884b5b6003f4f38412e42a09e7aa4685adec50338c294bd1
private_dev_included=false
authorizes_training=false
```

The test suite independently reconstructs each answer using only each row's
public prompt, verifies row uniqueness and an exact corpus commitment,
rejects invalid generator indices, and checks that every authority flag
remains false. It is a SOURCE test, not model evaluation or host attestation.

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

- These examples are structurally templated. High performance can be template
  recognition rather than generalization, and 256 variants per family are not
  a substitute for a broad, human-reviewed curriculum.
- The task set omits long-context discourse, robust code execution, real tool
  side effects, adversarial dialogue, and genuinely open-ended writing.
- No Qwen tokenizer 64-token fit, contamination-free private DEV, independent
  oracle custody, copyright permission record, owner signature, authenticated
  base checkpoint, actual optimizer training or inference result has been
  established by this source package.
- The original-source claim is about the newly written templates here, not
  legal assurance about third-party contributions or a production grant.

**Promotion rule:** Hold until actual rights, evaluator, model, host and
recovery evidence are independently qualified. Passing eight source unit
tests by itself never promotes the corpus.
