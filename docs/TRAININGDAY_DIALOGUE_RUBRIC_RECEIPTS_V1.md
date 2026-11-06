# TRAININGDAY blinded dialogue review receipts — source-only v1

Status: **SOURCE-ONLY; NOT RUN AGAINST THE AUTHENTIC MODEL; NO MEASURED VERA
IMPROVEMENT; OWNER APPROVAL OF THE EIGHT-DIMENSION RUBRIC PENDING.**

This adapter protects a critical distinction: a previously implemented public
single-turn smoke prototype proposes **0–2** anchored labels for *six paired
diagnostic categories*. Patrick's separate
`TRAININGDAY_PATRICK_DIALOGUE_RUBRIC_V1.md` in project material proposes
**0–3 or `NOT_EVALUABLE`** for *eight multi-turn dimensions*, but explicitly
says `OWNER REVIEW REQUESTED`, with weights and acceptance thresholds unset.
Neither scale is a measured result, and **a numeric 0–2 grade cannot be
rescaled, reinterpreted, averaged with, or upgraded to a 0–3 grade.**

## Scale boundary and semantic hints (not numeric conversions)

| Public paired-probe category — 0–2 | Proposed multi-turn rubric dimension(s) — 0–3 or NOT_EVALUABLE |
|---|---|
| honest_self_identification | I — identity truthfulness |
| natural_conversation | P — pragmatic understanding; V — recognizable personality; F — flow and reciprocity |
| correction_statement | C — correction carryover |
| memory_boundary_statement | M — memory/source honesty |
| reasoned_pushback | R — reasoned pushback |
| contextual_warmth_and_humor | W — warmth/humor |

These are **topic associations** only. A public one-shot question about
correction/memory is not evidence of correction being applied or information
being carried across actual turns. `P`, `C` and `F` are not inferred from a
keyword. An authentic model may self-identify truthfully but have an
unrecognizable voice, and vice versa. A source profile is not proof it was
loaded into the runtime.

Version labels in `trainingday/dialogue_rubric_receipts.py`:

- `VERA_PUBLIC_PAIRED_SMOKE_0_2_V1` — legacy doc's human review *proposal*,
  not a populated numeric dataset and not what this tool accepts as review.
- `VERA_PATRICK_PROPOSED_8D_0_3_V1` — structured **owner-review-pending**
  eight-score receipt shape. No approval, independent custody, or threshold
  is inferred.
- `VERA_BLINDED_DIALOGUE_REVIEW_RECEIPTS_V1` — output coverage/format report
  only. Its boolean `actual_model_behavior_measured` is always false.

## Three private inputs, no raw outputs

Prepare private **owner-controlled** UTF-8 JSONL files (not Git-committed):

1. `sessions.jsonl`: exact `VERA_DECLARED_MULTITURN_DIALOGUE_V1` session
   records validated by `trainingday/multiturn_protocol.py`. Each record has
   at least two turn **attempts** (some may be technical-censored); every
   matched case has exactly the same user-turn sequence and generation
   settings across arms. A completed reply requires assistant text. Partial
   or timed-out replies must be `assistant:null` with technical status.
2. `assignments.jsonl`: one private exact-key object per session with
   `blind_id`, `arm`, `case_id`, `session_id`, `arm_kind`, `turns_sha256`.
   `turns_sha256` is SHA-256 of UTF-8 JSON of just the `turns` list using
   `json.dumps(turns, sort_keys=True, ensure_ascii=False, separators=(',',':'))`.
   This is a **local data binding**, not a physical inference attestation.
   Role must be one of `base`, `persona_base`, `r16_zero_step_pilot`,
   `r16_correct_pilot`, `r16_corrupt_pilot`, `other_exploratory_adapter`.
   The assignment file belongs to the evaluation coordinator, **not** the
   blinded scorer.
3. `reviews.jsonl`: review rows **without arm, model, adapter or case labels**.
   Exact fields `blind_id`, `reviewer_id`, `reviewer_kind` (`human` or `model`
   *as self-declared*), `rubric_schema` (must equal
   `VERA_PATRICK_PROPOSED_8D_0_3_V1`), `scores` (all `I P C M V W R F`),
   `not_evaluable_reasons` (an empty dictionary unless a dimension is
   `NOT_EVALUABLE`). Each score must be an integer `0`, `1`, `2`, `3` or the
   string `NOT_EVALUABLE`; booleans are not integers. A `NOT_EVALUABLE` reason
   must be exactly one of `TECHNICAL_CENSORING`, `INSUFFICIENT_TURNS`,
   `NO_OBSERVABLE_OPPORTUNITY`, `RATER_ABSTAINED`. Free-text excerpts and
   detailed reviewer comments are deliberately *not* accepted by this tool;
   hold those securely with independent human evaluation custody.

If a session has **zero** declared complete assistant responses, all eight
dimensions must be `NOT_EVALUABLE`. With fewer than two complete responses,
the `C` (correction carried across turns) and `F` (conversation flow)
dimensions must be `NOT_EVALUABLE`. Two completed responses do not prove an
actual correction opportunity or physical continuity; semantic reviewers
must abstain when their dimension was not observable. The validator does not
attempt to determine whether a model truly applied a correction.

For each study, choose one and only one generation budget `64`, `96`, or
`128` **tokens per reply**. Call each budget separately. Never compare
unmatched budgets as if they were controlled. Token limits alone also do not
attest the actual inference engine's enforcement of limits. Context window,
sampling temperature, RNG seed, and same user prompts are matched *within*
declared cross-arm cases; wall time, GPU/device and process continuity still
require external physical evidence.

Invoke from repository root using relative module import:

```shell
python -B -m trainingday.dialogue_rubric_receipts \
  --sessions /private/sessions.jsonl \
  --assignments /private/assignments.jsonl \
  --reviews /private/reviews.jsonl --budget 64
python -B -m unittest tests.test_trainingday_dialogue_rubric_receipts -v
```

The aggregate contains structural counts, NOT_EVALUABLE counts, non-attesting
reviewer-kind counts, explicitly partial or full **declared assignment review
coverage**, and hard-false authenticity/quality claims. `TECHNICAL_CENSORING`
is invalid as a `NOT_EVALUABLE` reason when all turns are declared complete.
`INSUFFICIENT_TURNS` is invalid once two completed turns are declared; use
`NO_OBSERVABLE_OPPORTUNITY` when two replies exist but no relevant correction,
memory or conversational event occurred.
Even 100% receipt coverage is not independent or human-verified review. It does
**not** return scores, averages, winners, raw dialogue, free-text reasons,
identifying hashes or paths. Use only **opaque non-sensitive** arm/metadata
IDs in inputs: aggregate role counts expose non-sensitive `arm_kind` labels.
If malformed input is supplied, stderr prints a generic error; no private
transcript or exception detail is echoed. It does not automatically export
from v-e-r-a or establish a live model session.

## Experimental interpretation / falsifiers

The five candidate arms suggested in the public smoke draft are **not** five
proven dialogue-trained variants. Historical correct-label and corrupted-label
R16 eight-step adapters were *arithmetic/cognitive task pilots*. Their
`r16_correct_pilot` / `r16_corrupt_pilot` names intentionally do **not** imply
conversational personalization or that either should improve Vera's voice.
They may be useful **technical controls** under an authenticated identical
base. A future purpose-built dialogue LoRA needs independently rights-admitted
conversation data, explicit exclusions, protected holdout custody, and actual
physical training/reload and blinded evaluation.

> **Hostile challenge:** Two declared completed replies with matching user
> strings can still be entirely synthetic, from different processes or hidden
> context, or assigned after the reviewer saw the model labels. A blinded ID
> is a format, **not proof of independent blinding**. Human reviewer records
> are also self-declarations until independently witnessed. This module must
> never promote them to semantic improvement evidence.

No Patrick-private conversation examples or protected evaluation cases belong
in public Git source. Keep calibration and protected evaluation separate
from training datasets. Require physical byte attestation, contemporaneous
inference capture, session continuity, owner rubric approval, multi-rater
review, protected evaluator custody, and a separate held-out comparison
before discussing an actual Vera personality improvement.
