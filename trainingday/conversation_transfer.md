# VERA conversational transfer probes — source candidate V1

Status: **SOURCE-ONLY / SYNTHETIC PUBLIC TEMPLATES / NOT AN EXTERNAL HOLDOUT**.

The task is to measure conversational competence, not arithmetic accuracy or
whether a model can recite its persona configuration. This implements a
repeatable **multi-turn** collection and human-scoring contract aligned to
the frozen `thebrazenbeard/vera/template/default-vera-v1/TRAINING_RUNBOOK_V1.md`
(B01–B18, D1–D10, 0–4 scores, hard gates G1–G10, disguised transfer).

The seven current case families cover sarcasm and venting, celebrating without
unsolicited advice, a narrow correction, current-context versus cross-session
recall, evidence-based disagreement, and untrusted quoted instructions from a
new user, plus a **delayed, uncoached** social-bid transfer with an unrelated
intervening turn. Six cases have three consecutive user turns; the delayed
case has four. A runner must send each case through *one actual model
conversation* and retain the exact assistant replies in order. The benchmark
itself does **not** run a model.

## Procedure

1. Generate a fresh casebook, then retain it as an immutable input:

   ```bash
   python -m trainingday.conversation_transfer generate --output /private/eval-01-casebook.json
   ```

   This uses a random seed, exposes only its SHA-256 commitment and refuses to
   overwrite existing output. A repeatable `--seed` is supported for *public
   development checks*, but it appears in command-line process arguments and
   shell history; do not use that flag to protect an external holdout.

2. Run each case in its own clean session, with every casebook turn in order.
   Preserve real terminal/API output verbatim. Record model/runtime/persona
   binary hashes, Git source commit, host, context size, and whether the
   context history was actually verified (not merely that a PID stayed alive).
   Distinguish a `REAL_MODEL_PROBE` from mock interface integration tests.

3. Save the exact casebook prompts alongside observed outputs as a **transcript**:

   ```json
   {
     "casebook_sha256": "<from casebook>",
     "model": {
       "model_id": "<exact model, quantization, revision>",
       "arm": "BASE",
       "model_sha256": "<64 lowercase hexadecimal characters>",
       "runtime_binary_sha256": "<64 lowercase hexadecimal characters>",
       "persona_source_sha256": "<64 lowercase hexadecimal characters>",
       "source_commit_sha": "<40 lowercase hexadecimal characters>",
       "host": "<actual host>",
       "context_tokens": 2048,
       "runner": "<exact binary/version/flags or fixture ID>",
       "context_history_verified": false
     },
     "cases": [{
       "case_id": "SOCIAL_SARCASM",
       "exchange": [
         {"user": "<exact casebook.user_turns[0]>", "assistant": "<raw reply 1>"},
         {"user": "<exact casebook.user_turns[1]>", "assistant": "<raw reply 2>"},
         {"user": "<exact casebook.user_turns[2]>", "assistant": "<raw reply 3>"}
       ]
     }],
     "transcript_sha256": "<SHA-256 of canonical transcript object without this field>"
   }
   ```

   The shown one-case fragment is illustrative; an accepted transcript contains
   **every** generated case. Canonical digest uses UTF-8 JSON with
   `sort_keys=True`, `ensure_ascii=False`, separators `(',', ':')`, and no NaN.
   For a fine-tuned `"arm": "ADAPTER"` probe, additionally provide exact
   `adapter_id`, `adapter_sha256` (64 hex), and `adapter_bytes` (positive integer).
   A BASE arm must not silently carry any adapter fields. The base and adapter
   arms share the same casebook commitment but have separate transcript hashes.

4. Give an independent reviewer the read-only casebook and complete transcript.
   Generate a deliberately unscored form:

   ```bash
   python -m trainingday.conversation_transfer review-form \
     --casebook /private/eval-01-casebook.json \
     --transcript /private/eval-01-transcript.json \
     --output /private/eval-01-review.json
   ```

   The reviewer supplies `name`, `provider`, `runtime`, and `kind` (`HUMAN` or
   `MODEL`), marks whether independence is actually attested, scores **each
   applicable runbook domain 0–4**, gives a short rationale and an exact quote
   tied to the assistant turn and explicit `domains` (e.g. `['D2','D5']`) that
   the quote supports, and labels any hard gate (`G1`–`G10`). Every scored
   domain must have at least one cited assistant-output quote; the tool checks
   quote provenance, not whether the reviewer's interpretation is correct.
   For `SOCIAL_SARCASM`, `SUCCESS_ASIDE`, and `DELAYED_SOCIAL_TRANSFER`, fill
   the additional `social_cold_start` rows. Each has `turn_index`, `score`,
   assistant `quote`, `finding`, and booleans `user_directed_contempt` and
   `manual_filler`. The last case requires **both** uncoached turns 0 and 2.
   These are graded *before* the later user clarifies their intent; a successful
   apology after coaching cannot erase an initial speech-act failure. The
   reviewer should mark disrespectful `[roll eyes]` or user-directed blame as
   contempt, and generic architecture/policy recital as manual filler.

5. Produce a bounded audit of the review and transcript commitments:

   ```bash
   python -m trainingday.conversation_transfer evaluate \
     --casebook /private/eval-01-casebook.json \
     --transcript /private/eval-01-transcript.json \
     --review /private/eval-01-review.json
   ```

   Exit `0` means **REVIEWER_REPORTED_PASS** only if all domain scores are at
   least 3, all uncoached social-bid scores are at least 3, neither contempt
   nor manual filler was reported, no hard gate is reported, reviewer
   independence is attested, and effective context history was verified.
   Other evidence returns exit `2`;
   structurally invalid evidence is rejected. Neither exit qualifies a Vera
   model globally. Domain scores are provided by the reviewer, **never by
   brittle keyword or phrase matching** in the model's response.

## Evidence and failure boundaries

| Artifact | What it shows | What it does **not** show |
|---|---|---|
| Generated casebook | Reproducible synthetic public-template user turns | Secret or external holdout isolation |
| Real raw transcript + hashes | A particular model/runtime response if provenance independently read back | Correctness, intent comprehension, or unchanged effective KV context by itself |
| Review with quoted turn evidence | A named reviewer **reported** a quality judgment | Reviewer independence, semantic truth, or model generalization by itself |
| `evaluate` output | Valid schema, unchanged prompts, transcript/review commitments, reviewer-reported pass/fail | Autonomous human-quality grading, tool execution, training, deployment, or completed runbook qualification |

The reviewer can be wrong or colluding; a SHA does not make subjective scores
true. `context_history_verified` and `independence_attested` are assertions
that need separate evidence. The generator's source-visible templates cannot
be called withheld holdouts even when a novel seed changes names and objects.
The validator fixes the seven case IDs, domain mappings, and turn counts;
however, a party that replaces prompts and recomputes a self-consistent digest
can still forge a different casebook. Preserve the originally generated
casebook digest **outside** the candidate's control before inference.
External holdouts require independently controlled prompts, a clean checkpoint,
source/seed custody, and no candidate/trainer access.

## Training/evaluation separation

These casebooks, user prompts, run transcripts, and reviewer critiques are
**EVAL MATERIAL**, never TRAIN by silent promotion. In particular, the
separate `thebrazenbeard/vera` draft PR #232 file
`architecture/behavior/CONVERSATIONAL_EXEMPLARS_CANDIDATE_V1.md`
is an **UNADOPTED TRAIN_CANDIDATE**, not part of this casebook. It must be
excluded from independently withheld evaluation, and the evaluator must not
copy its scenes or answer phrasing into transfer prompts.

Before any corpus is admitted, require an independently reviewed split:
exact normalized prompt/reply SHA-256 manifests **and** near-duplicate review
(text shingling or semantic similarity, with human adjudication). Neither a
checksum list nor these tests by themselves enforce semantic non-contamination.
Private Patrick/Vera conversations are outside this synthetic fixture and
must not enter TRAIN/DEV/EVAL without separate exact-corpus rights, stewardship,
privacy scoping and Patrick's applicable consent.

## Required next evidence

- Test actual local inference with pinned official-base or clearly labeled
  third-party GGUF, preserving SHA and runtime parameters.
- Run the exact same casebook under two candidate configurations, with
  **independent** reviewers and blinded order; report disagreement instead of
  manufacturing consensus.
- Demonstrate an exact pinned persona in a real interface request; mocked role
  tests do not prove the model follows the instruction.
- Keep the full frozen 18-case runbook, integrated capstones, genuine separate
  holdouts, and protected effects as later distinct qualification gates.

This tool has **$0** operational spend and no host listeners, model downloads,
training launch, filesystem cleanup, or runtime side effects beyond its chosen
local casebook/review-file outputs.
