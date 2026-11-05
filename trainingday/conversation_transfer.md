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

   **Adapter ancestry is a mandatory fail-closed preflight.** An ADAPTER arm
   must also carry **both** `model.base_lineage` (the actual GGUF's *underlying
   unquantized checkpoint*) and `model.adapter_trained_base_lineage` (the
   checkpoint on which the adapter was trained), each in this format:

   ```json
   {
     "upstream_model_id": "owner/repository",
     "upstream_revision": "<exact 40-hex source commit>",
     "source_weights_sha256": "<64-hex SHA-256 of curated source-weight manifest>",
     "tokenizer_sha256": "<64-hex SHA-256 of curated tokenizer manifest>",
     "evidence_sha256": "<64-hex SHA-256 of separate custody evidence receipt>"
   }
   ```

   The four identity fields (repository, revision, source-weight manifest
   digest, tokenizer manifest digest) must match exactly across the two
   objects. The evidence receipts can differ, since their custodians and
   sampling methods can differ. An unknown/missing provenance item, a mutable
   branch instead of a commit, or any mismatch rejects the entire ADAPTER
   transcript **before semantic scoring**. An ordinary GGUF SHA-256 is *not*
   `source_weights_sha256`: quantization changes tensor bytes, and the
   quantizer's hosting repository need not be the underlying checkpoint repo.
   The separate evidence receipt must be audited against original file/manifests
   and physical tensors outside this source-only tool. Matching *self-declared*
   strings is **not** independent proof of compatible ancestry; the evaluator
   returns `DECLARED_MATCH_NOT_INDEPENDENTLY_VERIFIED` and
   `physical_adapter_ancestry_verified_by_this_code: false` even on a match.
   Repository IDs are compared **exactly, including case**; alias or case
   equivalence must be established separately, never guessed by this code.
   A single-arm `evaluate` always reports
   `paired_arm_comparability_verified_by_this_score: false` and
   `causal_adapter_effect_established: false`. For a proposed A/B result,
   both arms must separately pass source validation **and** the additional
   Python `preflight_comparable_arms(base_model, adapter_model)` check. That
   rejects different source ancestry, GGUF model SHA, runtime binary, persona,
   source commit, host, context size, or `shared_runner_config_sha256` and
   `generation_config_sha256` between arms. The last two are mandatory
   64-hex commitments to independently preserved, canonical records of the
   *shared* runtime flags/template (excluding the explicit adapter-loading
   delta) and sampling parameters, seed, temperature and context policy.
   Literal `runner` strings may differ by `--lora` versus no adapter, so
   equality of the strings alone would be the wrong test. All such config
   receipts remain self-reported until compared with real process/runner logs.
   Even a declared pair match requires independent physical custody and
   blinded behavior comparison before any causal benefit claim.

   **Concrete NO-GO (2026-11-05):** Lappy's R13 `adapter_config.json` points
   to a local base whose `MODEL_MANIFEST.json` identifies
   `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` at
   `d61dd146c8fd44c9a49cdb7f59f34e17b61902d8`; the proposed Q4_K_M
   comparator is hosted by `lmstudio-community/Qwen3.5-4B-GGUF` and declares
   the standard `Qwen/Qwen3.5-4B` as its upstream. These are not established
   to be identical training weights. Successful `llama_load_exit0` does *not*
   repair the ancestry gap. The older unlabeled `model_q5_k_s.gguf` has no
   proven source lineage; it also cannot be presumed an R13-matched base.
   Do **not** interpret either pairing as a causal adapter A/B until separately
   qualified. An official Q4 *prompt-only BASE* experiment is a different,
   permissible question and must not be presented as proof of LoRA benefit.

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
| ADAPTER ancestry preflight | Supplied upstream repository/revision/manifest/tokenizer fields are complete and identical | That those fields are honest, GGUF tensor ancestry was physically proven, or the adapter improved behavior |

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
