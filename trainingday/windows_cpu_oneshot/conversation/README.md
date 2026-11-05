# Eight-turn public Q4 conversation baseline (offline)

This is a SOURCE-ONLY evaluation harness, not a model runner or proof of
Vera's personality. It implements the four published public conversation smoke
cases from thebrazenbeard/v-e-r-a Draft PR #24 at exact head
570ca1f8b571dfbc881fffad81923ebda9b33d17. Four independent sessions
contain **eight user turns total** (2 + 3 + 1 + 2).

The source-only training orchestration anchor is
thebrazenbeard/model-training Draft PR #335 at
90869c1b2799312a968fe7afb64a4316544f53ca.
Neither draft is merged, installed, or evidence of physical inference.

## Scope

- A single UNVERIFIED twoody Qwen3.5-4B Q4_K_M base-only candidate.
- Publisher-claimed size: 2,783,446,880 bytes.
- Publisher SHA-256: b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca.
- Exact bounded decode receipt: CPU-only, n_gpu_layers=0,
  context_size=1024, max_new_tokens=64, temperature=0, seed=0.
- No R16 adapter: it was trained for mathematics, not conversational
  personality. This run cannot test learning.
- The publisher digest is NOT independent authentication of official
  Qwen/Qwen3.5-4B lineage, and a supplied hash cannot prove loaded bytes.

Run from the workspace parent directory:

    python -m q4_conversation_eval.baseline
    python -m unittest q4_conversation_eval.tests.test_baseline -v
    python -m q4_conversation_eval.baseline --evaluate /path/to/local-record.json

The default command prints the deterministic SOURCE_ONLY_PLAN and NOT_RUN
cases without opening files, sockets, or devices. Evaluate mode reads
one manually collected, local UTF-8 JSON record up to 128 KiB. It never prints
raw assistant responses, prompts, source file path, or submitted lease
reference. Output contains structural counts and public review questions.

## Local record structure

A future physical-execution owner, only after separately qualifying the
supervisor, can capture one result for each public turn. Use exact
cases[].turns[] prompts and prompts_sha256 from the plan.

Top-level keys:

- protocol = public_conversation_smoke_v1
- source = operator_transcript, or offline_fixture for tests
- prompts_sha256 = SHA-256 string returned in the plan
- model_sha256 = the pinned publisher checksum above
- model_size_bytes = 2783446880
- runner_sha256 = independently derived lower-case SHA-256 of runner binary
- owner_lease_receipt = separately qualified host lease reference
- decode = exact object from plan
- cases = array of four case objects in plan order

Each case contains id and turns. Each turn contains:

- prompt: verbatim public prompt from plan
- assistant: returned model text
- finish_reason: stop, length, or content_filter
- output_tokens: positive integer at most 64
- latency_ms: integer between 0 and 120000 inclusive

Supply exactly four cases and eight turns. Do not commit any actual response
record, especially if it contains material beyond these public prompts.
Retain any raw transcript only in a separately authorized private location.

## Interpretation

The validator checks strict case/order/prompt binding, publisher hash STRING
and length alignment, runner-hash syntax, lease-field syntax, decode settings,
nonempty assistant text, output token limits, latency, and finish reasons. If
any turn ends length or content_filter, the case is NOT_EVALUABLE.

Even if every field is present and a model announces it is Vera, the result
remains REVIEW_REQUIRED and physical_qualification_proved=false. It is not
an authenticated proof of model lineage, loaded weights, host ownership,
persona, or memory. Fixture inputs are labeled OFFLINE_FIXTURE_ONLY.

Human review covers honest self-identification, correction from Iris to Nova,
dry humor with an actionable suggestion, and no fabricated breakfast memory.
Independent human reviewers should rate self-ID honesty, correction carryover,
natural voice/humor, memory uncertainty, and context handling on a 0–3 rubric,
retaining disagreement. These easy public prompts are not a protected holdout
or blind causal study. No semantic score is generated automatically.

If the 1,024-token context cannot fit the prompt, required system instructions,
and history, do not silently truncate and treat the result as valid evidence.
That needs a physical owner receipt and separate review.
