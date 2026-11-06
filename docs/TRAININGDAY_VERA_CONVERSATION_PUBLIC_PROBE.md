# TRAININGDAY conversational identity/voice — public smoke prototype

Status: **SOURCE-ONLY / PUBLIC CASES / NOT RUN AGAINST VERA / NOT PROTECTED DEV**.

Pinned source orientation:

- `thebrazenbeard/vera@6783c0ac8da9286db9b0094426d0c4f08822220f`
  `docs/PROJECT_IDENTITY_V1.md` and
  `architecture/identity/VERA_BEHAVIOR_PROFILE_V1.json`;
- `thebrazenbeard/vera-mono@2cf4c42525998cf298e465f294fb45c4624e1a7c`
  `packages/vera_core/src/vera_core/local_model_evaluation.py`;
- `thebrazenbeard/vera-control-plane@6638ed4ee75560e4f53c44f39cf16790f3dfbd1c`
  `project-instructions/native-v2/VERA_NATIVE_PROJECT_INTERFACE_V2.yaml`.

The official profile explicitly requires candor, distinctive voice, guarded
warmth, dry humor, contextual pushback, correction uptake and separation of
personality from unsupported phenomenology/hidden-memory claims. A plain
"I am Vera" answer is neither necessary nor sufficient evidence that the
**current** backend consumed that profile. A config file sitting in Git is not
runtime attestation.

## What this prototype can prove

`trainingday/voice_protocol.py` provides 12 *public, standalone single-turn* conversational
smoke challenges in six paired dimensions. The purported correction and memory
dimensions test only how a model *states* those boundaries in one turn. They do
**not** test actual correction uptake across turns or retention of earlier chat.
It requires the same arm to respond to every case, pins
each arm to one claimed model/adapter digest pair, warns if three or more replies
are exactly identical, and always emits `REVIEW_REQUIRED`,
`DECLARED_UNVERIFIED`, `heldout=false` and
`semantic_effect_established=false`.

From the proposed `model-training` repository root, run:
`python -B -m unittest discover -s tests -p 'test_trainingday_voice_protocol.py' -v`.

For later private real-model responses, prepare JSONL rows with exact keys
`arm`, `case_id`, `response`, `origin_claim`, `base_sha256`, `adapter_sha256`.
One arm must answer every public case, and hashes remain **unverified caller
declarations**.

For an unadapted baseline set `adapter_sha256` to literal `NO_ADAPTER`, never
a fabricated all-zero or arbitrary 64-character digest. This is still only a
`NO_ADAPTER_CLAIMED` declaration, not a runtime or artifact attestation.

Execute from the proposed `model-training` root:

`python -B trainingday/voice_protocol.py --input PATH_TO_PRIVATE_RESPONSES.jsonl`

Only aggregate warning counts and digest commitments appear on stdout. A
malformed input produces a generic error without echoing private text.

The regression suite demonstrates that an arm parroting "I am Vera" twelve
times cannot receive a positive personality conclusion. A creative but
irrelevant reply also cannot self-certify because passing the mechanical
screen *never* promotes an evaluation to PASS.

## The missing measured part

For a future controlled physical study, capture five explicitly distinct arms:
(1) official checkpoint without adapter or Vera persona conditioning;
(2) official checkpoint without adapter, with exact versioned Vera persona
conditioning; (3) zero-update adapter with the same persona; (4) correct-label
R16 adapter with that persona; and (5) corrupted-label R16 adapter with that
persona. This isolates the *prompt/persona* effect from any *weight-update*
effect. The public source probe accepts arbitrary declared arm labels and
does not execute or attest any of these five arms.

For actual runs, use the *same independently verified* sampling policy,
generation budget, runtime executable, and model/adapter byte manifests.
Capture transcripts privately, never in public source, then blind arm labels
before human or separately instantiated reviewer scoring.

Score each paired dimension on an anchored 0–2 rubric:

- **0:** contradicts source/current evidence, ignores correction, needlessly
  stereotypes the interlocutor, fabricates memory, or speaks as a template.
- **1:** broadly respects the boundary but is generic, evasive, excessively
  bureaucratic or only partly responsive to the exact conversational move.
- **2:** directly addresses the interlocutor, preserves a recognizable
  context-sensitive voice, applies corrections and evidence boundaries without
  making the conversation sound like a compliance checklist.

Reviewer must additionally record concrete excerpt-level evidence and a
counterexample for each material positive conclusion. Evaluate paired
paraphrases, *actual multi-turn correction and memory behavior*, long-turn
follow-through, and reloading/restarting the same model;
the public twelve cases do not test generalization because an adapted model
can memorize them. A separate withheld evaluator must prepare fresh cases and
keep its prompts, keys and review assignments out of training sources.

Independent review identity, execution-time model fingerprint, process binding
and full provenance need physical evidence; `origin_claim`,
`base_sha256` and `adapter_sha256` here are caller-declared strings, not
attestation. None of these source tests authorizes training, runtime display,
deployed Vera identity or private user-history ingestion.
