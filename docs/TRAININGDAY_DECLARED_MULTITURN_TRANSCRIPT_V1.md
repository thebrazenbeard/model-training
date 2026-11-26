# TRAININGDAY — declared multi-turn conversation transcript V1

Status: **SOURCE-ONLY / REVIEW_REQUIRED**, not a completed Vera evaluation.

This companion to the existing public, one-turn smoke probe validates the
*structure and declared comparability* of recorded multi-turn conversations.
Patrick's required outcome is an honest self-identity and recognizable,
responsive, naturally conversational Vera—not an arithmetic demonstration.

## Hard evidence boundaries

- A successfully parsed JSONL transcript is **not proof that a model ran**.
  Its base, adapter, persona prompt, process and source claims remain
  unauthenticated. No particular trained adapter is certified by this tool.
- Two or more user/assistant attempts under one declared `session_id` establish
  **multiple-attempt input format**, not actual server session continuity,
  completed exchanges, memory retention or conversational competence. A record
  containing only two timeouts is still a valid attempted-turn structure, not
  an observed conversation. Separate host/process witnesses are necessary for
  a physical continuity claim.
- No programmatic sentence matching can establish that Vera sounds natural,
  remembers a correction, responds with appropriate humor, respects memory
  boundaries or retains Patrick's preferred personality. **Human blinded review
  remains mandatory.** A model repeating “I am Vera” is insufficient.
- `source_claim=private_unverified` is a *claim*, not protected held-out
  custody; no private test bank is created by this change. Exposed W40/W42
  fridge/coffee/printer examples and the twelve earlier public smoke prompts
  must not enter independent protected evaluation.
- The executable never writes or emits raw transcripts, private user turns or
  model replies. Input must stay owner-local and excluded from Git/CI. Do not
  paste user trauma, credentials, or personal conversations into fixtures.

## JSONL contract

One strict JSON object per declared session. Every object has exactly these
fields, with **no permissive extras**:

| Field | Required value |
| --- | --- |
| `arm`, `case_id`, `session_id` | Slug identifiers (`[a-z][a-z0-9_-]{0,79}`), unique session and arm/case |
| `source_claim` | `mock_test`, `public_exposed`, or `private_unverified`; never automatically promoted to verified |
| `origin_claim` | Nonempty operator-supplied identifier; not authenticated |
| `base_sha256` | Lowercase 64-hex **declared** digest |
| `adapter_sha256` | Lowercase 64-hex declared digest or `NO_ADAPTER` |
| `prompt_sha256` | Lowercase 64-hex declared digest of conversation/persona prompt |
| `generation` | Exact integer `context_window_tokens`, `max_reply_tokens`, `temperature_milli` and `rng_seed` settings |
| `turns` | Two to sixteen user/assistant attempts in original order |

Every attempt has exactly `user`, `assistant`, `outcome`. A `complete` outcome
requires bounded nonempty assistant text. For a `truncated`, `timeout`, or
`upstream_error` outcome, `assistant` must be JSON `null`; a partial response
must never enter the transcript as if it were completed. A case is only
*declared structurally comparable* when every arm has that case, the complete
fictional user-turn sequence is identical, and the declared generation
settings and source class match. Prompt and adapter hashes are deliberately
allowed to differ among arms, so a prompt-only condition can be compared
with a trained-adapter condition without pretending the prompts are identical.

The small unit-test specimens are explicitly synthetic **format fixtures**, not
live model conversations, and cannot be used as protected evaluation items.

## Usage and interpretation

```bash
python -m trainingday.multiturn_protocol --input /owner/private/candidate_sessions.jsonl
```

The bounded command reads no more than 2 MB of private JSONL and prints a
small aggregate JSON report without any dialogue text. Nonzero exit status
and a generic error class indicate malformed or incomparable input without
echoing potentially confidential data. Mechanical output includes declared
session counts, completed turns, technical censoring counts and flags such as:

```json
{
  "model_identity_attested": false,
  "actual_multi_turn_conversation_attested": false,
  "process_session_continuity_attested": false,
  "heldout_independence_verified": false,
  "human_review_performed": false,
  "semantic_personality_qualified": false
}
```

Those fields are intentionally **always false**, as is
`cross_arm_model_quality_compared`. `cross_arm_input_format_compared` only
records whether two or more declared arms were available for input matching.
`matched_user_context_validated` and `matched_generation_settings_validated`
describe equality of bytes and
parameters *supplied in the input*, not independent host attestation; they
remain false when fewer than two arms are provided.

The proposed human rubric is maintained separately as
`TRAININGDAY_PATRICK_DIALOGUE_RUBRIC_V1.md` (source-only, Patrick approval
pending). Before any meaningful result, independent reviewers must be allowed
to see the **entire fictional user-and-model conversation** while blinded to
arm/model labels; use per-dimension `NOT_EVALUABLE` with reasons, register
eligibility/censoring rules before unblinding, and calibrate disagreements.
Host-owned model and process identity, actual weights, compatible context
windows, token and wall budgets, rights-admitted data, protected custody and
repeated seeds require separate evidence.

## No implied operational effect

This introduces no host execution, model loading, training, evaluation dataset,
weight modification, semantic score or autonomous runtime behavior. Its tests
verify only the declared contract. The intended next proof is authenticated
Qwen3.5-4B base versus prompt-only and correct/zero/corrupt R16 arms, using
owner-authorized physical runtimes, independent held-out fictional dialogues,
and honest human conversation scores.
