# TRAININGDAY — owner-local Vera browser transcript import V1

Status: **source-only interoperability / no conversation qualification**.

The user-facing `v-e-r-a` browser currently keeps messages as an ordered
`[{role:"user",content:"..."},{role:"assistant",content:"..."}]` array. This is
source-inspected in `thebrazenbeard/v-e-r-a` Draft PR #24 at exact commit
`3e7a766da005bddee642017970a1b0544f3bdf3e`,
`interface/static/app.js`: the reply is added only after `/api/chat` returns
success; technical failures create a separate `system` bubble and restore the
user's prompt instead of appending model text. The request uses
`messages.slice(-12)`. Saved-vault recall can instead send a single new user
turn plus separately restored encrypted history. **These are source facts,
not live backend/provenance attestation.**

This importer reads **owner-prepared local JSONL** wrapping that browser
messages array plus explicit declared model metadata. The current interface
does *not* provide an authenticated automatic JSONL export endpoint; we do not
claim one exists, scrape browser storage, read the encrypted vault, or call a
model. Tests use invented fictional strings as schema fixtures; they are
**not live Vera conversations** or a protected evaluation dataset.

## Input schema

Each JSONL line must contain exactly these fields:

| Field | Contract |
| --- | --- |
| `schema` | Literal `VERA_DECLARED_LOCAL_CHAT_EXPORT_V1` |
| `arm`, `case_id`, `session_id` | Distinct, bounded labels checked by the downstream multi-turn protocol |
| `source_claim` | `mock_test`, `public_exposed`, `private_unverified` (not rights/evaluation custody verification) |
| `model_selection_claim` | UI model-choice label, **unverified**, not a process/weight witness |
| `base_sha256`, `adapter_sha256`, `prompt_sha256` | Operator-declared hashes; adapter may use `NO_ADAPTER` |
| `generation` | Explicit `context_window_tokens`, `max_reply_tokens`, `temperature_milli`, `rng_seed`; never filled with guessed defaults |
| `used_saved_recall`, `restored_from_vault` | Explicit `false` for this visible-only comparison; `true` is rejected because hidden saved context defeats matching |
| `messages` | Even array of 2–12 alternating browser `user` and `assistant` `{role,content}` objects; each pair is a *declared displayed* completed turn |
| `terminal_failure` | `null` or explicit `{user:"...",outcome:"truncated"|"timeout"|"upstream_error"}`; no partial assistant content |

Two attempted turns are required. A failed last attempt may follow one
completed pair. A dangling browser `user` message is rejected: it cannot
be silently converted into a fabricated model response. The optional
`terminal_failure` cannot be reconstructed from the browser's saved `messages`
array alone, because the interface deliberately removes failed prompts from
chat history. It must be recorded as a separate *operator claim*, and it is
always counted as technical censoring, never a model-authored assistant turn.

Use only independent fictional test conversations or appropriately authorized
private records under owner-local custody. No childhood history, passwords,
secrets, therapy notes, unconsented third-party dialogue or other sensitive
text should be placed in public fixtures, CI, repository examples or comments.
There is no private prompt bank or independently validated rights custody here.
Use **opaque, non-sensitive `arm` identifiers** such as `base` or `trained`:
the aggregate report necessarily includes arm labels for comparison, so those
labels must not contain names, account IDs or private references.

## Local, aggregate-only command

```bash
python -m trainingday.local_chat_import --input /owner/private/local_chat_exports.jsonl
```

The command accepts at most 2 MB of JSONL. It does not write normalized
raw dialogue files or emit user/model text. It converts records *in memory* to
the existing `trainingday.multiturn_protocol` contract, which checks case
coverage and exact user-turn/generation matching across different arms. The
stdout JSON reports only bounded counts, technical censoring, the supplied
opaque arm labels and permanent negative claims. It does not reproduce user
prompts or model replies. New flags include:

```json
{
  "local_export_import_format_validated": true,
  "browser_snapshot_authenticated": false,
  "assistant_finish_reasons_attested": false,
  "saved_context_absence_attested": false,
  "model_identity_attested": false,
  "actual_multi_turn_conversation_attested": false,
  "semantic_personality_qualified": false
}
```

`local_export_import_format_validated=true` only means the supplied data fit
the import contract; it is **not evidence** that a live browser, official
Qwen3.5 base, R16 adapter, profile seed or unchanged physical process produced
it. The private file itself and all source claims remain owner-controlled,
unverified input. Converting visible fictional correction/memory turns
preserves their *order and exact text*, not whether the assistant understood
or remembered the correction.

## Remaining human/host qualification

To establish the actual desired Vera behavior, authenticate the local model,
adapter, prompt/version, server PID and context custody independently; use
matched token and wall budgets across the official base, prompt-only,
zero-step, correct-label and corrupt-label arms; capture real continuous
multi-turn interactions with source-owned process receipts; protect held-out
fictional dialogues; and conduct blinded human review using Patrick's
owner-approved rubric. A 64-token prototype that times out or truncates must
be counted separately as a technical failure, not an ordinary bad personality
score and certainly not a semantic PASS.

No new web endpoint, browser exfiltration path, vault access, physical model
run, training, saved-data transformation, or runtime change is introduced.
