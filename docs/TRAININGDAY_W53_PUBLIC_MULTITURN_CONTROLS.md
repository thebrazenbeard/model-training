# TRAININGDAY W53 — public multi-turn causal-control gate

Status: **SOURCE-ONLY PUBLIC SYNTHETIC PROTOCOL. NOT A VERA RUN. NO HELD-OUT
EVALUATION. NO NEURAL LEARNING MEASUREMENT.**

`trainingday/public_multiturn_controls.py` supplies two original fictional
three-user-turn scripts. The scripts differ only in whether turn two contains
an explicit correction. They are deliberately **public** and must never be
called a protected/independent evaluation bank. They include no model answers
and no Patrick-private chat, history or persona text. They are not approved
training data.

## Actual control defect closed

The existing `multiturn_protocol.validate_sessions()` correctly matches
declared user turns, per-case generation settings, and within-arm model
declarations. It **does not** demand all candidate arms come from the same
base checkpoint or that persona and control adapters are arranged as a
causally interpretable experiment. The W53 auditor adds those checks while
leaving provenance explicitly unverified.

The required *declared* factorial pilot contains 30 sessions:

| Role | Persona input intent | Adapter input intent | Historical meaning |
|---|---|---|---|
| `base` | Absent | None | Official base baseline, not identity-qualified |
| `persona_base` | Same externally versioned profile as remaining arms | None | Isolates persona prompt effect |
| `r16_zero_step` | Same persona | Zero-update adapter | Adapter-loading control |
| `r16_correct8` | Same persona | October correct-label eight-step adapter | Arithmetic/cognitive task pilot, **not dialogue-tuned** |
| `r16_corrupt8` | Same persona | October corrupted-label eight-step adapter | Corrupted-label technical control, **not dialogue-tuned** |

Each role is run against both public scripts at budgets **64, 96, 128**
reply tokens (5 × 2 × 3 = 30 planned sessions). Each session has three
user-turn attempts; some actual model replies may legitimately be absent
due to technical timeout/truncation and must be censored, not scored.

Controls enforced on owner-supplied rows:

- Every arm covers all six `script × budget` slots, with identical user
  text and turn ordering across arms within a slot.
- Every arm claims the **same base weight hash and origin string**. There is
  no authenticated official revision field in the imported session schema;
  the actual model revision remains unverified.
- `base` and `persona_base` claim `NO_ADAPTER`. The three R16 pilot arms
  declare distinct adapter SHA-256 strings (not necessarily authentic bytes).
- All persona-conditioned arms claim the **same prompt/profile digest**;
  the no-profile baseline must claim a different prompt digest.
- Every slot claims context window 4096, temperature 0.700, seed 17, and
  exactly its named maximum reply budget. These are preliminary fixed
  public settings, **not a qualified inference recipe**.
- All records come from a single class, either `mock_test` or
  `public_exposed`. Mixed or `private_unverified` rows are refused to avoid
  conflating public fixtures with private evaluation.

The `prompt_sha256` field in the existing transcript schema is an **unverified
caller declaration**. Matching it does not establish that the profile was
installed or that its rendered bytes reached the model. The same warning
applies to weight/adaptor hashes and origin claims. External evidence must
bind verified bytes, runtime executable, session process/PID, prompt payload,
hardware lease, tokenization/stopping policy, and actual output capture.

## Reproducible local use

Print the full original public USER-only plan (no model outputs):

```shell
python -B -m trainingday.public_multiturn_controls
```

For a future owner-prepared `VERA_DECLARED_MULTITURN_DIALOGUE_V1` sequence:

```python
from trainingday.public_multiturn_controls import audit_public_study
result = audit_public_study(owner_prepared_session_rows)
```

`audit_public_study()` validates the *declared* 30-slot matrix and returns
only aggregate counts and hard-false attestation flags. It does not perform
inference, produce assistant messages, emit raw dialogue, assess correction
quality, score personality, or certify physical experiment performance.
It is intentionally separate from the W51 blinded 8D review receipt
validator, where owner approval of the proposed 0–3/NOT_EVALUABLE rubric,
thresholds, independent raters and protected evaluation custody remain
pending. Do not convert the older 0–2 public single-turn rubric to 0–3.

Run focused source tests:

```shell
python -B -m unittest tests.test_trainingday_public_multiturn_controls -v
```

## What a defensible physical study would additionally require

1. Rehash and authenticate official Qwen base and each R16 artifact; preserve
   same base bytes across all five arms; obtain provenance and runtime PID
   binding. The October correct/corrupt LoRAs were **task pilots**, not a
   claim of personality training or expected voice improvement.
2. Ensure persona bytes are actually loaded only in intended arms. Capture
   model outputs at 64, 96 and 128 separately, recording truncation,
   stop-token behavior, peak memory, latency, session continuity and reload.
   Repeat across independent seeds; this public one-seed matrix cannot
   establish reproducibility.
3. Keep public scripts out of protected holdout. Rights-admit any subsequent
   training curriculum with exclusions. Independent evaluator prepares new
   unseen dialogue and holds it separately; no protected prompts or keys
   belong in this code or GitHub.
4. After Patrick reviews/approves the eight-dimension rubric and actual
   reviewer thresholds, collect fully blinded human ratings with excerpt
   evidence, counterexamples and inter-rater agreement. Compare base versus
   persona before attributing anything to adapter updates. Preserve
   NOT_EVALUABLE for technically censored or unobservable dimensions.

> **Hostile challenge:** A fully populated 30-session matrix can still be
> scripted fake assistant replies, or five distinct backends masquerading
> behind one claimed base hash. Structural pairing is necessary, not
> sufficient, for any measured identity, retained personality or learning
> conclusion. A technical arithmetic LoRA might do nothing useful for
> conversation—or even make it worse. This protocol cannot decide.
