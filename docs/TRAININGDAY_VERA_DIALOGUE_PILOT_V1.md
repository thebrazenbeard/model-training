# Vera dialogue pilot v1 — fictional candidate, rights HOLD

**Status:** SOURCE CANDIDATE ONLY. Not an admitted training corpus, a physical
experiment, an identity/personality transfer, or independently protected evaluation.
This work extends Draft PR #335 without modifying its arithmetic, R202, evaluation,
or model-loading paths.

## Why this pilot exists

Patrick's immediate target is **recognizable, practical conversation**, not an
arithmetic score: honest self-identification; useful advice without robotic
checklists; carrying forward explicit corrections; distinguishing facts given in
the session from invented long-term memory; evidence-calibrated uncertainty;
dry humor when it helps; respectful challenge; and fluid conversational repair.
The source generator in `trainingday/dialogue_curriculum.py` contains **80 short,
fictional dialogue cases** (19 have an explicit invented prior exchange): 64 TRAIN
and 16 DEV, balanced over eight
behavior families (eight TRAIN and two DEV per family). There are no verbatim
private user conversations, proprietary Vera profile passages, synthetic W40/W42
smoke transcripts, passwords, or scraped copyrighted corpora in this proposal.
No PII or real operational host state is required to generate the cases.

All rows say `rights_status=NOT_ADMITTED`, `training_authorized=false`, and
`heldout=false`. The deterministic manifest hashes bind the exact public candidate
bytes; they are not a signature, an owner's license grant, or quality validation.
The examples are new fictional drafts that may be AI-assisted. That provenance
does **not** magically prove exclusive copyright or a right to all downstream uses.
Human review and a defensible lawful-use basis are required before training.

## Admission and signature gate

The separate `docs/VERA_DIALOGUE_ADMISSION_ATTESTATION_SCHEMA_V1.json` defines the
minimum host-verifiable record. It is deliberately not instantiated with invented
owner names or signatures. A qualified owner-attended signer must bind:

1. The manifest's exact `candidate_sha256`, plus TRAIN and DEV hashes and generator
   Git commit; official `Qwen/Qwen3.5-4B` model ID and immutable base revision
   commit; a changed byte or base revision invalidates this grant. The
   schema's 40-hex revision format check is **not official-source
   authentication**. The host verifier must compare that value to its separately
   authenticated **official revision allowlist**; no such allowlist is admitted
   or bound by this document.
2. Rights-holder and steward identities, their actual authority over the exact
   admitted text, human edits/review and legal/permission basis, with explicit
   limitations on claims for unprotectable AI-generated material.
3. Explicit bounded purpose: noncommercial local Qwen3.5 Vera research/training,
   named processors and hosts, retention, deletion handling, and exclusions.
4. A real signed owner/steward assertion and an independently checked cryptographic
   verification result, not merely schema-valid JSON or a self-set `verified` flag.

**Signable message v1 (source-only specification):** form the assertion document
*without* the top-level `signatures` object and without the field
`signing_payload_sha256`. Canonicalize that reduced object using RFC 8785 JSON
Canonicalization Scheme (UTF-8). Prefix the canonical bytes with exactly
`VERA_DIALOGUE_ADMISSION_V1\n`; sign those full bytes with Ed25519 using each
actor's independently authorized host-owned key. The `signing_payload_sha256`
field equals SHA-256 of that same prefixed byte sequence. Both actor signatures
are detached base64 signatures over the identical bytes. A verifier must parse,
canonicalize, recalculate SHA, validate both signatures and actor authority, and
reject any mismatch/unknown key. *No such signer or verifier has been run here.*

The schema validates **format only**. It grants nothing and cannot attest a
signature's truth. A trusted host admission verifier must check cryptographic
signatures, signer authorization, immutable content hash and version, current
exclusion policy, and corpus review. Until then: **HOLD**. One human may be both
rights-holder and steward only where they genuinely have both authorities;
independent third-party human roles are not intrinsically required.

### Exclusions

- Patrick's private chat exports, autobiographical material, family records,
  credentials, device logs and personal history unless separately scoped and
  deliberately authorized in a future *private* exercise.
- Proprietary `vera` repository behavior-profile or other unlicensed passages.
- Public W40/W42 conversation/smoke prompts, PR #335 public probing cases,
  model outputs, or anything from the protected evaluation pipeline.
- Untraceable web scrapes, copyrighted books, chat logs, another person's data,
  and examples for which source or lawful-use evidence is absent.

## Dataset and evaluation boundaries

| Set | This source | Intended role | Authority |
|---|---:|---|---|
| TRAIN | 64 synthetic fictional pairs | tiny SFT/infrastructure candidate | HOLD for training |
| DEV | 16 distinct fictional pairs | public tuning smoke tests only | Not blind |
| PROTECTED HELDOUT | **0 generated** | target 32 independently authored, unseen cases | Separate evaluator custody |

The eight behavior families deliberately share desired skills. TRAIN and DEV
scenario IDs and exact prompts are disjoint, but this is **not proof of semantic
independence**; humans should audit near-paraphrase and task leakage. No model
author/optimizer may author or inspect protected heldout labels/prompts. The
independent custodian should precommit hash/volume and protect the 32 examples
*before* training, then perform contamination review against all train/dev/smoke
data. Exposed conversations are never retroactively promoted to heldout.

## Feasibility and acceptance gates

- **Tokenizer HOLD:** No exact Qwen3.5-4B chat-template tokenizer run was performed
  here. Measure the entire templated TRAIN sequence, official model tokenizer and
  chosen LoRA pack/pad behavior; independently measure the runtime system prefix
  (owner persona seed is ~1,751 characters) and full evaluation contexts. Do not
  assume a `<=256`-token budget across those surfaces or silently truncate.
- **No train equivalence:** One- and two-exchange fictional scenarios cannot prove
  multi-turn correction carryover, memory discipline, or response personality in a
  sustained conversation. Include separately designed fresh live probes.
- **Rubric unresolved:** PR #335's public conversation rubric uses a 0–2 scale;
  Patrick's previously reviewed proposal uses 0–3 plus NOT_EVALUABLE. Do not
  collapse these silently. Owner must designate a canonical rubric version.
- **Host qualification:** Authenticate actual model weights, rights grant, lease,
  resource limits, checkpoints, deterministic runs, independent evaluator and
  genuine inference reload before calling any model improvement real.
- **Experimental design:** Compare same-checkpoint zero-update, correct-label and
  corrupted-label controls against at least one candidate training method;
  repeat across seeds and evaluate untouched conversations, not just loss.
- **Pilot size:** 64 short pairs may establish whether the code and rights
  admission gates work. They are not remotely sufficient to promise stable
  personality transfer, improved generalization, or a natural voice.

**Source verification:** run
`python -m unittest tests.test_trainingday_dialogue_curriculum` from repository
root; generator prints only a source manifest with
`python -m trainingday.dialogue_curriculum`. Neither command trains a model.
