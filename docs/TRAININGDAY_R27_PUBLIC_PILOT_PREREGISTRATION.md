# TrainingDay R27-06: public, answer-free multi-skill pilot preregistration

Status: **SOURCE/CPU research artifact**, never a real Qwen3.5-4B evaluation,
neural-improvement receipt, private examination, or model-promotion grant.

## Frozen intended comparison

Three narrow supported task families: `ARITHMETIC`, `PROGRAM_REASONING`, and
`SOURCE_PRIORITY`. The caller supplies 3–512 hypothetical future task IDs,
their SHA-256 prompt commitments, and SHA-256 *references* to their rights
evidence. The plan records **no prompts, answers, expected outputs, private
labels, or paths**. Task IDs, unique prompt hashes, explicit nonintersection
with caller-supplied known training prompt hashes, four opaque arm IDs, model/
tokenizer/train-corpus hash commitments, one-sided paired-sign-test alpha,
minimum absolute accuracy gain, and a 50% per-family floor are committed in
canonical JSON/SHA-256 **before outcomes**.

Four logical arms, mapped only during the later synthetic scoring call:

- Frozen unmodified baseline.
- Zero-update sham control.
- Correct-label updated candidate.
- Corrupted-label negative control.

For any future real study these must have identical admitted training data,
identical ordered example schedule/epochs/token budgets where applicable,
matched elapsed compute accounting, frozen initial weights/seed, independent
arm assignment custody, independently measured adapter-byte hashes and actual
optimizer receipts. No such evidence exists in this module. **The opaque arm
mapping is a caller declaration, not demonstrated human/model blinding.**

## Statistical decision rule

On public-only *synthetic Boolean correctness outcomes* for every paired case
in every arm, count discordant wins/losses for the correct-label arm against
each frozen, zero-update, and corrupted-label control. Compute exact one-sided
binomial sign-test p-values under win probability 0.5.

The scorer checks exact dictionary type and role-map, outer-arm, and per-arm
cardinality with constant-time size gates **before** visiting keys or items.
Oversized 100,000-extra-entry maps in any of these three layers are rejected
without iterating their content.

Require **both**:

1. Minimum preregistered absolute gain on the same cases.
2. p-value at most one-third of the preregistered alpha per comparison
   (Bonferroni adjustment for three controls).

Also require correct-label accuracy of at least 50% in **each** family and no
family's accuracy may fall more than two percentage points relative to its
frozen baseline. This guards against hiding critical skill loss behind a large
aggregate gain. If any condition fails, outcome is `NO_IMPROVEMENT`. If all
pass, outcome is only
`SYNTHETIC_THRESHOLD_MET_NO_CLAIM`: claimed Boolean bits are not independently
scored model behavior and cannot support a real neural-improvement claim.
No repeated tuning against the reserved future set. A truly independent
test-set custodian must control raw tasks, answer oracle, access, and final
one-shot unseal outside the trainer's OS principal.

## Evidence boundaries and hostile nonclaims

- `DOC`: this preregistration proposal and planned fairness constraints.
- `SOURCE`: canonical public-hash protocol and deterministic synthetic scorer.
- `TEST`: CPU cases proving negative/positive *toy* decision behavior.
- `REAL_SYSTEM_PROBE`: none performed by this lane.
- `MODEL_REVIEW`: internal hostile source review required at the exact commit.
- `FORMAL_VERIFY` / `HUMAN_REVIEW` / `EXTERNAL_INDEPENDENT_REVIEW`: absent.

The module cannot authenticate that hashes refer to unseen tasks, that source
rights or privacy consent exist, that the known training inventory is complete,
that arm allocation was blinded, that all runs saw comparable compute, that
model outputs correspond to reality, that an independent answer oracle graded
them, or that model adapters changed. Repinning all declarations and hashes
does not create trusted custody. These remain **UNKNOWN**, not silently PASS.

R18's original MeZO overrun (531 seconds allowed versus 19,813.562 seconds
elapsed, including Modern Standby) remains **TIME_BUDGET_OVERRUN_INVALID**.
No real MeZO–QLoRA equal-time comparison or Qwen behavior improvement is
established by this fixture.

Do not run on protected OOD inputs, admit an unchecked rights source, merge,
train, spend, or deploy as a consequence of this source result.
