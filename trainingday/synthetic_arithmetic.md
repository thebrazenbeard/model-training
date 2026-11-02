# R199 first-principles arithmetic corpus — separate source candidate

**Source-only, unauthenticated HOLD.** This is a newly generated experimental
dataset lineage, **not** R18/R185. Do not substitute its hashes for the frozen
R185 `R18_ARITHMETIC` owner/source pin or treat it as a grant for its inherited
data, protected evaluator, GPU, or Qwen weights. This does not change or enable
`one_step_kernel`; there is still no physical model-training run.

`trainingday.synthetic_arithmetic.train_bytes()` produces exactly 1,024
canonical JSONL TRAIN examples (addition, subtraction, multiplication and exact
division) using only independently implemented integer arithmetic and a public
deterministic generation domain. Every prompt and answer can be recomputed from
the Python source. No third-party dataset, chat export, or private answer bank
is ingested.

`commitment(nonce)` requires a 32-byte nonce supplied by a future independent
evaluator and produces a **hash/count-only** commitment to 64 deterministic
DEV examples in disjoint operand ranges. It does not return DEV rows or gold
answers and does not persist the nonce. The unit-test nonce is intentionally
public and **not** protected evaluator custody. Real separation, OS-generated
nonce entropy, exclusion-bank comparisons, evaluator authorization, and private
DEV lifecycle remain unimplemented. A hash is not proof of who generated or
holds the underlying examples.

Stable public TRAIN digest (SHA-256 canonical UTF-8 JSONL):

`1ba1b03adbfeccebaadd47674d8624ea80e6b4dabf8678d1d7084c88055b1f63`

Source-only verification:

`python -m unittest tests.test_trainingday_synthetic_arithmetic`

## Rights limitation

The U.S. Copyright Office distinguishes unprotected facts, ideas, and
mathematical methods from potentially protected *expression* and original
selection or arrangement. This implementation originates simple arithmetic
from first principles and avoids copying a third-party expression or
compilation; that reduces provenance risk but **does not authenticate a
rights-holder or itself issue a training-data license**. The repository's
current proprietary license and separate authorization boundaries still
apply. Sources: [Copyright Office FAQ](https://www.copyright.gov/help/faq/faq-protect.html),
[Circular 33](https://www.copyright.gov/circs/circ33.pdf).

## Hostile caveats

- Public TRAIN and the public generator permit memorization or pattern
  exploitation; the private split must be protected and evaluated genuinely
  out of sample, not merely inferred from a commitment hash.
- Train/DEV operand separation prevents exact prompt overlap here, not broad
  contamination of related arithmetic skills or independent model evaluation.
- The full future training job still needs validated official model bytes,
  a GPU fit result, host-owned authority, rights, and protected DEV custody.
