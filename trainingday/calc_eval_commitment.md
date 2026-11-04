# TrainingDay Calc private DEV commitment candidate

**Status: SOURCE ONLY / NOT ENROLLED / HARD HOLD.** This is a new R201
mathematics task lineage, not a replacement or ratification of frozen R18/R185
data. It does not unseal a protected evaluator or authorize GPU training.

The public TRAIN curriculum is separately developed using exact selected Calc
primitives. This DEV candidate deliberately uses an **independently coded,
integer-exact power-rule oracle** for polynomial derivatives, with targeted
cross-checks against `calc.algebra.Polynomial`. This limits common-mode oracle
failure; it does not constitute a formal proof of Calc or mathematical
generalization.

Source reference (not a trusted installed runtime binding):

- `thebrazenbeard/calc@17e12d0f659b42732fb98e5730732aba9ff1b488`
- `src/calc/algebra.py` Git blob `54b7bb78e94481d80b973f46edbaab48b9877b96`
- SHA-256 `6a1b269d14ddb34fbe4e8e5a9f56ac98c401fa98636cd3b13ab423cba1bead08`

`commitment(private_nonce, public_train_prompts=...)` is a local-only pure
function that returns a canonical aggregate SHA-256 for 64 DEV examples,
counts and source declarations. It does **not** return example text, answers,
IDs or the nonce. The separately held 32-byte high-entropy nonce must never
be published, hard-coded, placed in an environment variable log, or reused
between actual evaluator runs. Tests use a public nonce **only as a fixture**.

DEV tasks use degree 4–6 polynomials with nonzero leading coefficient and
evaluation points with absolute value 11–53. The proposed public curriculum
uses cubic polynomials with points −5…5, so this is an explicit distribution
shift; it can test extrapolation but cannot by itself isolate mathematical
generalization from format and difficulty effects. Protected evaluation should
also contain matched-distribution unseen cases held by a separately
authenticated evaluator.

`matched_commitment(private_nonce, public_train_prompts=...)` is a **second,
distinct source-only candidate**: 64 cubics with the same coefficient ranges,
small evaluation-point range and exact prompt form as R201 TRAIN. Its secret
HMAC draws have a dedicated `R201_MATCHED_CUBIC_V1` domain and independent
example IDs, so they are distinct from the shifted DEV candidate even when
the same secret fixture is used locally. A real evaluator should hold its
own fresh secret and should not publish it, examples or answers. The function
returns only an aggregate SHA-256 and explicit FALSE authority flags.

The matched candidate now requires the **exact ordered 1,024 frozen R201
TRAIN prompts**, independently regenerated from the pinned schema and checked
against the frozen complete public TRAIN JSONL SHA-256
`fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b`.
Arbitrary 1,024-item lists, reordered or altered lists, and duplicate prompts
are rejected. This provides a SOURCE-content match, **not** authenticated
authorship, rights, approved runtime source custody or independent evaluator
identity. The source identity and rights/custody/grant flags remain FALSE.

Besides exact-text overlap, it compares the canonical derivative-task tuple
`(3*a, 2*b, c, x)` for both TRAIN and private DEV, which catches a duplicate
task with a different irrelevant constant `d` or whitespace formatting. It
draws replacements deterministically when such a task would overlap the
frozen TRAIN set or an earlier matched DEV row, with a finite fail-closed
attempt bound. This is a bounded **cubic-task equivalence** check, not proof
of arbitrary symbolic semantic disjointness; the independent verification
flag remains false. Matched distribution describes the same task *family and
ranges*, not statistical identity of finite samples or generalization.

`matched_commitment` specifically rejects the two **known public nonce
fixtures** embedded in the source tests. Its positive source tests create
transient process-local random nonces instead. That rejects accidental replay
of those public constants; the function still **cannot authenticate nonce
entropy or secrecy** and must not be treated as a protected evaluator.

An optional public TRAIN prompt list rejects duplicate public prompts and
exact-string TRAIN/DEV overlaps. The result labels this **surface check**, not
semantic disjointness. Equivalent polynomials can be formatted differently,
and a caller can fabricate a public TRAIN list. Neither hash consistency nor
this comparison authenticates lineage, ownership, curriculum exclusions,
evaluator independence, or nonce entropy.

### Evidence required before real use

1. Independently source-pin the exact public TRAIN corpus and separately
   enroll an evaluator-controlled secret nonce and sealed DEV examples outside
   public GitHub and the training process.
2. Authenticate evaluator identity and commitment with a protected signing key
   and physical readback of enrollment/revocation on the intended NAS/host.
3. Independently verify complete TRAIN and both DEV semantic exclusions;
   test protected matched- and shifted-distribution performance across
   independently selected unseen seeds, with identical answer-frequency
   controls per optimizer prefix.
4. Obtain the separate rights grant, physical host process/GPU reservation,
   model hash and protected evaluation release. A passing unit test does not
   create these grants.

All authority flags in this module remain false. There is no network, CUDA,
model-load, private-file-write or host service effect.
