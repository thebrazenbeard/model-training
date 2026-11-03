# TrainingDay R201: Calc-backed polynomial TRAIN source

**Status:** source-only, new corpus lineage, unauthenticated **HOLD**. No
physical Qwen training, GPU lease, rights grant, protected evaluator release,
or model-quality result follows from this generator.

## Actual Calc dependency and provenance

- Source repository: [`thebrazenbeard/calc`](https://github.com/thebrazenbeard/calc)
- Exact source commit: `17e12d0f659b42732fb98e5730732aba9ff1b488`
- Inspected `src/calc/symbolic.py` SHA-256:
  `aba705f0d4370afa35538280f3db00ecbe36da1b3f3ca083ee29ebc292113b0b`
- Inspected `src/calc/__init__.py` SHA-256:
  `4bb35910f0f5897476911d7846498915e8a7150f7993d79a12360570c5199cf4`
- Dependency: `calc.symbolic` (`Var`, `Const`, `Expr.diff`, `Expr.eval`).
  The package initializer and oracle source are checked against these pinned
  hashes **before** normal Python import; the real module type, import spec,
  loader/source path, `sys.modules` identity and oracle file hash are then
  checked again after import. A concurrent source replacement, import-hook
  attack, compromised interpreter or filesystem can still subvert this static
  check; it is not a sandbox, host-trust attestation or package-signing grant.
- No Calc source is vendored into `model-training`. The live Calc repository
  does not yet have a verified root `LICENSE` file in this integration review;
  an independent third-party redistribution license is **not inferred**.

The output is deliberately distinct from the frozen R18/R185 corpus and from
the R199 simple-arithmetic generator. It uses no chat exports, inherited
samples, copyrighted exercise sheets, or protected DEV examples. Arithmetic
truths are cross-checked against a separately implemented integer power rule.

## Reproduce locally (source only)

After obtaining the exact Calc commit in a trusted local checkout, from a
model-training checkout run:

```bash
PYTHONPATH=/path/to/calc/src:. python3 -m trainingday.calc_curriculum
PYTHONPATH=/path/to/calc/src:. python3 -m unittest -v tests.test_trainingday_calc_curriculum
```

The CLI emits a JSON source commitment, not model weights or hidden DEV rows.
The `train_rows()` function returns **1,024 deterministic public TRAIN** rows
with the existing `{id, prompt, answer, split}` shape. Each prompt asks for
the derivative of a bounded integer cubic polynomial at an integer point;
all answers are cross-checked using:

`f'(x0) = 3*a*x0*x0 + 2*b*x0 + c`.

Calc's public `Expr.eval` returns a floating-point number, but this candidate
only permits bounded integer expressions whose results are smaller than
`2**53` and compares them to the exact integer result. This does **not**
make arbitrary Calc numeric approximations exact or justify using finite
differences at nonsmooth points, aliased quadrature, or unbounded operations
as correct-answer labels.

Expected source commitment at this revision:

```text
schema=VERA_TRAININGDAY_R201_CALC_POLYNOMIAL_V1
train_count=1024
train_sha256=fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b
oracle_family=CALC_SYMBOLIC_BOUNDED_INTEGER_POLYNOMIAL_DERIVATIVE_AT_POINT
```

The model-tokenizer contract, actual 64-token sequence fit, worker-owned
process identity, and permission to load any pinned Qwen checkpoint are
separate. The prompts are short and contain no protected labels; they have not
been used in an optimizer step. Running the generator under an environment
without Calc **fails closed** rather than substituting an unverified oracle.

## Evaluation and promotion boundary

There is **no private DEV** in this module. A separate evaluator custodian
may generate secret-nonce-derived, distribution-shifted problems and publish
only a blinded commitment. A public unit-test nonce is never a real secret.
The model should be evaluated against genuinely unseen tasks with controls,
not judged by whether it memorizes deterministic published training rows.

To promote a source candidate to admitted training data, independently
authenticate the owner/steward's permission, frozen exact-source provenance,
private evaluator custody, model identity, and a legitimate host-owned
training lease. This document and its passing source tests issue none of them.
