# R201 owner rights-claim candidate — source integrity only, HARD HOLD

This module is an explicit **new R201 subject**, not a reuse of the historical
R18/R185 rights signature. It permits **one person to serve as both the rights
holder and data steward**; it does not demand an unrelated second human.
It accepts a canonical, versioned, detached Ed25519 signature for exact scope:

| Commitment | Required value / meaning |
| --- | --- |
| Corpus ID | `R201_NEW_CALC_POLYNOMIAL_TRAIN` |
| Source/row type | `VERA_TRAININGDAY_R201_CALC_POLYNOMIAL_V1`; exactly 1024 TRAIN rows |
| TRAIN SHA-256 | `fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b` |
| Calc revision | `17e12d0f659b42732fb98e5730732aba9ff1b488` |
| Calc symbolic source SHA-256 | `aba705f0d4370afa35538280f3db00ecbe36da1b3f3ca083ee29ebc292113b0b` |
| Model | Official `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Purpose | `QWEN35_VERA_TRAININGDAY` |
| Protected evaluation | Both DEV-commitment and exclusion-receipt SHA-256 values required; these **must be produced and authenticated by an independent custodian** for a real request |
| Signer and validity | Owner identity, owner key fingerprint, epoch, UTC interval, explicit `ASSERT_RIGHTS_TO_TRAIN` or `WITHHOLD` |

The implementation has **no signing API, private key, key-enrollment endpoint,
host integration, service modification or positive grant**. It rejects wrong
R18 corpus, model, purpose, Calc source, altered TRAIN, missing protected
digests, malformed canonical JSON, stale caller-supplied epochs and invalid
signature. A correctly signed `WITHHOLD` is recognized and remains HOLD.

**A valid synthetic/self-issued signature proves nothing about permission.**
An attacker can generate a matching key and expected DEV/exclusion hashes,
then sign a structurally perfect claim. The source result intentionally keeps
owner enrollment, independent source-rights review, evaluator custody,
time/replay, corpus admission, GPU lease, physical training and evaluation
authorization **all false**. Its `rights_basis` string describes the owner's
as-yet-unreviewed assertion, not a license adjudication. In particular, the
Calc repository's missing root LICENSE must not be papered over by this code.

To progress beyond this Draft, an authenticated host operator must independently
pin the actual owner's key and issuer, verify origin and use-rights for the
precise Calc generator and output, obtain independently protected DEV and
exclusion commitments, ratify exact signed bytes under nonrollback custody,
and bind them to host-owned immutable plan, process identity and exclusive
GPU lease plus protected Q6 NAS witness. Those are separate technical effects
and currently unperformed. No repository-side signature verifier can issue
them by itself.

The API is `trainingday.r201_owner_claim_candidate.inspect_r201_owner_claim_candidate`.
It takes the *exact* claim bytes, detached signature, **supplied** owner key,
expected identity, two independently sourced expected evaluator hashes,
**supplied** minimum key epoch and **supplied** current UTC time. The Python
function never loads Calc or protected DEV, never writes data and never calls
the optimizer or broker. It throws `ClaimEvidenceError` on malformed evidence.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q tests.test_r201_owner_claim_candidate
```

Test keys and protected hashes are fabricated and confer zero authority.
The test suite's success is a portable source property, **not** host
installation, authentic owner consent, protected evaluator custody or training.
