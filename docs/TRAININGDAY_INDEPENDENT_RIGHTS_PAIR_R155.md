# R155 — detached dual-signature training-data rights evidence (SOURCE ONLY)

`trainingday/rights_pair_evidence_r155.py` examines the integrity of a canonical
JSON evidence record signed separately by a putative rights holder and data
steward. **It cannot admit data.** A caller can invent two Ed25519 keys, issue
an `ADMIT` document and pass every local cryptographic check. Accordingly, the
verifier always returns `corpus_rights_admitted: false`,
`physical_training_authorized: false`, and
`protected_evaluation_authorized: false` even when both signatures verify.

The test suite signs only fabricated synthetic commitments, never actual
TrainingDay data. These are cryptographic integrity tests, **not independent
human review, licensing evidence, signed host receipts or an ADMIT event**.

## Exact receipt commitment

The receipt is no larger than 16 KiB, duplicate-free strict UTF-8 JSON with
sorted keys, no padding, and exactly the `VERA_TRAININGDAY_RIGHTS_PAIR_V1`
fields. Both Ed25519 signatures cover the domain-separated **exact bytes**.
The receipt binds corpus family/ID, manifest, train, dev and protected-exclusion
digests, a license-evidence digest, separate signer IDs and key fingerprints,
the official Qwen3.5-4B base revision, training purpose, UTC validity window,
and a revocation-snapshot digest. A `REJECT` receipt remains `SIGNED_REJECT`.
Changing one byte invalidates the detached signatures. The implementation
rejects reused signer keys, changed corpus/split hashes, expired documents,
revoked *supplied* fingerprints, wrong license-family strings, unknown corpus
types, and malformed/canonicality-ambiguous JSON.

The family-specific labels are routing assertions, **not licensing conclusions**:

| Family | Expected rights evidence label | Unresolved independent review |
| --- | --- | --- |
| `DOLLY_V4` | `CC-BY-SA-3.0` | Exact 80-train/16-dev bytes, share-alike and privacy/copyright obligations, row rights |
| `REZON_NEGATIVE_PEER` | `REZON-SECTION-5-PRIOR-WRITTEN` | All 28 quarantined candidates, contributor title and actual prior written §5 grants |
| `LABYRINTH_SYNTHETIC` | `PROVENANCE-REVIEW-REQUIRED` | Independently replayed gold, training-use license and heldout exclusions |

All three are `NOT_ADMITTED` under the current Bus #404/#406 source state.
No identified independent rights-holder, steward, or signed dataset receipt
is included in this repo.

## Installation boundary, still absent

1. The host operator must authenticate a **separately governed** rights-holder
   public-key registry and a noncolluding steward registry with accountable
   identities, issuer rotation/revocation and audit control. Caller-supplied
   public keys do not establish either identity.
2. A separate authority must prove the time source and that the revocation
   snapshot is the newest accepted monotonic checkpoint. A passed-in clock
   and `frozenset` cannot provide rollback resistance.
3. The steward must bind verified actual row bytes, source rights and consent,
   protected-bank exclusions, data revocations, and corpus deduplication to the
   exact receipt. Hash strings alone are assertions.
4. Only a host-owned immutable admission broker, independent evaluator and
   valid GPU/NAS lease can authorize the intended physical use. The source
   verifier has **no transition** to an `ADMITTED` status, model invocation,
   evaluator reveal, launcher, or promotion.

`cryptography` is imported only when checking a signature. If unavailable the
operation fails closed with `AdmissionError`; there is no fallback algorithm,
automatic dependency download, network access, or signing key in this module.
No existing TrainDay admission/physical launch code is modified.

Focused local test:

```shell
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_rights_pair_evidence_r155 -q
```

The ten tests include a deliberate fail-closed missing-dependency test. The
other nine use synthetic signatures, including a valid **self-issued** pair
that deliberately remains unadmitted. If `cryptography` is missing those nine
are explicitly SKIPPED; a green run with skips is not signature verification
evidence. This is eligible only for a source-only Draft and later independent
owner integration review.
