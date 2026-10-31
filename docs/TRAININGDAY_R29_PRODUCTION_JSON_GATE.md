# R29 production admission strict JSON boundary

**Scope:** `trainingday.admission.admit()` as actually invoked by corpus
consumers, not the R28 public-toy wrapper alone. This is a source-content
validation improvement; it authenticates no legal rights or host authority.

## Reproduced pre-change bypass

The production entrypoint read the manifest and JSONL rows with default
`json.loads`. A manifest containing two valid but conflicting `source_commit`
keys was accepted with the **last** one selected, although the R28 hostile
wrapper rejected the same bytes. Nested rows likewise accepted duplicate
`provenance.rights_review` keys whose last value was `PASS`. Both were direct
`admit()` tests against a deterministic, public-only toy corpus; neither used
private bank data, authentic credentials, or GPU compute.

## Implemented gate

1. Manifest and each JSONL row are decoded by the *production* strict JSON
   parser. Exact duplicate object keys and keys colliding after Unicode NFKC
   normalization plus casefold are rejected at **every** object level.
2. UTF-8-invalid bytes, unpaired JSON surrogate escapes, `NaN`, `Infinity`,
   overflowing nonfinite floats, and excessive nesting are rejected with
   `AdmissionError`, not allowed to surface late as encoding failures.
3. Manifest reads are capped at **64 MiB** and JSONL line reads at **1 MiB**
   including the newline; JSON value depth is capped at **64** levels.
   Sources beyond these bounds require deliberate policy review, not silent
   acceptance.
4. Shard names whose NFKC-casefold forms collide are rejected to prevent
   distinct-looking paths aliasing on differently configured filesystems.
   Existing traversal/symlink and case-ID/holdout checks remain in force.
5. A row must explicitly declare `evaluation_holdout` as Boolean `false`;
   missing declarations, numeric `0`, and protected `true` are denied. This
   preserves the existing `training_use` split contract and public toy
   fixtures while removing permissive falsy-value coercion.
6. The receipt's `manifest_sha256` is computed from the exact `manifest_raw`
   bytes admitted by the parser. A concurrent producer mutation cannot
   silently change the receipt to describe a different later version. This
   is receipt fidelity, not an atomic immutable-filesystem guarantee.
7. The ordered selection now also carries `selected_id_v2_schema` and
   `selected_id_v2_sha256`: SHA-256 of ASCII
   `VERA_TRAININGDAY_SELECTED_ID_DIGEST_V2`, a zero-byte separator, and the
   UTF-8 encoding of a compact JSON array of the selected IDs in order
   (`ensure_ascii=False`, separators `(',', ':')`). Unlike joining ID strings
   with newlines, this is unambiguous even when IDs contain embedded newlines.
   The existing `selected_id_sha256` is retained for R18 consumer compatibility
   but explicitly labelled by `selected_id_sha256_status` as
   `LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE`.

## Evidence / repeatability

From repository root, without external packages:

```bash
python -m unittest discover -s tests -p 'test_r29_production_admission_strict.py' -v
python -m unittest discover -s tests -p 'test_r28_input_adversary.py' -v
```

The R29 direct-entrypoint test first ran RED against the unmodified production
parser (six substantive bypass failures, three missing-bound/error failures,
one valid control pass). After the first production change, 16/16 R29 passed.
A follow-up independent hostile review exposed two more production gaps:
coerced/missing `evaluation_holdout` and manifest-receipt TOCTOU. Three new
direct-entrypoint regressions ran RED (3 failures) before their repair, and
the `true` holdout control already passed. After the second production change,
20/20 R29 and 8/8 R28 passed. A further independent hostile review then
demonstrated that two distinct, fully admitted ID lists (`['a\nb', 'c']`
versus `['a', 'b\nc']`) collided under the legacy selection digest despite
different manifests. Its direct-entrypoint regression ran RED (missing V2
receipt field), then GREEN: **21/21 R29** plus **8/8 R28** tests passed.
The existing pytest-based
`tests/test_trainingday_admission.py` could not run in this sandbox because
`pytest` was not installed; it remains a required upstream CI check.

## Explicit non-claims

- `ADMITTED_SOURCE_ONLY_NOT_TRAINED` is not a genuine rights/consent grant.
- Self-attested `rights_review: PASS`, `privacy_review: PASS`, and `license`
  fields are not independently authenticated here. They must be bound to an
  externally governed rights/lease gate before actual training.
- No neural weights were changed; no host GPU, real corpus, deployment,
  release, and no external credentials were accessed.
- This work does not claim complete fuzz coverage, host filesystem race
  elimination, archive supply-chain authenticity, or preemptive prevention of
  every possible non-JSON malicious input. R28's toy-only wrapper remains
  separate; independent exact-head review is required before integration.
- Existing R18 MeZO/QLoRA/benchmark/evaluation consumers still recompute the
  legacy newline-delimited digest; they have **not** been migrated to V2.
  Accordingly any release or admission whose identity depends solely on that
  legacy digest remains **source-only HOLD** until a separately owned, tested
  consumer migration and cross-receipt verification are complete.
