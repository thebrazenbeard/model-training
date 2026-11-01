# R164 — signed rights registry evidence (SOURCE ONLY / no ADMIT)

This draft adds `trainingday/rights_registry_evidence_r164.py`, an offline,
fail-closed integrity check for a canonical signed registry snapshot and the
existing R155 detached rights-holder/steward pair. It is stacked on the R155
source-only draft. **It authorizes no training, corpus admission, protected
evaluation, deployment, or GPU lease.**

## Checked invariants

- An Ed25519 issuer signature covers the exact domain-separated canonical
  registry bytes. The issuer key fingerprint must match the snapshot.
- A bounded roster assigns distinct rights-holder and data-steward identities
  to non-reused public-key fingerprints. Neither may reuse the issuer key.
- The exact registry SHA-256 must match the rights pair's revocation-snapshot
  commitment. A rights signer must exist in the registry in its claimed role.
- Explicit caller-supplied generation floor, previous-snapshot digest and
  timezone-aware clock reject inconsistent, stale, expired or replayed inputs.
- Revoked issuer or subject keys, duplicate roster entries, ambiguous JSON,
  wrong role, changed signatures and digest mismatches reject.
- The original R155 rights-pair signature and exact corpus/train/dev/protected
  commitments are still checked. An R155 `REJECT` remains `SIGNED_REJECT`.

## What this cannot establish

Every test key, snapshot, role and corpus commitment is fabricated. The issuer
public key, generation floor, previous snapshot digest and clock are supplied
by the caller. **An attacker can create a consistent three-key issuer/holder/
steward system.** The success result therefore remains
`REGISTRY_INTEGRITY_ONLY_HOST_ROOT_UNVERIFIED` with
`host_registry_issuer_authenticated=false`,
`host_registry_rollback_fence_authenticated=false`,
`corpus_rights_admitted=false`, `physical_training_authorized=false`, and
`protected_evaluation_authorized=false`.

Independent host installation must separately establish: a protected issuer
trust anchor under owner custody; independently governed rights-holder and
steward identity enrollment; durable monotonically increasing revocation
state and trustworthy time; actual legal/consent review of dataset bytes and
derivatives; protected exclusion comparison; and an enforceable GPU/NAS lease.
None of those can be established by a repository-side fixture or by supplying
more signatures. The `physical_launch_authority.py` hard hold is unchanged.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_rights_registry_evidence_r164 \
  tests.test_rights_pair_evidence_r155 \
  tests.test_ci_portable_source_contract -q
```

Nine registry tests use synthetic Ed25519 keys, and are **SKIPPED** rather
than green cryptographic checks if optional `cryptography` is absent.
The exact-head portable-source CI manifest includes the new suite but does
not substitute for physical host or rights admission qualification.
