# Training: attributed dissent, false challenges, and safe source inventory V1

**Status:** synthetic research instrumentation, not validated cognitive independence, live model evaluation, host source attestation, authority, or deployment.

## Problem and conceptual referents

The four-chat and Mosaic/Cricket designs need to distinguish the shared logical identity **Vera**, physical nodes (Lappy/VN1), local actors/agents, and replaceable model artifacts. The strings are separate. A changed role label, multiple nodes, two model copies, and two different hypotheses do not automatically produce independent evidence or two minds. Cricket's existing contract explicitly distinguishes critique, disagreement and verification. Mosaic's current main remains a harness for specialist composition; no trained model partition behavior is proven.

The `independent_challenge_eval_v1.py` prototype takes two attributed source claims. It preserves *both* records and their canonical content hashes, including contradictory values. A material challenge is only **selected for further review** when exact referent, predicate, scope, revision, comparable observational claim type, receiver-verified evidence, and receiver-verified independence all match. Differences in scopes, claims, revisions or evidence groups remain visible, but are not treated as accepted global contradictions. All classifications have `effect_authorized=false`, `accepted_as_fact=false`, `model_independence_proven=false`.

The development fixture `training/attributed_dissent_devset_v1.json` has 15 explicitly synthetic paired situations: two independently trusted-evidence conflicts and thirteen negative controls. A naive rule, "different agent IDs with different values means challenge", produces **11 false challenges out of 13 negative cases**; the provenance-aware source-level gate produces **0/13 false challenges and 2/2 selected conflicts**, *under its deliberately injected synthetic truth/evidence callbacks*. With no independent verifier the two conflicts remain **UNVERIFIED_DISSENT**, not falsely accepted. This small hand-authored fixture is not a held-out benchmark and does not establish generalization, cognitive independence, performance against real agents, or the quality of reviewer prose.

Nested semantic values are captured to immutable canonical bytes at claim admission. Mutating the caller's original value or a returned mapping cannot silently rewrite an accepted claim digest. A content digest is not signer authentication.

## Source mirror without executing potentially unsafe content

The `source_tree_manifest_v1.py` prototype reads an **exact pinned Git commit** via `git rev-parse` and `git ls-tree -r -z` in an existing local checkout. It emits *only* typed Git object identifiers, source path inventory, modes, present/missing folder flags, and a hash of the metadata manifest. It does **not** checkout, open, import, execute, package, copy, or publish donor/source file contents. Tree entries marked executable, symlink, or submodule remain metadata only. Dirty working-tree changes cannot alter pinned commit inventory.

For the requested 44-folder V-E-R-A lineage, PORTAL/Integration should first produce an exact-commit **folder coverage and provenance inventory** instead of mirroring potentially flagged executables merely to fill a count. Missing components are **MISSING**, not quietly represented as present or installed. Source metadata is not a safety scan, license to republish, semantic compatibility, runtime installation, or authenticated origin. Any actual protected publication/installation still requires a separately authorized, reviewed plan.

## Hostile review and corrections

> **HOSTILE REVIEWER:** An evidence-group label can be fabricated, and a verifier callback in the source tests approving synthetic digests proves nothing about authentic independence.

**Accepted.** The gate defaults to fail-closed unverified dissent and requires receiver-owned external verification. Test callbacks intentionally provide synthetic fixtures, not real attestation. This distinguishes evaluation mechanics from deployed runtime verification.

> **HOSTILE REVIEWER:** All observed contradictions aren't necessarily true; the stronger reviewer might simply be confidently wrong.

**Accepted.** The gate chooses evidence conflicts *for review* without adopting either claim. Deterministic truth/authority/effect processing remains in the governed receiver. Independent challenge quality must be measured on genuinely held-out, source-verified cases and must include false positives as well as missed violations.

> **HOSTILE REVIEWER:** Listing exact Git tree object identifiers does not establish that the source is safe, runnable, or an authentic upstream release.

**Accepted.** The result explicitly denies authenticity, execution, compatibility, and publication. Metadata inventory is the safe alternative to unreviewed source copying; further qualification is mandatory.

## Reproduction

```powershell
$env:PYTHONPATH="."
python -B -m unittest discover -s tests -p test_independent_challenge_eval_v1.py -v
python -B -m unittest discover -s tests -p test_source_tree_manifest_v1.py -v
```

**Training ownership:** evaluation instrumentation only. PORTAL owns orchestration/acceptance. Integration owns exact source/effect-chain controls and any safe 44-folder mapping. VN1 owns installed host and peer/recovery verification. No merge, training-job, scheduled task, credential change or node activation is part of this research unit.
