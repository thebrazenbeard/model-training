# VERA::TRAININGDAY — Restricted program source custody

This is a **train-only candidate builder**, not proof of model learning or
general Python correctness. It does not execute any downloaded source module.

## Provenance gate

Pass a separately trusted absolute Git checkout for **every** MIT source
repository declared in the pinned Labyrinth v2 manifest using repeated
`--trusted-repo-root OWNER/REPO=ABSOLUTE_CLONE_PATH` arguments. The builder:

1. Checks each checkout's exact `HEAD` against the manifest's 40-hex revision.
2. Uses `git cat-file blob REVISION:PATH` to compare original raw code bytes to
   each selected train row's text and content digest, without importing the
   file or checking out that revision.
3. Reads every repository's pinned Git `LICENSE` blob and requires its exact
   bytes and SHA-256 to match the copied MIT notice in the corpus manifest.
4. Rejects symlink roots, unsafe source paths, stale checkouts, tampered
   source/license bytes, and incomplete source-root mappings.

Without the root arguments, the old source-only build remains possible but
reports **zero Git blob verified files**. It is not allowed to imply upstream
source authentication. A passed Git check proves that bytes match the pinned
local Git commit, not independently that a repository is operated by its
claimed GitHub owner; that identity belongs to a separate source-authority
review.

## Semantic gate

Only a small pure-integer subset of Python functions is accepted. Three AST
evaluation paths are compared: direct, postfix stack, and a third arithmetic
reference using `operator` / `divmod` rather than their shared helpers.
Tests deliberately corrupt the shared arithmetic helper and confirm the
reference detects it. Statement execution still uses a common restricted
control-flow interpreter, so this is not independent verification of general
CPython behavior. Unsupported functions are skipped, not guessed.

## Training admission remains closed

Every output manifest declares `qualified_for_neural_training: false` and
`DERIVED_TRAIN_ONLY_NO_WEIGHT_UPDATE`, even when Git blob checks pass. A
future supervised consumer must independently verify this derivative's
manifest/shard SHA, source ownership and license rights, code-oracle limits,
deduplication, structure-level train/eval exclusion, and tokenizer loss mask
before a bounded optimizer run. The downstream runner must not mistake
source custody or a learned adapter for demonstrated reasoning gains.

The writer creates a fresh `.building` staging directory and atomically
renames it only on success. A failed run deliberately leaves staging artifacts
for review; never re-run into the same path or treat staged data as admitted.
