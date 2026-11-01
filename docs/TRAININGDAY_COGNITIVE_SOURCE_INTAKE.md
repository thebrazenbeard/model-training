# R154 — Cognitive SFT source intake (source-only, non-authorizing)

`trainingday/cognitive_sft_intake.py` is an **opt-in, read-only inspection
boundary** for the labeled synthetic cognitive-perspective TRAIN export in
Labyrinth Draft PR #104. It is deliberately independent of the older numeric
R16/R18 contracts. It does not load Qwen, adapters, `torch`, a model checkpoint,
or a training host, and does not calculate gradients or touch protected exam
contents.

Even a structurally valid, tokenizer-parity-clean caller-supplied export has
exactly one response state in R154:

`UNVERIFIED_REJECTED`

`SOURCE_PREPARED_NOT_ADMITTED_FOR_PHYSICAL_TRAINING` is a **reserved future
status that this module never emits**: it requires independently authenticated
canonical producer replay, not self-reported Git labels and hashes. A consumer
must not interpret the returned rejection summary as admission to physical
train, replay, exam, promotion, graduation, or any weight update. The report
contains dataset and source
commitment hashes, counts, and negative authority flags only; never rows,
prompts, answer choices, token sequences, or supervised labels. There is no
CLI for printing/streaming gold.

## Immutable source pin and distinct trust domains

- Producer repository: `thebrazenbeard/labyrinth`, Draft PR #104, exact Git
  head `b69502c4718333c6055a69817110e2bd061a6346`, tree
  `0c7038397f7fe65fbf781be2237a34cb5a93633f`.
- Producer row schema: `labyrinth.cognitive-supervised-train-example.v1`;
  provider `synthetic_cognitive_perspective_train_v1`; rubric
  `perspective_choice_train_v1`. Only the exact fourteen canonical fields and
  `split=train` are accepted. `eval`, `dev`, `test`, holdout fields, rights
  assertions, extra metadata, and legacy numeric schemas are refused.
- Consumer schema: `vera.trainingday.cognitive-sft-source-intake.v1`, entirely
  separate from frozen R16/R18; this source branch does not change shared CI,
  physical launch, numeric loaders, private data, or a model launcher.
- The producer head above **predates a strict whole-context tokenizer parity
  repair** being developed independently on the Labyrinth side. It is a fixed
  source-analysis pin, not proof that the export is qualified for real Qwen
  tokenization or host training. Updating the producer requires independent
  review and a versioned source pin change; no silent moving-branch trust.

Git head/tree labels are checked for equality against constants, but an
untrusted argument is **not an authenticated Git read**. The calling operator
must independently acquire and authenticate the producer tree and exact export
SHA; `dataset_sha256` and `independent_dataset_sha256` are checked against the
same canonical bytes, but self-supplied hashes cannot establish provenance.
There is no authenticated producer-replay broker on this branch. **Even if a
caller modifies a synthetic gold choice or source-binding hash and then
recomputes every dataset digest, the result stays `UNVERIFIED_REJECTED`.**
The report explicitly says `producer_replay_verified=false`,
`gold_verified=false`, and `source_binding_verified=false`.

## Source checks

The bounded, in-memory inspector validates:

1. Canonical UTF-8, no repeated/Unicode-ambiguous JSON members, sorted compact
   JSONL serialization, exact field set, terminal line feeds, no blank lines,
   at most 262,144 bytes and 15 source rows.
2. One expected objective ID, contract SHA, and integer cycle across all rows;
   case-to-room ID relation and independent prompt SHA replay.
3. An independently recalculated R90 semantic scenario-and-choice digest,
   including NFKC/casefold normalization, deterministic removal of the
   producer's cosmetic setting phrase, and option-order independence. Repeated
   case, normalized scenario/choice semantics, or operator-forbidden semantic
   digests fail. The operator supplies only forbidden *digests*, never heldout
   exam text or keys. Hash comparison cannot prove every possible natural-
   language paraphrase is distinct.
4. An immutable Qwen3.5-4B source binding with full model revision,
   `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` hashes,
   EOS ID, and chat-template hash. The exact in-memory artifact bytes and
   tokenizer-config template are hashed and compared with the independent
   binding commitment. The injected operator-owned tokenizer's `name_or_path`,
   `init_kwargs["revision"]`, template and EOS must match. This is a local
   source claim; it cannot attest which tokenizer is later loaded on a host.
5. Every candidate completion ` A`, ` B`, ` C` receives distinct, non-EOS
   tokens. For **every row and all three choices**, tokens for the full string
   `prompt + "\nAnswer: " + choice` (special tokens enabled) must exactly equal
   tokens for the special-token-bearing `prompt + "\nAnswer:"` prefix plus
   separately tokenized `" " + choice` (no special tokens). Thus contextual
   merges cannot silently move the supervision boundary. EOS cannot appear in
   the prefix, and the full sequence plus EOS must fit the fixed token budget.

The inspector does **not** return gold-masked token arrays; a future private
trainer must separately implement and verify its own label masking and exact
tokenizer/host identity under an authority broker. Injected tokenizer callbacks
are privileged in-process code: only operator-controlled implementations may
receive labeled TRAIN source. The inspector itself performs no network I/O.

## Missing admission authority — explicit hard hold

Before any future physical cognitive SFT, independent external actors must
issue and verify **all** of the following *outside this function*:

- Rights-holder/data-steward approval for each dataset's actual origin,
  licensing, permitted training use, privacy, and any derivative rights; a
  row-level `rights_review=PASS` string is not evidence and is rejected.
- A producer-side deterministic replay or authenticated signed export with
  exact source tree, data bytes, semantic constraints, and objective binding.
- Externally controlled, independently frozen heldout/exposure exclusions,
  with access control separating TRAIN gold from examination and evaluation.
- Host-mediated, independently attested tokenizer artifact and chat-template
  identity on the actual chosen Qwen revision, whole-context parity, model
  identity, budget, resource permission, and immutable training/evaluation
  custody. No operator-provided SHA or in-process object is sufficient.
- The separate physical-launch and human approval protocol, and independent
  measured evaluation. Existing physical-launch authority remains hard-held.

The R154 report intentionally sets `rights_admitted`,
`producer_replay_verified`, `gold_verified`,
`source_binding_verified`, `heldout_custody_verified`,
`tokenizer_host_attested`, `authorizes_training`,
`authorizes_host_execution`, `authorizes_exam`, and
`authorizes_graduation` to false and `optimizer_steps` to zero. These fields
are not configurable.

## Deterministic verification

`tests/test_cognitive_sft_intake.py` contains only synthetic example prompts
and a fake, in-memory tokenizer; it never fetches a model or a private dataset.
It is discoverable by both the repository's normal pytest job and a bare
Python stdlib runner:

```bash
python -m unittest discover -s tests -p test_cognitive_sft_intake.py -v
```

The suite covers valid non-authorizing output, false provenance/dataset
commitments, semantic duplicates despite rehashed identifiers, heldout
digest exclusion, evaluation/rights-field smuggling, duplicate JSON keys,
noncanonical wire bytes, tokenizer artifact and chat-template mismatches,
invalid token values, and a hostile context-sensitive merge-tokenizer. It
also demonstrates that caller-rehashed forged TRAIN gold, source room digests,
and an empty holdout-exclusion list are **explicitly rejected**, not admitted.

This change does not resolve the divergent model-training Drafts #182 and
#183. It is based on #183's R152 head; their shared CI/base-custody integration
must be reconciled by the parent coordinator before a combined merge.
