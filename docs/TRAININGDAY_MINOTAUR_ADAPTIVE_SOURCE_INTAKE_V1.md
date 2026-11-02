# Labyrinth adaptive Minotaur TRAIN source inspection (v1)

The source-only consumer is `trainingday/minotaur_adaptive_source_intake_v1.py`.
It accepts the public `labyrinth.hf-adaptive-source-manifest.v1` payload
from Labyrinth's #94/#95 integration, without loading a model or acquiring a
host, GPU, NAS, rights-holder, data-steward or evaluator grant.

## Deliberate separation

| Route | Purpose | Source size | Admission |
| --- | --- | --- | --- |
| Minotaur adaptive v1 | Opt-in feedback-routed tier-one arithmetic TRAIN | 2–32 rooms, 1–128 visits | Public source-shape diagnostic only; `UNVERIFIED_REJECTED` |
| Cognitive SFT | A/B/C cognitive perspective TRAIN | 15 cases | Separate `cognitive_sft_intake.py`; no interchange |
| R185 arithmetic | Frozen numeric arithmetic path | 1,024 cases | Separate rights/evaluator/host custody and existing gates |

The consumer pins an **operator-asserted** Labyrinth #94/#95 source revision
`20c48b89ce359c43313e8a0fc3aec4d4023d2bd6`, tree
`c01920ae7a5fea9e3bc9cf180cdfead61b953da1`. These are source references,
not proof that an authenticated host produced the supplied bytes.

```python
from hashlib import sha256
from trainingday.minotaur_adaptive_source_intake_v1 import (
    PINNED_LABYRINTH_HEAD, PINNED_LABYRINTH_TREE,
    inspect_adaptive_minotaur_source,
)

result = inspect_adaptive_minotaur_source(
    manifest_bytes=public_manifest_bytes,
    expected_manifest_sha256=sha256(public_manifest_bytes).hexdigest(),
    producer_git_head=PINNED_LABYRINTH_HEAD,
    producer_git_tree=PINNED_LABYRINTH_TREE,
    explicit_adaptive_opt_in=True,
    staged_roster_present=False,
    forbidden_case_ids=(),  # Local independently maintained exclusions if available.
)
assert result["status"] == "UNVERIFIED_REJECTED"
assert not result["model_training_granted"]
```

No CLI runner, training import, runtime hook or GitHub Action consumes this
result. It is designed for offline source inspection, never launch admission.

## Implemented source checks

- Strict UTF-8 JSON (duplicate keys and nonfinite numbers rejected), 256 KiB
  raw-input ceiling and 128 KiB canonical payload ceiling.
- Exact public schema keys and bounded scope, cycle-zero, tier-one budget,
  2–32 TRAIN rooms, 2–32 public diagnostic EVAL size, 1–128 optimizer visits.
- Explicit adaptive-mode opt-in and rejection of a simultaneous staged roster.
- Internal self-hash domain verification plus raw-file SHA-256 consistency.
- Unique room, case and prompt digests; room/case ID binding; expected
  planner-source stage; per-room feedback correspondence and checkpoint chain.
- Exact source-level optimizer exposure ordering derived from the seed and
  per-epoch tier-one shuffle; bounded canonical TRAIN case-ID exclusion
  intersection if supplied (at most 1,024 unique IDs).
- Public report includes only status, boolean trust flags and counts. No prompt,
  answer, answer hash, feedback bit, per-case digest, checkpoint, private DEV
  label, source contents or raw case identifier is emitted.

## Non-evidence / remaining custody

> An attacker can forge the source, all digests, producer revision claims and
> feedback. The final feedback success bit is especially uncheckable without
> the actual response and trusted scorer: there is no subsequent routing
> decision to constrain it. `structural_source_consistent=True` is only a
> source-format diagnostic, not the Labyrinth policy-replay proof.

The producer's private `reconstruct_adaptive_source` enforces additional
deterministic Minotaur policy replay and train/heldout arithmetic-semantic
disjointness. **This consumer performs neither of those checks:** a fabricated
room/tactic history with a recomputed self-hash can pass its structural-only
inspection. Neither the producer's replay nor the public self-hash establishes an
independent immutable artifact or authenticated evaluator custody. This
consumer therefore intentionally does **not** claim any Q6 monotonic witness,
rights-holder/data-steward grant, signed tokenizer/weights provenance,
protected DEV control, Windows-owned process custody, GPU lease, or physical
optimizer step. **All grants remain false and status remains
`UNVERIFIED_REJECTED`, even for valid source shape.**

Before any future physical-training consumer exists, a separate host-owned
signed/append-only evidence chain and evaluator-controlled independent Q6
check must be designed, implemented, challenged and qualified. Nothing in
this source diagnostic permits bypassing those gates.

## Verification

```sh
python -m unittest tests.test_minotaur_adaptive_source_intake_v1 -v
```

The regression suite covers valid-but-untrusted shape, self-hash/producer
claim mismatch, malformed/extra keys, private answer field injection,
duplicate JSON keys, missing opt-in, staged-mode collision, room/feedback/
exposure drift, forbidden case overlap and recomputed-final-bit co-forgery.
An interoperability smoke probe with the exact Labyrinth producer fixture
accepted 3 rooms and 5 visits structurally while keeping every grant false.

For a repeatable cross-repo smoke test, check out Labyrinth at the pinned
head/tree above, then from this repository's root run the following with
`LABYRINTH_SOURCE_DIR` set to that checkout's absolute path. The fixture
builders are Labyrinth's own test helpers; this command does not log TRAIN
answers, feedback or serialized manifest content:

```sh
PYTHONPATH="$LABYRINTH_SOURCE_DIR:$PWD" python -c '
import hashlib, json
from tests.test_minotaur_hf_roster_bridge import _config, _runner, _scope
from tests.test_journey_policy_runner import toy_arithmetic_learner
from labyrinth.minotaur_hf_roster_bridge import collect_minotaur_hf_roster
from labyrinth.hf_adaptive_source_v1 import build_adaptive_manifest
from trainingday.minotaur_adaptive_source_intake_v1 import (
    PINNED_LABYRINTH_HEAD, PINNED_LABYRINTH_TREE,
    inspect_adaptive_minotaur_source,
)
config = _config()
roster = collect_minotaur_hf_roster(_runner(), config, toy_arithmetic_learner)
raw = json.dumps(build_adaptive_manifest(config, _scope(), roster)).encode()
result = inspect_adaptive_minotaur_source(
    manifest_bytes=raw, expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    producer_git_head=PINNED_LABYRINTH_HEAD,
    producer_git_tree=PINNED_LABYRINTH_TREE,
    explicit_adaptive_opt_in=True, staged_roster_present=False,
)
assert result["structural_source_consistent"] is True
assert result["train_room_count"] == 3
assert result["optimizer_exposure_count"] == 5
assert result["status"] == "UNVERIFIED_REJECTED"
assert all(not value for key, value in result.items() if key.endswith(
    ("_authorized", "_authenticated", "_verified", "_granted")))
print("source-shape interoperability PASS; training HOLD")
'
```
