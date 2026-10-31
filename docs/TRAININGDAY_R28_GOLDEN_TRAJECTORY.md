# TRAININGDAY R28-01 — bounded public CPU golden trajectory

Status: **SOURCE/CPU TEST ONLY — HOLD**. Branch proposed from the exact
source tree of model-training Draft #157 (`67d9247e281fe26f10085b16f46d82adaa813b63`
remote head). This adds a dynamic optimizer/RNG replay after R27's static F32
checkpoint portability probe. It is **not** a real Vera Qwen3.5-4B training
receipt, GPU run, independent host attestation, or release gate.

## Reproduce safely

Only Python standard library and six invented linear-regression cases are used.
No model download, internet dependency, private corpus, evaluator bank or host
credential is loaded.

```powershell
python -m unittest tests.test_r28_toy_trajectory -v
python -m trainingday.r28_toy_trajectory --seed 721 --steps 10
```

Both commands work as equivalent Python invocations on Linux; on Windows the
`python` executable must be on `PATH`. The CLI writes no files and emits one
JSON document with a per-step trace, local process environment fingerprint,
and a bounded hex-encoded checkpoint. `--resume-checkpoint-hex` takes only a
public-toy checkpoint printed by this module; future `--steps` count is bounded
by the 256-step total. Never pass host credentials or private data as CLI args.

The reference library functions are `new_trajectory(seed)`,
`advance(state, count)`, `save_checkpoint(state)`, `load_checkpoint(bytes)`,
`compare_trajectories(left, right, tolerance=...)`, and
`verify_public_reference_trajectory(receipts, tolerance=...)`.

## Pinned toy optimizer and evidence

| Item | Exact contract |
| --- | --- |
| Model | Two binary64 scalar parameters, zero initialized, public synthetic linear regression |
| Data | Six constant invented `(x0, x1, target)` triples; dataset canonical hex digest bound in every checkpoint and step |
| Sampler | Nonzero seeded 32-bit Xorshift, one sample per step, checkpointed full PRNG state |
| Update | Squared error, gradients `2*error*x_i`, momentum `m_i = 0.125*m_i + 0.0125*gradient`, weight `w_i -= m_i` |
| Budget | 256 total steps, one fixed seed and optimizer state, no hidden gradient batch or schedule |
| Receipt | Step, seed, sample, PRNG, current two parameters, both momentum scalars, before/after squared loss, little-endian binary64 state digest |
| Checkpoint | Canonical bounded JSON with exact `float.hex()` state and SHA-256 payload checksum; strict schema, dataset digest and duplicate-key/encoding guards |
| Resume | Clean second Python process can reload and produce the exact continuation of an uninterrupted local CPU trajectory |
| Comparison | Per-step signed digest equality plus bounded absolute numerical divergence (default **zero**, configurable maximum `0.001`) |
| Fingerprint | Process-reported Python version/implementation, OS family, byteorder, binary64 size and module SHA; no username/hostname/physical host identity |

First step example for seed `721`: sampled toy case `(1, 1, 0.2)`;
weights change from `(0, 0)` to approximately `(0.005, 0.005)`;
the sample's squared error decreases from `0.04` to `0.0361` in that update.
This is a tiny real *toy* optimizer update, not Qwen tensor proof.

## Adversarial and negative acceptance

The unit tests exercise:

1. Fixed-seed byte-for-byte replay and per-step SHA receipts.
2. Split training/restart with full optimizer and PRNG continuity; clean-process reload.
3. Changed seed, input truncation/reordering, wrong sample and wrong RNG rejecting exact comparison.
4. Tampered checksum/canonical bytes, recomputed but wrong dataset commitment, duplicate/invalid scalar boundaries.
5. Recomputed but **wrong-gradient** fake receipt: two matching fake traces compare exactly, but the independent in-process public reference recomputation detects divergence at the wrong step.
6. Explicit finite tolerance handling that cannot conceal wrong sampler/seed or invalid receipt SHA, and never upgrades source tests to model improvement or host attestation.

`compare_trajectories` intentionally compares *declared* receipts. A malicious
issuer can rehash invented bytes. `verify_public_reference_trajectory` directly
recomputes the toy optimizer's expected trajectory from step one to detect
wrong updates, but this is still one source runtime and not an independently
authenticated Windows/Linux host execution. Its output's `EXACT_MATCH`
describes a bounded public numerical comparison only.

## Disallowed interpretations and remaining gate

> A byte match on the same synthetic source is not independently verified
> checkpoint custody; a mathematically reproducible two-scalar CPU optimizer
> is not comparable evidence of 4B MeZO vs QLoRA training. Cross-OS exactness
> must be measured on separately authenticated physical processes and may
> legitimately require a preregistered finite tolerance.

No real Qwen3.5-4B base, LoRA adapter, torch optimizer, WDDM/Modern Standby
mitigation, GPU lease, 1,024-row admitted corpus, independent private OOD
oracle, or matched equal-time control arms are involved here. The historical
R18 MeZO 531-second budget versus 19,813.562-second elapsed run remains
`TIME_BUDGET_OVERRUN_INVALID`. Passing this isolated code cannot merge,
release, train, deploy, unseal data, or claim improved behavior.

This mechanism changes the TrainingDay workflow only when a separately
governed integration/review validates its exact head; source review does not
grant merge or execution authority.
