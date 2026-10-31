# R34: frozen selected-ID V2 gate at R18 MeZO preflight (SOURCE ONLY)

This is an **additive preflight gate**, not an R18 benchmark/schema migration,
GPU lease, or demonstration of Qwen3.5-4B neural improvement.

## Why

The legacy V1 R18 plan commits the selected IDs by joining them with `\n`.
Different ordered ID lists can share that digest, for example
`["a\nb", "c"]` and `["a", "b\nc"]`. R32's bridge and production
admission supply the domain-separated, ordered JSON V2 digest, but the old
MeZO entrypoint did not consume it.

## Implemented boundary

`trainingday.r18_mezo.validate_source()` now requires
`--selection-v2-binding PATH` and checks it after the R18 V1 plan/manifest
custody checks and **before importing torch or touching CUDA**.

The separately frozen JSON sidecar (schema
`VERA_TRAININGDAY_R34_R18_MEZO_SELECTED_V2_FROZEN_BINDING_V1`) binds the
validated V1 plan SHA, manifest SHA, ordered typed V2 selected IDs, the
legacy digest (disclosed as ambiguous), row count, and source admission
status. The gate rejects a missing/symlinked/unbounded/malformed sidecar,
duplicate JSON keys, wrong schema/pins, selected-ID collisions, altered
counts, holdout markers, and non-train source rows. For the production
admission schema, the V2 digest must ALSO match its original admission
receipt; legacy-only production receipts cannot pass.

The producer-facing `make_frozen_selection_v2()` creates a **declaration**
from separately admitted public source rows and a previously validated R18
plan SHA. It does not authenticate the author or freeze the output. A real
operator must obtain a custodian-held, immutable pre-run sidecar, not
generate or replace it after training begins. Binding file bytes are SHA-256
reported in the source preflight; historical V1 plans can still be inspected,
but cannot start a **new MeZO** training operation through this entrypoint
without the V2 sidecar.

The R18 MeZO synthetic fit-child launcher propagates the sidecar path. Its
synthetic probe path remains separate from training source admission.

## Limits / next gate

- R18 QLoRA remains on its legacy-only selected-ID check because a separate
  R34 specialist owns QLoRA runtime edits. Full optimizer-pair parity and
  comparable neural-learning claims remain **HOLD**.
- Historical R18 runs have no retroactively authenticated V2 sidecar.
- The new sidecar itself is a source declaration. Independent custodian
  attestation, strict OS read-only custody, and a dual-arm frozen-plan
  migration are not established here.
- No Lappy/VN1/VN2 host probe, new MeZO/QLoRA run, CUDA allocation, protected
  OOD bank access, fresh-process real 4B adapter reload, tensor delta,
  or behavioral improvement was performed.
- R18's historical 531-second MeZO budget versus 19,813.562-second elapsed
  process remains **TIME_BUDGET_OVERRUN_INVALID** as equal-time evidence.

Run public CPU tests with:

```sh
python -m unittest -v tests.test_r18_mezo tests.test_r34_r18_mezo_selection_v2
```

The source-only test fixture explicitly mints a synthetic sidecar; it is
not proof of independent freeze custody. New physical runs need both the
real external sidecar and an authorized, unconflicted GPU lease.
