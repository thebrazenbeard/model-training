# Original curriculum → public evaluator: source-only bridge

`trainingday.public_dev_bridge` maps the exact reviewed `original_curriculum:v1`
TRAIN/DEV JSONL into the input fields expected by draft evaluator #329:

| Producer | #329 consumer |
| --- | --- |
| `dev[].case_id` | `fixture.cases[].case_id` |
| `dev[].answer` | `fixture.cases[].expected_semantic` (public caller gold) |
| `dev[].domain` | `fixture.cases[].stratum` (six diagnostic strata) |
| `dev[].source_id` | `fixture.cases[].source_group_id` |
| `train[].prompt` / family / source / template | `train_rows[]` leakage guards |
| reviewed source manifest and train digest | identified `train_manifest`, **NOT** a training grant |

`build_w31_public_scaffold(train_bytes, dev_bytes, manifest_bytes)` checks
the **pinned** original 432/144 JSONL byte digests and final generator-source
digest, manifest status, row schema, split counts, prompt and grouping exclusion.
It produces *no predictions*. This is deliberately a source-only adapter; it
does not perform weight updates, run a model, or write to any evaluator store.
The scaffold includes a canonical digest to catch accidental mutation between
generation and prediction attachment; it is **not** a cryptographic signature
or independent attestation of a caller's submitted predictions.

After an independent operator provides all five matched per-case predictions,
`attach_independent_predictions(scaffold, five_arm_predictions,
train_jsonl=..., dev_jsonl=..., manifest_json=...)` produces the shape accepted
by `evaluate_public_fixture` in draft #329. The original three byte payloads
are **required again** at attachment, and the entire canonical scaffold is
rebuilt from their pinned SHA-256 digests. Merely recomputing a forged unkeyed
scaffold digest can no longer smuggle altered gold, strata, or TRAIN examples.
Prediction rows must have exactly `case_id` and `semantic_answer`. No
predictions are synthesized or inferred from the public gold. The source-byte
check does **not** independently verify predictions or model execution.

**Hard qualification limit:** #329 uses literal `fixture.split="transfer"`
as a fixed parser token. The producer's six DEV domains all share their rule
archetypes with TRAIN. Therefore this adapter labels its *actual* scope
`SAME_ARCHETYPE_DIAGNOSTIC_ONLY`. A scorer outcome called
`PUBLIC_DEV_DIFFERENTIAL_ONLY` would be a descriptive outcome on caller-owned
inputs, **not** evidence of unseen-archetype transfer or Q6. The published #329
receipt does not independently verify caller gold, predictions, runtime, or
model provenance.

All source examples remain `rights_status=NOT_ADMITTED` and
`training_authorized=false`, even though they were source-authored. The bridge
must never be used to admit data rights, seal a protected bank, or launch
training. For protected evaluation, use a genuinely independent custodian.

This PR is deliberately **main-based and dependency-neutral**. It provides the
adapter contract alone so the source integrations #329/#334/#335 can import it
when reviewed; it does not copy, merge, or supersede those draft PRs.
