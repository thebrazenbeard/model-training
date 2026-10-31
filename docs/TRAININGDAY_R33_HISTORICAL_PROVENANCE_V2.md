# TRAININGDAY R33 — historical/current corpus provenance V2 (public synthetic shadow)

R28 could bind one newly declared training shard to one candidate adapter and
checkpoint, but it could not bind a previous-generation base model, adapter,
checkpoint and evidence to their **different historical training corpus**.
Its explicit `previous_baseline_lineage_verified=false` remains correct.

`trainingday.r33_historical_provenance_v2` adds a separate public historical
graph with these declared roles:

- Exactly one `BASE_MODEL` root, with a caller-declared model-byte digest and
  rights/license declarations.
- One or more `HISTORICAL_SHARD` roots, separately owned from the R26 current
  `TRAIN_SHARD`.
- Derived `HISTORICAL_ADAPTER`, `HISTORICAL_CHECKPOINT`, and
  `HISTORICAL_EVIDENCE` nodes. Their checked ancestry must include the base
  model and historical corpus sources. Checkpoints require **exactly one direct
  adapter parent**, and evidence requires **exactly one direct checkpoint
  parent**. The baseline-bound checkpoint and evidence must additionally have
  the baseline-bound adapter and checkpoint as their respective **direct**
  parents; a merely indirect ancestor is insufficient. Extra unbound
  historical shards fail.
- A frozen historical **no-event** graph commitment and later source-specific
  revocation, privacy, or license-change events. Historic taint flows into the
  proposed new learning-cycle shadow; a clean current source cannot erase it.
- An explicit cross-graph claim linking the historical checkpoint, declared
  current shard, and candidate adapter. The current side still uses the existing
  R26→R27→R28 single-*current*-shard assessor without changing its contract.

This is intentionally a **separate schema and new code path**, not a rewrite
of R26 or R28 or an automatic migration of any real training runner. A
historical shard and newly admitted shard may have genuinely different source
hashes. They do not need to be falsely declared as the same dataset.

## Run public CPU tests

```bash
python -B -m unittest -q tests.test_r33_historical_provenance_v2
python -B -m unittest discover -s tests -p 'test_r[23]*.py' -q
```

Negative cases include incorrect frozen base/previous checkpoint/evidence
hashes, swapped current source or previous checkpoint cross-graph references,
disconnected extra historical shards, historical and current revocations,
wrong source role, source/lineage cycles, malformed events, and mutation at
an assessor boundary. An independent hostile review of the first R33 commit
found a false-clean case where a checkpoint named **two sibling adapter
parents** or evidence named **two sibling checkpoint parents**: the original
kind-set test erased multiplicity. Both are now rejected even when the caller
repins the historical graph commitment. A third test rejects replacement of
the previous checkpoint by a descendant checkpoint merely because the
original still occurs in its ancestry. The assessor copies exact built-in,
bounded data shapes before any JSON serialization or downstream callback.
Its frozen dataclass snapshots reject undeclared dynamic fields, custom
copy hooks and oversized strings or integer values (including a 70K
`issuer_role` field), and the wall-clock argument must be an exact bounded
integer.

## Evidence and effect boundaries

Everything here is an *untrusted caller declaration*. A self-consistent graph
can be constructed by a single malicious caller. All SHA-256 strings are
syntactic commitments; no artifact bytes, source licenses, owner signatures,
physical model weights, optimizer updates, protected evaluation cases or
active host leases are independently observed. This private in-process clone
is **not** an atomic capture of a concurrently mutating filesystem or remote
rights registry.

`synthetic_cross_generation_lineage_consistent=true` means only that the
declarations are internally consistent under this v2 contract. The result
always contains:

- `status=HOLD`
- `previous_baseline_lineage_verified=false`
- `cross_graph_edge_independently_verified=false`
- `historical_corpus_rights_independently_attested=false`
- `current_corpus_rights_independently_attested=false`
- `consent_independently_authenticated=false`
- `gpu_start_authorized=false`
- `continuous_weight_update_authorized=false`
- `model_promotion_authorized=false`
- `model_deployment_authorized=false`
- `protected_eval_unseal_authorized=false`

An actual learning-cycle gate would need an independently committed historical
artifact graph, trusted source/rights custodian, exact-byte model and corpus
hashes, separately held evaluation, OS-enforced training lease, real adapter
weight-delta/checkpoint proof and an explicit owner-governed training permit.
Nothing in this module may unseal a bank, train, merge, promote or deploy.
The invalid historical R18 MeZO time-budget overrun remains invalid for a
fair MeZO/QLoRA physical comparison.
