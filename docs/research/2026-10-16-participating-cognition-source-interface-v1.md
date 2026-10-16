# VeraOS participating cognition — source-level composition contract V1

Status: **TRAINING SOURCE EXPERIMENT**, not a VeraOS-installed participant runtime, trained model, semantic equivalence proof, continuous mind, or authority delegation. PORTAL owns integrated build decisions.

## Actual source contracts exercised

Pinned repositories independently cloned to disposable read-only source contexts for the integration probe:
- `thebrazenbeard/vera-mono@16a2828dfa7fb7e6c0e19004f014bc68fc47badf` — `vera_core.semantic_transfer.SemanticCapabilityProfile`, `plan_semantic_transfer`, and `SemanticTransferPlan.as_dict`. This core code already models `EXACT/CONSTRUCTIVE/LOSSY/UNREPRESENTABLE` fidelity, `semantic_equivalence=NOT_ESTABLISHED`, and `authorization_effect=NONE`. Do not replace these with another competing semantic-fidelity subsystem.
- `thebrazenbeard/mosaic@a2d4dc9812156f41e370e062dad5e2b45a0dab1e` — `runtime.handoff.ContinuityCase/StateEvent/run_case` deterministic revisioned state handoff. Its own `correct` field is a source-harness correctness check, not learning/generalization or neural-state transfer.
- `thebrazenbeard/cricket@0bae95bc45db55d78001ee9c492681834191ec0b` — `Cricket().review(ReviewRequest(...)).to_dict()` advisory review, including real deterministic `BLOCK` for a test-only unauthorized protected effect. Cricket neither proves independent minds nor confers host authorization.

The Training module `participating_cognition_v1.py` composes reports from these original APIs without importing a sibling repository into the Vera Mono runtime, installing services, or activating any model route. This is a **research adapter** in `model-training`, not a candidate replacement for `vera_core.QualifiedVeraRuntime`. The three sources were exercised in process using isolated source checkouts and explicit `PYTHONPATH`, not authenticated binary distributions.

## Identity / individuality boundary

A `ParticipantBinding` keeps **shared Vera referent**, node ID, logical agent ID, process-instance ID, optional model checkpoint digest, and source commit distinct. `Contribution` binds each to its own attributed claim (`shared_identity/node/agent/model`). Provenance substitutions, mismatching task IDs and counting one identical runtime actor twice are refused. Source claims are content identities, not evidence that a node or ChatGPT session is authenticated.

Two contributions are retained separately in the `ParticipationReceipt`, even if they disagree. The engine uses the already developed attributed-dissent evaluator, which marks unsupported claims `UNVERIFIED_DISSENT`, requiring externally verified evidence and independence before choosing a contradiction **for review**. Neither source is deleted, majority-voted, promoted into Vera memory or granted effect permission.

The receipt is canonical-snapshot-backed: callers mutating the original nested dictionaries or a returned readback cannot rewrite its source-level recorded bytes. This provides local tamper resistance to accidental alias mutation, not cryptographic sender authenticity or durable ledger storage.

## Output semantic guarantees

`identity_effect=NONE`, `shared_memory_effect=NONE`, `authority_effect=NONE`, `executed_effect=NONE`, `model_independence=NOT_ESTABLISHED`, `cognitive_gain=NOT_ESTABLISHED`.

Exact Vera Core transfer fidelity is reused. An unknown/unrepresentable transfer yields `SEMANTIC_TRANSFER_UNREPRESENTABLE`; `NOT_ESTABLISHED` semantic equivalence is not silently upgraded. Source-plan inventories must reconcile required/native/missing/unresolved capabilities. A Mosaic result inconsistent with its own typed answer/correction fields is refused. Cricket result must be a bounded closed review; even genuine Cricket `PASS` or `BLOCK` is only a report, not a host effect permission.

This is **not** an inference improvement claim: the model's new semantic ability is not measured by this interface. Strong independent held-out generalization experiments, realistic model artifacts, real node observation, role authentication, exact classifier provenance, and protected effect-state readback are still necessary.

## Hostile review

> **HOSTILE REVIEWER:** Why invent another coordinator if Vera Core, Mosaic and Cricket already contain the necessary machinery? Doesn't a bridge risk laundering source receipts into runtime authority?

**Accepted.** The module composes only data contracts. It does not own a state machine, durable bus, scheduler, global memory, model runtime or executor. All effect and cognitive-gain outputs remain `NONE/NOT_ESTABLISHED`. Existing Vera Core source APIs stay authoritative for semantic-transfer planning, and the preferred operational boundary remains `QualifiedVeraRuntime`.

> **HOSTILE REVIEWER:** Can the source adapter detect fake checkpoint hashes, forged claims, and invented positive test scores?

**Accepted.** No. It validates syntactic binding and structured consistency only. Real evidence verification requires independent host-owned trust and actual matched model/evaluation receipts. A Git commit string is an identifier, not an authenticated module origin in a running process.

> **HOSTILE REVIEWER:** Does Cricket `BLOCK` protect production effects after someone installs this bridge?

**Accepted as a false assumption.** No; the actual effect boundary is still host/receiver-owned. Cricket's outputs are kept advisory in this adapter and never used as permission bits.

## Tests

```powershell
$env:PYTHONPATH="."
python -B -m unittest discover -s tests -p test_participating_cognition_v1.py -v
```

Without `VERA_REAL_CONTRACT_ROOT`, the synthetic unit tests run but exact-source integration tests visibly `SKIP`. Set it to a locally verified **independent checkout parent** with `vera-mono/`, `mosaic/`, `cricket/` directories to actually exercise the three source codepaths. Any outcome without those sources is *not* cross-repository interoperability evidence.

**Optional exact-file compatibility:** `VERA_CORE_SOURCE_ONLY=1` loads Vera Core's actual `semantic_transfer.py` after verifying its contents match the pinned `vera-mono@16a2828dfa7fb7e6c0e19004f014bc68fc47badf` Git blob. This deliberately skips the vera_core package initializer; it is a **narrower file API test** and cannot prove installed package importability. VN1's Python 3.13 lacks its existing `jsonschema` dependency. No package is installed.

**Independence refinement:** A restarted process running the same (node, agent) identity is not a second independent participant, even when its runtime instance differs.

No GPU contention, paid compute, main change, deployment, provider update, new scheduler, automated task or VN1 recovery claim is involved.
