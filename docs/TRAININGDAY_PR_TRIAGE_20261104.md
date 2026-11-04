# TRAININGDAY integration triage — 2026-11-04 snapshot

**Evidence: live GitHub REST PR listing and exact main readback on this date;
not a deployed runtime probe.** `main` was
`2b83430dd36076c5f65624709968e372b170ea36`. GitHub returned **222 open
PRs, 221 drafts**; 28 had `base=main`. These counts and heads are ephemeral;
refresh before selecting/merging any PR. The repository has no `AGENTS.md`
on the exact inspected main tree; follow `PROTOCOL_V2_CURRENT.md`, project
instructions, and current protected-effect gates.

## Minimal source-to-physical chain

| Order | Interface/evidence | Candidate source | Not established by candidate |
| --- | --- | --- | --- |
| 1 | Official base identity, local weight **and metadata byte** witness | Merged #310 + draft #327 | local byte readback, trusted metadata enrollment, host ownership |
| 2 | Recover correct8/corrupt8/zero adapter and receipts | Historical R16 stacked #140 | independently recovered bytes, rights and checkpoint provenance |
| 3 | **Early, independent** VN1/VN2 destination restore | external artifact transfer | checksum-verified restore and Lappy-independent recovery |
| 4 | Real base-model inference (can occur before new training) | #326 and runtime work | local base-model process/tokenizer/latency readback |
| 5 | Rights-admitted TRAIN/DEV curriculum | #265; matched-evaluation #296 | actual data-owner grant and independent evaluator custody |
| 6 | Exclusive host process/GPU lease and durable anti-replay witness | #319–#321, negative #320 | installed positive-grant broker or independently monotonic Q6 |
| 7 | Physical bounded optimizer/checkpoint/reload | #330 and R16 #140 | authorized optimizer under a present host-owned lease |
| 8 | Protected held-out independent comparison | #296 and #329 | fresh same-budget MeZO vs QLoRA, multiple seeds, generalization |
| 9 | Recovered trained-adapter GGUF conversion | #328; lineage #316 | exact official adapter compatibility and converter validation |
| 10 | Trained Vera conversation through intended UI | #326 + runtime work | live model-serving readback and measured latency/quality |

**Order is a dependency interpretation, not a pass claim.** #328 is synthetic
conversion evidence; it does not prove the recovered adapter converts. An
installed `VeraTrainingDayHostBroker` returning `HOLD` is not a GPU lease.
The source module `trainingday.physical_launch_authority` still denies all
physical starts; `trainingday.pinned_base` currently has no trusted metadata
byte-pin enrollment. No draft PR should be merged merely for reducing counts.

## Consolidation selection rules

1. Prefer exact-main-targeted testable slices with clear files/ownership and
   fresh exact-head CI; require protected review before promotion.
2. Treat PRs with non-main bases as stacked history. Resolve each parent and
   compare the net merge-base diff; don't replay a child diff without its
   required parent.
3. Reject duplicate head/tree promotions. For example, #323 and #324 were
   observed referencing the same source head
   `1aa944c70d59c7227fe2d7f951f15a073c30d87e`; one source head is not
   two independently valuable changes.
4. Retain negative/falsification PRs (#320 replay rollback and #312 Linux
   child-orphan witness) as safety evidence even if their implementation
   cannot yet be promoted. `UNKNOWN` is not `PASS`.
5. Integrate **one vertical path** through exact ownership gates and actual
   host readback. Report tests, live effects, and recovery separately; source
   readiness must not be confused with installed/behavioral readiness.

`python -m trainingday.integration_preflight --json` provides the initial
machine-readable, fail-closed source gap inventory, not a training launcher.
