# TRAININGDAY W27 source-only integration checkpoint — HARD HOLD

**Evidence class:** local source composition and synthetic tests, **not** a physical training, evaluation, conversion, recovery, or serving run. All work is $0 and does not require a model download, GPU allocation, or installed native runtime.

## Exact input heads

This aggregation starts from `thebrazenbeard/model-training` `main` at `2b83430dd36076c5f65624709968e372b170ea36` and preserves the original PR commits as merge ancestors in this order:

| PR | Exact source head | Included scope |
| --- | --- | --- |
| [#327](https://github.com/thebrazenbeard/model-training/pull/327) | `f8f3e5e2c7def24b98cc422fad197105312f147a` | Official metadata byte witness; not weight or rights admission |
| [#326](https://github.com/thebrazenbeard/model-training/pull/326) | `1adc835b9f5f56054ce1984f265aabfd0a21a93d` | Official inference admission planning and local checks; no download or server launch |
| [#334](https://github.com/thebrazenbeard/model-training/pull/334) | `638386f35d41efa76fd107d5e33592e697292640` | Synthetic six-domain curriculum with fail-closed publication checks; rights not admitted |
| [#330](https://github.com/thebrazenbeard/model-training/pull/330) | `143e964e66dea7c7d27179a210cc833327107d4a` | Toy optimizer replay and private fixture checkpoint persistence; no model autograd |
| [#329](https://github.com/thebrazenbeard/model-training/pull/329) | `8364b3788d715fb38f9caf6edf1a3277bd8e0230` | Caller-supplied public five-arm matched evaluator; not protected Q6 |
| [#328](https://github.com/thebrazenbeard/model-training/pull/328) | `18940552e027d8e43e57b2ffe8a299650a123a95` | Separate random/tiny CPU LoRA, GGUF and loader harnesses; not official weights |
| [#332](https://github.com/thebrazenbeard/model-training/pull/332) | `7cfdef3b6eda54a53973639c585431ceb8c86a91` | Distinct R202 synthetic train source, owner-admission request; HARD HOLD |
| [#333](https://github.com/thebrazenbeard/model-training/pull/333) | `afae4505fa0d1f0733e5b392d942a6382cf3b696` | Read-only integration preflight and local GGUF probe source; no launch |

All eight PR heads share this main commit as an ancestor. A detached local sequence merged all eight without textual conflicts (55 changed paths), and `git diff --check` passed. This checks Git composability; it does **not** establish live behavior or semantic coupling among the components.

## One command that actually exists

From the repository root:

```bash
python -m trainingday.integration_preflight --json
```

**Expected exit code: 1**, intentionally. The schema `TRAININGDAY_INTEGRATION_PREFLIGHT_V1` reports `ready=false`, `physical_launch_attempted=false`, `source_metadata_pins_enrolled=false`, local official weight authentication `HOLD`, physical GPU/lease `HOLD`, and restore, data rights, inference, checkpoint, protected evaluation, conversion, and serving as `UNKNOWN`. This command is a read-only source gap inventory. **There is no single executable training → evaluation → checkpoint → recovery → serving pipeline in the combined tree.**

## Integration boundaries and required evidence

| Boundary | What composes at source level | What blocks a single run |
| --- | --- | --- |
| Official identity and inference (#327 + #326 + #333) | Shared pinned source identity and fail-closed planning/preflight | Trusted source enrollment, exact locally checked official model shards, approved custody, real base inference and qualified host process are absent. Source metadata hashes do not authenticate weights. |
| Curriculum (#334 or #332) → optimizer (#330) | Both curriculum producers and toy replay code can import/test independently | Both data sources remain rights/admission HOLD; neither output feeds the replay, which uses synthetic PRNG pseudo-gradients and a synthetic loss rather than tokenizer/model autograd. The two curricula are alternative unadmitted candidates, not one admitted training corpus. |
| Optimizer (#330) → evaluator (#329) | Independent checkpoint byte reload and five-arm scorer tests pass | Toy checkpoint encodes its own fixture tensors, not a PEFT model adapter. Evaluator demands independently produced matched predictions, train manifest, disjoint transfer cases and five arms; it does not load or run #330 checkpoints. Its gold/predictions remain untrusted caller input. |
| Optimizer (#330) → GGUF / inference (#328 + #333) | CPU random/tiny hybrid model, adapter reload, conversion and loader probes have separate harnesses | #328 does not consume #330 fixture bytes. Sampled tensor-parity checks do not prove complete conversion. Neither #328 nor #333 demonstrates official pretrained LoRA extraction, full official adapter→GGUF provenance, a qualified native server, or trained Vera conversation. |
| Checkpoint → recovery (#330 + #333) | In-process fixture fsync/no-clobber/reload checks and read-only abandoned-stage inventory exist | No independent destination restore, source-authenticated crash orphan ownership, cross-host snapshot recovery, or durable checkpoint chain; inspecting a staged filename is not authority to delete/recover it. |

### Local synthetic test receipt

At the eight-PR local source candidate after #334 W28 final head (`4a2de22ae33786e199951e835d9eb2ac099aa01e`):

```text
python -m unittest discover -s tests -p 'test_trainingday*.py' -q
Ran 230 tests — OK (skipped=3)

python -m unittest -q tests.test_real_qwen35_inference tests.test_optimizer_pilot tests.test_lowmem_experiment
Ran 43 tests — OK

python -O -m unittest -q tests.test_trainingday_original_curriculum
Ran 15 tests — OK

git diff --check origin/main HEAD
PASS

python -m trainingday.integration_preflight --json
exit 1, intentional HARD HOLD
```

All tests ran inside a fresh detached, independently fetched local Git clone. No physical pretrained optimizer steps, qualified GPU, model-server launch, live GGUF load, independent evaluator receipt, or official model byte readback were performed by this integration run. Native CI outcomes are separate evidence for the exact individual PR heads and must not be substituted for this local composition test.

## Hostile falsifiers and decision

> A dataset generator's provenance fields or deterministic checksum cannot grant data rights. Admission is still required for **each** proposed training/dev split.

> A toy optimizer's changed bytes and successful reload do not prove trained model behavior; its binary fixture format is not the PEFT adapter format exercised in the GGUF harness.

> Caller-provided evaluator outcomes can be fabricated even if every scorer assertion passes. Q6/protected transfer remains HOLD without an independent evaluator, sealed bank, matched controls, and real checkpoint lineage.

> A local diff/test PASS cannot prove restored checkpoint durability after process crash, conversion correctness across every tensor, or that a real server binds to its verified files.

**Review decision:** SOURCE COMPOSITION SURVIVES (no textual conflict, synthetic regression green); ONE-RUN PIPELINE FAILS (no artifact handoff/admission/runtime chain); physical feasibility and model improvement remain UNKNOWN. Keep the authority gates fail-closed. Do not relax them to make this integration 'green'.

## PR provenance / retirement map

This single draft aggregate is a **review destination**, not an authorization to merge or close the source branches. All eight PRs above remain independent and open pending source-author and hostile-review reconciliation. If the aggregate is later accepted, maintain a PR-by-PR receipt of exactly which source head it supersedes; do not silently retire unreviewed or changed heads. Any subsequent source PR change requires a new aggregate merge/test receipt. A separate Vera serving adapter PR (if considered) needs its own explicit integration and physical readback; it is not silently included here.
