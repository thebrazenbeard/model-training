# TRAININGDAY W27→W31 source-only integration checkpoint — HARD HOLD

## W37 one-time source refresh (2026-11-04) — still HARD HOLD

The historical W31 evidence below remains a fixed receipt. W37 composed its exact
`#335@b0b85052e567ec775959ffc11c49720789b9999a` tree with **three new,
distinct source heads** in an isolated Git worktree, without replaying or retiring
any source PR:

| Source PR | Exact head included | Additional source scope | Admission limit |
| --- | --- | --- | --- |
| [#333](https://github.com/thebrazenbeard/model-training/pull/333) | `c12ff5c802710ce4e329b0aae9fa021e526cb248` | Official Qwen3.5 GGUF source-workflow plan, tests and documentation (3 new paths beyond the W31 #333 ancestor) | Commands and physical execution remain HOLD; native `--version` is not model inference |
| [#336](https://github.com/thebrazenbeard/model-training/pull/336) | `0e3ae19ccaa0cfb1f3f8fc969351a6f5a745a137` | Read-only resource screen, tests and documentation (3 new paths, stacked on #333) | Numeric BF16 memory proxy is not peak-RSS qualification; CLI always declines execution; exact-head hosted CI was absent at W36 review |
| [#337](https://github.com/thebrazenbeard/model-training/pull/337) | `a9aef9025bcc3b1e6556a34d10f9541ac808dd95` | Pinned six-domain curriculum to public five-arm evaluator schema adapter, canonical revalidation and negative tests (3 new paths) | Source-authentic case mapping is not model predictions, unseen-archetype transfer, a rights grant, or protected Q6 |

All three source heads were rechecked open and unmerged before W37 composition.
Each was merged into the isolated `#335` tree without textual conflict. #337's
earlier gold and training-row co-forgery attacks were reproduced and rejected by
the corrected head before this integration; independent review is recorded at
[#5777595141](https://github.com/thebrazenbeard/model-training/pull/337#pullrequestreview-5777595141).
[#336's source-only hostile review](https://github.com/thebrazenbeard/model-training/pull/336#pullrequestreview-5777551244)
explicitly withheld physical and hosted-CI qualification.

W37 local combined-tree checks:

```text
python -m unittest discover -s tests -p 'test_trainingday*.py' -q
Ran 248 tests — OK (3 skipped)

python -m unittest -q tests.test_official_qwen35_gguf_workflow tests.test_qwen35_gguf_capacity_screen tests.test_real_qwen35_inference tests.test_optimizer_pilot tests.test_lowmem_experiment
Ran 63 tests — OK

python -O -m unittest -q tests.test_trainingday_official_inference_admission tests.test_trainingday_r202_admission_candidate tests.test_trainingday_r203_counterfactual_toolstate tests.test_trainingday_public_dev_bridge tests.test_qwen35_gguf_capacity_screen tests.test_official_qwen35_gguf_workflow
Ran 68 tests — OK

python -m trainingday.integration_preflight --json
exit 1 — ready=false, source_metadata_pins_enrolled=false,
physical_launch_attempted=false
```

These groups overlap; do not sum their test counts. The W31 #335 commit's seven
hosted workflows completed successfully against the **older** frozen head; any
W37 update must receive its own exact-head CI readback. No new model execution,
training, rights admission, checkpoint recovery, protected evaluation or live
Vera conversation was attempted in this source composition.

The separate [Bus custody receipt](https://github.com/thebrazenbeard/chat-communication-bus/issues/404#issuecomment-6645416922)
now reports full 26-file public official model checksum/restore verification on
VN2 as well as VN1. That **external host evidence is not an input to** this
read-only `integration_preflight` function and does not silently enroll trusted
metadata, a HostBroker lease, R16 optimizer-resume proof, or evaluator custody.
The full private 41-file R16 checkpoint/control recovery set was not yet
independently restored to VN2 at this W37 source checkpoint; preserve Lappy's
originals until it is and antirollback authority is separately addressed.

**Retirement map:** retain the W31 #316 deduplication decision below. #333,
#336, and #337 are now explicit incoming source dependencies, **not** closure
targets or endorsements of protected merge. A later source head drift requires
a fresh local composition/test witness; do not update this frozen receipt by
assumption.

**Evidence class:** local source composition and synthetic tests, **not** a physical training, evaluation, conversion, recovery, or serving run. All work is $0 and does not require a model download, GPU allocation, or installed native runtime.

## Exact input heads

This aggregation starts from `thebrazenbeard/model-training` `main` at `2b83430dd36076c5f65624709968e372b170ea36`. The initial draft #335 commit preserved the W27 PR heads as merge ancestors; W30 appended new #332 and #326 heads as fast-forward ancestry parents, and W31 appends the newer #326 safety head and #332 R203 source while preserving every earlier commit. The table records the **bounded W31 source snapshot**:

| PR | Exact source head | Included scope |
| --- | --- | --- |
| [#327](https://github.com/thebrazenbeard/model-training/pull/327) | `f8f3e5e2c7def24b98cc422fad197105312f147a` | Official metadata byte witness; not weight or rights admission |
| [#326](https://github.com/thebrazenbeard/model-training/pull/326) | `a48e2fcd4a0c850dcf65fc5526a10601c96d0d00` | Official inference admission planning, retained-FD and full-root pathname rewalk; no download or server launch |
| [#334](https://github.com/thebrazenbeard/model-training/pull/334) | `638386f35d41efa76fd107d5e33592e697292640` | Synthetic six-domain curriculum with fail-closed publication checks; rights not admitted |
| [#330](https://github.com/thebrazenbeard/model-training/pull/330) | `143e964e66dea7c7d27179a210cc833327107d4a` | Toy optimizer replay and private fixture checkpoint persistence; no model autograd |
| [#329](https://github.com/thebrazenbeard/model-training/pull/329) | `8364b3788d715fb38f9caf6edf1a3277bd8e0230` | Caller-supplied public five-arm matched evaluator; not protected Q6 |
| [#328](https://github.com/thebrazenbeard/model-training/pull/328) | `18940552e027d8e43e57b2ffe8a299650a123a95` | Separate random/tiny CPU LoRA, GGUF and loader harnesses; not official weights |
| [#332](https://github.com/thebrazenbeard/model-training/pull/332) | `8e316eb82fda11713d6250b75dca18640a08562a` | Distinct R202 synthetic train source, corrected WITHHOLD request and R203 synthetic counterfactual tool-state probes; HARD HOLD |
| [#333](https://github.com/thebrazenbeard/model-training/pull/333) | `afae4505fa0d1f0733e5b392d942a6382cf3b696` | Read-only integration preflight and local GGUF probe source; no launch |

All eight current PR heads share this main commit as an ancestor. A detached local sequence merged all eight without textual conflicts (55 initial source paths), and `git diff --check` passed. The W30 #332/#326 and W31 #326 updates, plus #332's R203 three-file slice, also merged cleanly (58 current source paths). This checks Git composability; it does **not** establish live behavior or semantic coupling among the components.

## One command that actually exists

From the repository root:

```bash
python -m trainingday.integration_preflight --json
```

**Expected exit code: 1**, intentionally. The schema `TRAININGDAY_INTEGRATION_PREFLIGHT_V1` reports `ready=false`, `physical_launch_attempted=false`, `source_metadata_pins_enrolled=false`, local official weight authentication `HOLD`, physical GPU/lease `HOLD`, and restore, data rights, inference, checkpoint, protected evaluation, conversion, and serving as `UNKNOWN`. This command is a read-only source gap inventory. **There is no single executable training → evaluation → checkpoint → recovery → serving pipeline in the combined tree.**

## Integration boundaries and required evidence

| Boundary | What composes at source level | What blocks a single run |
| --- | --- | --- |
| Official identity and inference (#327 + #326 + #333) | Shared pinned source identity and fail-closed planning/preflight. W31 #326 retains nine original FDs, then rewalks the absolute root-to-leaf snapshot path with `O_NOFOLLOW` and original directory device/inode recheck; the ancestor-rename/symlink swap mutant now HOLDs. | Trusted source enrollment, exact locally checked official model shards, approved custody, real base inference and qualified host process are absent. Sequential pathname checks are not an atomic snapshot or weight custody proof. |
| Curriculum (#334 or #332) → optimizer (#330) | Both curriculum producers and toy replay code can import/test independently. #332 accepts a minimal exact-subject `WITHHOLD` without invented evaluator/exclusion fields, always returning HOLD; it rejects spurious authorization fields. Its R203 synthetic tool-state counterfactual cases provide additional deterministic tests, not real tool execution. | Both data sources remain rights/admission HOLD; neither output feeds the replay, which uses synthetic PRNG pseudo-gradients and a synthetic loss rather than tokenizer/model autograd. The two curricula are alternative unadmitted candidates, not one admitted training corpus. |
| Optimizer (#330) → evaluator (#329) | Independent checkpoint byte reload and five-arm scorer tests pass | Toy checkpoint encodes its own fixture tensors, not a PEFT model adapter. Evaluator demands independently produced matched predictions, train manifest, disjoint transfer cases and five arms; it does not load or run #330 checkpoints. Its gold/predictions remain untrusted caller input. |
| Optimizer (#330) → GGUF / inference (#328 + #333) | CPU random/tiny hybrid model, adapter reload, conversion and loader probes have separate harnesses | #328 does not consume #330 fixture bytes. Sampled tensor-parity checks do not prove complete conversion. Neither #328 nor #333 demonstrates official pretrained LoRA extraction, full official adapter→GGUF provenance, a qualified native server, or trained Vera conversation. |
| Checkpoint → recovery (#330 + #333) | In-process fixture fsync/no-clobber/reload checks and read-only abandoned-stage inventory exist | No independent destination restore, source-authenticated crash orphan ownership, cross-host snapshot recovery, or durable checkpoint chain; inspecting a staged filename is not authority to delete/recover it. |

### Local synthetic test receipt

At the bounded W31 local composition of exact #326 `a48e2fc` and #332 `8e316eb` heads, including the two-line #316 workflow gate (`31be43d7e518759d4cc085028d8c008976633843`, before updating this document):

```text
python -m unittest discover -s tests -p 'test_trainingday*.py' -q
Ran 243 tests — OK (skipped=3)

python -m unittest -q tests.test_real_qwen35_inference tests.test_optimizer_pilot tests.test_lowmem_experiment
Ran 43 tests — OK

python -m unittest -q tests.test_trainingday_official_inference_admission tests.test_trainingday_r202_admission_candidate tests.test_trainingday_r203_counterfactual_toolstate
Ran 43 tests — OK

python -O -m unittest -q tests.test_trainingday_official_inference_admission tests.test_trainingday_r202_admission_candidate tests.test_trainingday_r203_counterfactual_toolstate
Ran 43 tests — OK

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

**Explicit #316 deduplication owner:** stacked draft [#316](https://github.com/thebrazenbeard/model-training/pull/316) at `cca0de0e90b9ce26a08621bfc64173b49cbd3d07` is **not included as an ancestry parent**. Its six changed paths all overlap the union of main-targeted #328 and #333. Three real-GGUF source/test/document blobs are identical. The two-line #316 workflow step that runs `tests.test_real_qwen35_inference` early in the GGUF CI job has been ported into this aggregate and **that fourth blob now also matches exactly**. The two remaining comparator code/test blobs differ: #328 demands a bounded 64-token server witness and full-HF-mass validity rather than #316's top-12-only witness, and adds hostile mutants; neither covers the server's whole vocabulary. Keep #316 open until its owner reviews those behavioral differences; path overlap alone is not proof of safe retirement. The main-targeted source owners for this aggregation remain #328 (synthetic CPU parity) and #333 (real-GGUF preflight); #316 is the explicitly tracked *deduplication-review target*.
