# TRAININGDAY R32 — public selected-ID V2 R18 bridge (SOURCE ONLY)

R29 production admission added `selected_id_v2_sha256` because the historical
R18 selected-ID commitment hashes newline-concatenated IDs. Two distinct valid
ordered ID lists, `["a\nb", "c"]` and `["a", "b\nc"]`, share that legacy
hash without a SHA-256 collision. The old digest is still consumed by the
R18 benchmark plan, MeZO, QLoRA, evaluator, structural-eval and Labyrinth
intake paths; removing or redefining it in place would invalidate existing
receipts and create unsafe plan/schema churn.

`trainingday.r32_selection_v2_bridge.assess_v2_selection_bridge()` is an
**opt-in CPU-only source adapter** with no import of torch, model weights,
remote hosts or protected data. It accepts public selected training rows,
the admission receipt, the existing frozen R18 plan, and a separately declared
frozen V2 selection hash. It copies exact builtin JSON-like inputs before
validation, requires the exact production admission schema, checks the legacy
R18 plan and training-content binding, computes
the domain-separated ordered JSON V2 digest, and compares it to both receipt
and frozen declaration. A missing V2 pin or legacy-only receipt fails closed.
It detects the two newline-colliding selections as distinct under V2.

The bridge reports only `synthetic_v2_source_consistent` and always returns
`status=HOLD`. It explicitly returns `r18_runners_v2_migrated=false`,
`independent_custody_verified=false`, and all neural training, GPU, model
promotion and private evaluation effect flags false. The separately declared
hash is **not** an authenticated independent custodian signature. An attacker
who controls all declarations can repin both source and purported frozen hash.

This source work **does not** modify or migrate `r18_benchmark.py`,
`r18_mezo.py`, `r18_qlora.py`, `r18_eval.py`, `eval_heldout.py` or
`labyrinth_intake.py`. The historical plan schema still contains only the
ambiguous legacy digest; the real runners must not be called V2-qualified.
Future migration requires an independent R18 plan/corpus owner, separate
public/protected eval custody, source change review and exact-runner receipt
compatibility tests before a protected training run may be considered.

Run public regressions:

```bash
python -B -m unittest -q tests.test_r32_selection_v2_bridge
python -B -m unittest -q tests.test_r32_selection_v2_bridge tests.test_r29_production_admission_strict tests.test_r28_input_adversary
```

The original R18 MeZO time-budget overrun remains invalid as an equal-time
MeZO/QLoRA comparison. No neural-weight improvements, physical batch fit,
fresh-process adapter persistence, authentic rights or release are asserted.
