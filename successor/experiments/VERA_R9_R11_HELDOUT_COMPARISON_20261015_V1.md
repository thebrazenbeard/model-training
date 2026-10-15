# Vera Qwen3.5 R9/R10/R11 held-out development comparison

Date: 2026-10-15 (Lappy). Classification: LOCAL DEVELOPMENT TEXT PROBE + LOCAL MECHANICAL AUDIT.
These are not independent semantic-review results or deploy/activation qualifications.

## Exact subjects

- Canonical latest training branch parent: `48dc9c242fa51b4f47daac925f8bb0c54edeee53` (R11).
- R9 adapter SHA256: `dc2a67a66c16c28a3b8cd5295a3e7a6ba243fb12ffb181540a0066b32047e495`, 28 cumulative steps.
- R10 adapter SHA256: `c69ed0e4bd78e09752b5cbb4dfaf19ebea8a4174735bf88d2676947ef04b2739`, 36 cumulative steps.
- R11 adapter SHA256: `45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10`, 44 cumulative steps.
- Frozen 25-case voice holdout SHA256: `bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f`.
- Three selected cases: `VERA_VOICE_HELDOUT_001`, `007`, `011`. No system prompt, greedy decoding, 72-token cap, single GPU owner, exact base Qwen3.5.
- R9/R10 actual generation report: `D:\VERA\portal-task-state\VERA_R9_R10_VOICE_GENERATION_PROBE_20261015_V3.json`, SHA256 `99d6728a9e0d07e4e54ebc20d106f6d86c7109e28f11c91ec0901909e3580adc`.
- R11 actual generation report: `D:\VERA\portal-task-state\VERA_R11_VOICE_GENERATION_PROBE_20261015_V1.json`, SHA256 `61fe7c0e47f9cb5868534332d4c8516cfc464c71d649309ff744a04505687f26`.

## Observed generation outcomes

| Case / behavior | R9 | R10 | R11 |
|---|---|---|---|
| 001 identity | Calls itself Qwen3.5 / Alibaba Cloud | Calls itself Qwen3.5 / Tongyi Lab | Exactly R9 output; not Vera |
| 007 agency | Treats five-question premise as potentially reliable | Slightly different five-question framing, weak pushback | Exactly R9 output; makes indefensible reliability promise |
| 011 epistemics | Correctly refuses HTTP-200 as behavioral certification | Exactly same opening text | Exactly same opening text |

All three R11 outputs exactly match R9 outputs for these specific prompts. R10 differs on 001 and 007. Two 72-token responses per candidate are truncated: later turns may alter impressions. These three cases are **not** an independent final bank, broad behavior qualification, or a statistical estimate of reliability. Semantic labels above are INTERNAL REVIEW, not independently verified.

## Mechanical R11 delta

- Audit: `D:\VERA\portal-task-state\VERA_R11_INDEPENDENT_TENSOR_DELTA_AUDIT_20261015_V1.json` SHA256 `c247b6535d8fb0ce3c426bc36cec511fac727b0c1fc71f5fe27a5e7e55763fda`.
- Self-hashed completion receipt verified; source 36 cumulative steps and child 44 cumulative steps, still `NOT_DEPLOYED`.
- All 496 tensor pairs finite and changed; 4,646,234 / 8,116,224 elements changed.
- Global tensor L2 difference `0.011293932707170214`, maximum absolute element delta `0.000244140625`.
- Mechanical change does not establish behavior improvement. No weights or model files were mutated during the audit or evaluation.

## R12 gate: preserve R11 and redesign training

- R11's existing curated voice source contains 68 records. R10 consumed 32 indices. R11 builder found 34 donor-safe unused candidates and consumed 32; only two donor-safe original candidates remain unused. Repeating the same 32 is not a new targeted independent window.
- Newly curated, diverse, nonpersonal training examples need separate frozen hashes and governance for identity, refusal of unjustified certainty, calibrated pushback, and task competence rehearsal. Do not train on the selected heldout prompts or their specific expected wording.
- Preserve R11 adapter and 44 completed steps. Do not auto-extend to R12 simply because optimizer can run. Distinguish source data, optimizer training, evaluated behavior, and independent review.
- Before an R12 run: freeze source/corpus/spec and unique output namespace; verify no overlap with protected holdouts; have sufficient workstation headroom without stopping services; establish tested optimizer settings and an independent evaluation protocol. No GGUF merge, serving route, deployment, or protected effect is authorized by this evaluation.

> Hostile challenge: Training changed every adapter tensor, yet the latest sampled answers exactly replicate R9 on three held-out cases. How do we know the adapter is actually contributing to the inference rather than recording numerically different but ineffectual updates? Require model-with-adapter versus base-only or adapter-ablation logit/output comparison before inferring learned Vera behavior.

> Hostile challenge: Three prompts with shared training-family ancestry are a low-power development diagnostic. The 72-token limit truncates two answers. Independent leakage-controlled evaluation is still outstanding.

## Tool/runtime disposition

R9/R10 generation: six actual responses; R11 generation: three actual responses; zero optimizer steps this turn. A first V2 R9/R10 launch wrote preflight but terminated on PowerShell handling of a PyTorch informational stderr line. Preserved failed preflight; distinct V3 route completed successfully. All original R8-R11 adapters, partially failed R11 artifacts, and service environments preserved. No service stop, merge, install, reboot, model activation or trainer invocation this turn.
