# Vera R12 local corrective experiment — source contract

R11 exact adapter: `45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10`, 44 cumulative optimizer steps, unchanged. Parent source: PR #94 head `cacd8fbc9817d454d36dfb5f78fb243f19c66fe2`. No R12 optimizer result is implied by this file.

## Failure and bounded remedy

R9, R10 and R11 actual no-system voice generations on three selected held-out prompts showed weak Vera identity and independent judgment; R11 reproduced R9 text exactly on those three prompts. R10→R11 adapter changed all 496 tensors, but R11 saved BF16 LoRA weights after 8 optimizer steps at a learning rate of `2e-6`. In a separate **synthetic CPU AdamW** test, a parameter initialized at 0.25 in BF16 did not change at `2e-6` or `2e-5`, whereas FP32 changed by about `-2e-6` and `-2e-5`. This supports a numerical-resolution explanation; it does **not** prove the actual model's LoRA updates were all lost to rounding.

R12 proposes exactly 8 additional optimizer steps at `2e-5`, resuming from the verified R11 adapter. Trainable LoRA weights are cast to FP32 **before optimizer creation**; posttraining FP32 is verified and preserved on save. Unrecognized precision policies fail closed. The quantized Qwen3.5 base, nonpaged AdamW 8-bit optimizer, training safety rules and protected-effect boundaries are unchanged.

## Data / evidence limits

- New 32-example nonpersonal, source-controlled, AI-curated Vera voice set, plus 32 unused general-competence rehearsal examples; train window rows 352–415.
- Only this 64-row window differs from the already verified 50,000-row R11 corpus.
- The frozen 25-case heldout prompts and source SHA are excluded from training; no claims of family-level independence or externally evaluated generalization.
- The new candidate must be tested against the same frozen development probes **and** subsequently a separate independent holdout. Do not use the development probes as fresh training labels.
- R12 output namespace is separate; overwrite, paid compute, merging, activation and deployment remain forbidden.

> Hostile challenge: A stronger learning rate and FP32 weights could simply overfit 32 newly authored examples. A numerically larger adapter delta would not establish reliability. Require actual paired behavior and tests for regressions, then independent evaluation.

> Hostile challenge: Holding out exact prompt strings does not establish independence from a shared behavioral family. Do not market the selected three-case panel as a final test.

> Hostile challenge: The RTX 3050 has 4 GiB total VRAM and Ollama was observed occupying about 2355 MiB on Lappy. The training gate requires at least 3900 MiB free and does not unload/stop/interrupt Ollama or any other service. A resource HOLD is an intentional safe outcome, not a completed R12 training run.
