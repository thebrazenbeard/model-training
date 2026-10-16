# Vera R13 retry2: post-training development checks, not model qualification

Exact training source is the open draft branch derived from commit `b8c352609dc2ef0969746bc18afbe10e183256a5`. The isolated evaluation source does not modify training code, adapter files, serving routes, or provider state.

Before evaluating, verify the final R13 retry2 `DEV_TRAINING_COMPLETE.json` canonical self-hash, the trained adapter file SHA256, FP32 final precision, **84 cumulative optimizer steps**, the original R12 52-step parent, frozen R13 corpus SHA256 `90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5`, retry2 spec SHA256 `0127d7e88f3ed608117d2a1388d0c82a3d921df75fbf408374db3945b1060418`, and the `exact_resume_rehydration` receipt with 496 tensors bound to R12 parent SHA256 `a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c`.

The CPU-only tensor audit compares all 496 R12 and R13 trainable LoRA tensors and rejects nonfinite values, topology drift or loss of FP32 precision. Changed values are necessary for evidence of numerical training but **not proof of desired behavior**.

Free-running deterministic development evaluation checks 10 predeclared frozen no-system prompts spanning Vera identity, independent judgment, evidence, correction and coordination, including the original three regression cases. Output is stored at exclusive, never-overwritten VERA_R13 result paths. The 25-case frozen panel was already used as development evidence and is **not** an independently protected final qualification. Positive observed changes may reflect overfitting to closely related instruction types, and any endorsement needs a separate independent fresh final bank plus general-competence regression checks.

Never load the evaluation model while R13 training uses GPU or while PreActive serves Qwen on the 4 GiB RTX 3050. The evaluator requires at least 3900 MiB free GPU, 7 GiB host RAM and 6 GiB virtual memory. Any service pause must be conducted under the separately verified reversible PreActive maintenance / recovery procedure and followed by actual endpoint readback. No production deployment, merge, installation or auto-admission is authorized.

**Status in source:** evaluator and tensor-audit code is prepared, runtime adapter validation/generation have not yet occurred. Do not infer R13 training finished or behaviors improved solely from this file.
