# R13 retry2: surviving FP32 adapter vs failed historical completion (2026-11-14)

This is an **independent saved-file forensic comparison**, NOT a recreated `DEV_TRAINING_COMPLETE.json`, attestation of historical optimizer execution, model qualification, deployment, or LoRA/GGUF pairing approval.

## Historical source and controller

- Immutable training source: `model-training` Draft PR #102 head `b8c352609dc2ef0969746bc18afbe10e183256a5`; frozen retry2 spec SHA-256 `0127d7e88f3ed608117d2a1388d0c82a3d921df75fbf408374db3945b1060418`.
- Frozen R13 train corpus SHA-256 `90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5`, requested 32 optimizer steps after cumulative R12 52 steps. These are source/controller claims; this audit does not independently hash the corpus.
- R12 verified completion receipt `D:\VERA\models\adapters\v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2\DEV_TRAINING_COMPLETE.json` raw SHA `111d0813cc93acb2adff1a04703d9cb3cccdc6bc01742c409e8f55c52673d2ae`, canonical receipt self SHA `5d7c636f361a1ea9871f6a523270d989233dd2276a911ab290c22b0484b34bc9`.
- Historical R13 V4 controller raw SHA `07a659c2985e2021a8e5e515af24282a2e613eab7f031b2772221372dfbf3554`: **R13_FAILED_OR_INTERRUPTED_NOT_VERIFIED** and “Adapter/receipt missing despite trainer exit 0”; it restored the earlier serving tasks, not a model deployment.
- Historical V4 trainer stdout SHA `aab9ee3c7f23abb6c1aab0a59f17dc8d4b772768bbcf7d26de01df0d1602d3be` has 32 loss rows, then `{"reason":"effect label step count is unsupported","schema":"V10R2_QWEN35_DEV_TRAINING_RECEIPT_V1","status":"HOLD"}`.
- The defect that prevented a 32-step receipt is **already separately fixed** in open, unmerged Draft [model-training #130](https://github.com/thebrazenbeard/model-training/pull/130), HEAD `c8455360eb3ce3daa60362f0a4422bb55c9305da`. This forensic change stacks on #130; it does not duplicate that fix or retroactively reconstruct a completion receipt.

## Actual full-file checks and observed differences

- R12 parent LoRA file SHA-256 `a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c`.
- Surviving R13 child LoRA file SHA-256 `51300ff3b6d2bf60d144c6ae4c0fa1a6a153ce94e86c141a93c4cfb5e77e8347`.
- Real bounded safetensors read: 496/496 matching named FP32 tensor keys/shapes, 8,116,224 FP32 scalar positions compared, 496 changed tensors, **8,116,219 changed scalar values**; maximum absolute delta `0.004037826554849744`. Header, dtype, finite data, contiguous extents, R12 receipt self-hash and claimed parent, R13 spec and failed controller, and 32 log rows are checked.
- Real evidence JSON is stored **off Git**: `D:\VERA\experiments\mini-vera-r13-unreceipted-forensics-20261114\real_savetensors_delta.json`, SHA-256 `348b7fb1104bef5300ff1f75bed721db086e83588af32a5b513b463aac70032f`; intentional CLI exit **2**.
- Independently streamed current HF base full-shard SHA-256 checks on Lappy (both files, 9,319,828,160 bytes) match pinned `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive@d61dd146c8fd44c9a49cdb7f59f34e17b61902d8` manifest. The R12 October 15 receipt records those same hashes. **These checks do not independently attest R13's October 16 exact loaded base shards.**
- The downloaded `lmstudio-community/Qwen3.5-4B-GGUF` candidate is a completely different published GGUF file whose publisher names official `Qwen/Qwen3.5-4B` upstream. An adapter/GGUF neural A/B is still **NO-GO** without identical proven base weights and converter custody.

## Safe usage

Run `python -B -m successor.experiments.r13_unreceipted_forensics` with **all six required paths**: `--r12-adapter`, `--r13-adapter`, `--r12-receipt`, `--r13-spec`, `--controller`, and `--trainer-stdout`. The command only reads those files, does not import torch or load any model or GPU, and returns **2** even when numerical deltas are observed. It refuses malformed safetensors, nonfinite/non-FP32 tensor data, missing/invalid R12 receipt binding, controller disagreement, a present R13 final receipt, or wrong terminal log.

Tests: `py -3.14 -m unittest tests.test_r13_unreceipted_forensics` and optimized Python both pass; see exact PR verification notes for the larger scope. `model-training` PR #103's **strict** auditor still requires the absent final receipt and is not a substitute.

**Evidence ceiling:** Local saved tensor differences are necessary evidence of a changed adapter, not independently verified causal optimizer updates, training success, improved speech/reasoning, durable identity, approved memory, private tool authority, or compatible GGUF ancestry. The original R13 controller remains the authoritative historical failure record. No original adapter, receipt, or source history was modified.
