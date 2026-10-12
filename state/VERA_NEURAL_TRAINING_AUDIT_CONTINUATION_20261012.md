# Vera neural-model training — source receipt / continuation — 2026-10-12

Status: PARTIAL SOURCE AUDIT; REAL DEVELOPMENT WEIGHT TRAINING IS SOURCE-EVIDENCED; NO FULL VERA SUCCESSOR QUALIFICATION ESTABLISHED. This is an evidence index, not current runtime proof.

## Source subjects
- Training repository: thebrazenbeard/vera_model_training; observed main SHA 73335b40868e56c4b01d97dd797514e49d7990a3.
- Vera Mono repository: thebrazenbeard/vera-mono; observed main SHA dd6fd951f2a935373d6d9e3a5958e77920626ba5.
- Active training Draft PR #92: work/vera-model-portal-veraos-training-target-20261012 at ba062a6dcfa515561b9df824b85c41dcda3877ff (read 2026-10-12); no merge.
- This isolated branch: work/vera-evidence-classifier-20261012; based on training main above. PR #92 paths were not modified.

## Direct source observations
- main README states the public zero-cost bootcamp produces an auditable identity capsule, not neural-weight updates; local student is a proxy.
- PROTOCOL_V2_CURRENT.md permits bounded isolated reversible source work but preserves privacy, cost, and qualification boundaries.
- PR #92's candidate verifier checks declared base/dataset/recipe digests and actual candidate artifact bytes when given local files; its return explicitly says ARTIFACT_BYTES_VERIFIED_NOT_MODEL_QUALIFIED. Its draft description reports no training, inference, or model activation.
- Deeper PR/history inspection found merged weight-changing development work. PR #81 records real local QLoRA development training on Qwen3.5 with nonpaged bitsandbytes AdamW 8-bit: 1-step, 4-step, and continued 12-cumulative-step adapters. The 12-step adapter_model SHA-256 is 15df1dcd9f5b5a4db35a8e2fdde20771c3af305b4269480af67e57c1f42fe2c3; receipt records weight_digest_changed=true and 64 continuation rows.\n- The committed PR #81 held-out diagnostic compares base, step4, and step12 on 16 SHA256-ranked samples from a 2,500-row validation set. Step12 beats base on all 16 sampled cases by loss and lowers token-weighted completion NLL from 1.709006929567075 to 1.6660596581431926. The receipt itself correctly limits this to a development diagnostic, not final/external qualification.\n- Merged PR #57 records a separate 96-row H07 V2 SFT adapter and a negative identity result: uncued Vera identity FAIL; the model identified as Qwen3.5 rather than Vera. This is trained-weight evidence and also evidence against treating that adapter as a full Vera successor.\n- The audit still has not independently re-hashed the local adapter bytes from D:\\VERA, proven current artifact custody, completed all branches' content review, or completed Vera Mono interface/bootstrap/effect/manifest inspection. Those remain UNKNOWN, not absent.
- Lappy RDC device discovery succeeded, but execution failed after the device became unavailable. WorkBridge endpoint also returned unavailable. No local runtime readback was obtained.
- Lantern currentness was not consulted; no WoWSQL readback occurred. Do not infer Lantern state from this file.

## Bounded change
- training/classify_model_evidence.py: offline schema/ordering/declared-hash classifier. Classification states are CONFIGURED_ONLY, WEIGHT_UPDATE_REPORTED, ARTIFACT_REPORTED, EVALUATION_REPORTED. All are unverified; hashes_verified, training_executed_verified, and model_qualified always false.
- tests/test_classify_model_evidence.py: five cases covering levels, missing digest, false independence, skipped stage, string booleans.
- Locally constructed matching source tested with Python unittest: 5/5 PASS. GitHub-committed byte-for-byte verification and full repository regression remain NOT RUN.

## Technical blockers
1. No inspected independent evidence of a weight-changing run with exact base revision, legally admissible dataset inventory, deterministic preprocessing, training code/environment, seed, hardware, logs and resulting weight digest.
2. No inspected preregistered, leakage-controlled held-out evaluation against the same-base untuned control, including safety, false-premise, authority spoofing, long-context, and regression cases.
3. No independently witnessed KoboldCPP inference bound to exact exported GGUF and no model qualification.
4. No complete source/runtime reconciliation, including Vera Mono interface/bootstrap/runtime-effect/manifest and Lappy artifact custody.
5. Public repository: no private chats, sensitive identity records, secrets or protected data may be introduced.

## Hostile challenge
> A syntactically valid SHA-256 string is not evidence the file exists or was hashed. A training log and adapter file can also be fabricated or unrelated to the declared base/dataset. Even a real weight delta may not improve behavior over the base model.

Accepted: classifier never verifies or qualifies. Next work must read exact bytes, bind reproducible lineage and run independent paired inference. No post-hoc thresholds.

## Next source-only unit
Complete read-only tree/history/branch/PR/artifact inventory and Vera Mono source contract read; reconcile with live D:\VERA state when remote connectivity returns. Then implement a byte-checking lineage verifier with negative tests for substituted base, dataset, artifact and contaminated eval splits. Do not collide with PR #92's handoff verifier.

## Authority
No paid training, credential changes, protected data publication, deployment, cutover, canonical memory writes, merge, P.O.R.T.A.L. runtime/connector/PR #22/Firesafe modifications. Patrick must explicitly authorize any such effect.
