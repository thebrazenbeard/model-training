# R20 preregistered comparison readiness check (source only)

`python -m trainingday.r20_comparator_gate --plan PLAN.json --evidence DECLARATIONS.json --now-unix UNIX_SECONDS`

This read-only CLI checks a declared R18 preregistered plan and corresponding R20 evidence shape. It emits machine-readable `HOLD_MISSING_PRECOMPARISON_EVIDENCE` (exit 2) and one or more concrete blocker codes for missing or contradictory evidence. A synthetically complete declaration emits `EVIDENCE_SHAPES_COMPLETE_OWNER_VERIFICATION_REQUIRED` (exit **3**), **never** GPU launch permission. It explicitly reports `decision_scope=DECLARED_EVIDENCE_CONSISTENCY_ONLY` and `gpu_launch_decision=NOT_AUTHORIZED`; `authorizes_gpu_start` is always false. The CLI deliberately has no success-shaped exit 0 for either case.

The evidence declaration must bind:

1. Exact plan digest and one pinned source head with passed CPU objective-parity and fail-closed checkpoint source tests. Both arms must declare the same supervised-target-token-weighted mean loss; unsupported periodic resume is not silently represented as complete.
2. Exact admitted corpus, selected train IDs and content hashes; `eval_access_before_arm_freeze` must be false.
3. Same initial LoRA adapter bytes, trainable tensor digests and base-model bindings for both arms.
4. Fresh exclusive Lappy/CUDA0 lease evidence, observed within 60 seconds and expiring within the next ten minutes; fresh awake/lid-open and actual suspend guard declarations, with no standby since preregistration.
5. Distinct pinned MeZO and QLoRA fit-probe receipts, equal primary-training and individual probe budgets, an overrun/suspend fail-closed policy, both arms and all four frozen/zero/correct/corrupt controls.

The module intentionally imports no CUDA, model, tokenizers or evaluation plaintext. It does not inspect or modify host policies, launch GPU processes, expose private bank labels, train weights or promote any model. **Declared booleans, timestamp values, hashes, and lease IDs are not authentication.** Operators must independently validate live lease ownership, host power state, physical optimizer receipts, source/test heads and actual adapter hashes before invoking existing runners. Existing runners are *not wired to require this optional gate*; that integration remains a separate safety change.

Historical R18 MeZO: the 531-second intended budget was exceeded by a 19,813.562-second reported duration, tightly correlated with Windows Modern Standby. Preserve its adapter/custody as an invalid equal-time arm, not a comparator baseline. R19 source repairs and R20 fixture migrations have been combined on a **local, unmerged** candidate, with source/CPU checks only. Private-evaluation scoring is separately held until independent pre-unseal attestation exists and is qualified. Hence the current physically justified result is HOLD; do not fabricate a positive declaration.
