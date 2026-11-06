# Neuro-Sync → TRAININGDAY governance (SOURCE-ONLY, read-only)

Reviewed `neuro-sync` main `89fdef1dd83536cf29875d3de7c2acb07e6f2cfa` and `model-training` main `2b83430dd36076c5f65624709968e372b170ea36` on November 6, 2026.

Proposed eventual paths:

- `model-training/trainingday/neurosync_statebus_gate.py`
- `model-training/tests/test_trainingday_neurosync_statebus_gate.py`

**Original flaw and correction:** `StateLedger.commit` exposes an already-committed receipt without rerunning authorization, but also performs writes for new events. A gate that detects the wrong receipt *after* invoking `commit()` may deny a response after metadata has already changed. The revised gate has **no commit operation**. It only calls the exact reviewed Neuro-Sync `StateLedger.events` read-only method and scans at most 2,000 events, failing closed if the event is beyond the bound. Pre/post host authorization **and** pre/post independently verified cross-repo attestation are mandatory for any receipt recovery. The output is a narrow immutable view, never raw ledger objects.

Test command from this directory: `python -m unittest discover -s tests -p 'test_trainingday_neurosync_statebus_gate.py' -v`.

> An attacker supplying a side-effecting fake `.events()` implementation can still cause effects. Python structural type checks cannot prove purity. A trusted host must pin the real implementation and isolate its DB.

> Pre/post checks do not establish atomic policy, attestation and DB custody. A change immediately after post-check remains possible. A protected multi-tenant deployment requires independently authenticated principals, host-owned grant serialization, artifact custody and receipt witnessing.

> Unit-test lambdas returning `True` only establish code-path behavior. They are not independent verification. Source files and evidence-class labels cannot authorize physical training.

`model-training/trainingday/physical_launch_authority.py` remains an unconditional **HARD HOLD**. No GPU, checkpoint, launch, optimizer, grant creation, or host effects are supported.

No remote writes, repository integration, service operations or actual physical training were performed. Keep this out of a production PR until the host authentication/custody boundary is established.
