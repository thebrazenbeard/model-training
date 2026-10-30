# TrainingDay R25: bounded budget / lease shadow assessment

Status: **source and synthetic CPU tests only**. This is not an operational
lease controller, power-state monitor, or GPU launcher.

The r25_budget_lease_policy.assess_budget_run() function consumes a bounded
tuple of fabricated/caller-declared observations and one logical lease claim.
It rejects malformed values and flags:

The lease claim includes an explicit chronological lease_renewals tuple.
For source-only intervals longer than the hard 30-second lease freshness
limit, renewal declarations must preserve the exact device/resource/owner/
lease ID, arrive before the previous expiry, have no gap greater than the
bounded source policy, and remain fresh through the final observation.
Synthetic 31-, 60- and 531-second windows are exercised with this chain.
No renewal is independently signed or enforced by a host, and an unexpired
lease cannot be silently reassigned to a different owner.

The caller must explicitly declare run_mode=TRAINING or
run_mode=ZERO_UPDATE_CONTROL. A completed TRAINING declaration with no
positive optimizer-active-time progress is rejected; ZERO_UPDATE_CONTROL
requires *no* optimizer-active-time advancement. Neither condition is proof
of weight updates or a correctly implemented zero-control experiment.

Safety-sensitive duration policies have source-enforced maxima independent
of caller-supplied limits: heartbeat freshness 10 seconds, sample gap
30 seconds, clock discrepancy 5 seconds, logical lease age 30 seconds,
and logical lease remaining horizon 120 seconds. A caller declaring longer
durations receives MALFORMED_POLICY; even a stricter policy is still only
a self-declared shadow check, not an authenticated host-enforced resource lease.

| Area | Rejection signals |
| --- | --- |
| Owner/resource | wrong Lappy/CUDA0 identity, stale or expired lease, conflicting active IDs, excessive logical lease horizon |
| Power/transport | any reported Modern Standby, lost connection, missing/stale supervisor heartbeat |
| Clock | time reversal, implausible active-step accumulation, unwitnessed sample gap, wall/monotonic divergence |
| Resources | thermal cap, GPU peak-byte cap, insufficient observed free bytes plus reserve |
| Budgets | active optimizer seconds, unadjusted wall seconds, watt-hours, USD cost, incomplete run |
| Integrity | nonfinite or malformed measurements, invalid policy or timestamps |

The **wall-time cap is never reduced by a reported sleep interval**. Original
R18's 531-second admitted budget versus 19,813.562-second elapsed receipt
remains TIME_BUDGET_OVERRUN_INVALID despite the strongly aligned
19,698-second Modern Standby interval.

Clock consistency checks both the divergence of each adjacent observation
and cumulative divergence of every observation from the first. A series of
small individually tolerable deviations cannot silently accumulate beyond
the declared threshold.

Even when synthetic_constraints_satisfied=true, these outputs are fixed:

- status=HOLD
- authorizes_gpu_start=false
- host_lease_authenticated=false
- valid_equal_time_neural_evidence=false
- authorizes_model_promotion=false

No module import hooks this policy into the production R18 trainer. Recorded
lease IDs, host names, connection booleans, timestamps, costs, GPU readings
and heartbeats are **untrusted declarations**, not host-signed evidence.
Sparse sampling can also miss sleep or thermal events between observations.
The source can demonstrate invalidity from an observed failure; it cannot
certify uninterrupted custody from an absence of such a failure.

## Future physical acceptance gate (not executed here)

1. Independently authenticate host identity and atomic exclusive GPU lease,
   including concurrent ChatGPT coordinators and Executor workload ownership.
2. Record Windows Kernel-Power 506/507 through an independent witness across
   the entire optimizer window. Missing power-event coverage is a HOLD.
3. Obtain signed supervisor heartbeats external to the training process and
   reject disconnects, staleness, discontinuities and unobserved intervals.
4. Collect real GPU memory and thermal counters with metric sources/units,
   plus actual energy and monetary costs if those policies are claimed.
5. Require an exact source/data/adapter/optimizer identity and matching
   MeZO/QLoRA budget, corpus, sample exposure and tokenizer-ID parity.
6. Verify frozen/zero/correct/corrupt controls and fresh-process adapter
   reload independently before any behavioral or release claim.

Source tests use no local model, host, GPU, private bank or credentials.
Test command: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest -q tests.test_r25_budget_lease_policy

Evidence class: TEST of synthetic policy behavior. No real-system probe,
neural tensor delta, formal verification, independent host attestation, or
model release is implied.
