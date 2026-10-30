# R21 strict MeZO budget-integrity guard (source-only)

The strict R18/R21 equal-training-time contract does not permit cross-process
resume or periodic optimizer checkpoints (the R19 fail-closed source change).

Before each complete paired MeZO update, the runner already reserves the
largest relevant measured step time times 1.5, plus two seconds. R21 now also
compares each **completed** step's actual monotonic duration against that
admitted reserve. A step longer than the reserve yields
`TIME_BUDGET_INTEGRITY_INVALID` even if the absolute training deadline has not
yet been reached. The more specific `TIME_BUDGET_OVERRUN_INVALID` takes
precedence whenever the deadline was exceeded.

The runner also captures a wall-clock duration around each step. A backward
clock or a wall/monotonic difference exceeding the greater of two seconds or
five percent of the monotonic duration invalidates clock custody. Such a
disagreement can reflect a clock adjustment and **does not prove** Windows
Modern Standby. Similarly, a large step could be caused by throttling or
device contention rather than a power transition. Both invalidate a strict
comparison without making a false causal attribution.

The receipt retains `completed_step_durations_seconds`, adds
`completed_step_wall_durations_seconds`, `step_timing_integrity_failures`, and
`budget_integrity_passed`. `deadline_respected` still means only that the
numeric monotonic deadline was not exceeded; it is not an admission decision.

This guard is **not** a hard watchdog or a power-state lease. It executes only
after a blocked optimizer step returns, cannot preempt a sleeping or hung
process, and cannot independently attest Windows power events. A physical
host supervisor and independently recorded power/lease history are required
before claiming Modern Standby prevention or a valid fresh comparator run.

The historical R18 531-second budget/19,813.562-second elapsed artifact remains
`TIME_BUDGET_OVERRUN_INVALID` and cannot be retrospectively rehabilitated.
