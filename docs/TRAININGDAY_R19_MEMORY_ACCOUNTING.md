# TrainingDay R19: memory and equal-time accounting

Status: SOURCE/CPU-TEST contract; not a GPU qualification or a training result.

The R18 source previously seeded QLoRA's final `peak_cuda_*` receipt fields
with the largest synthetic fit-probe peak, then took `max(probe, train)` after
primary training. The MeZO receipt instead reset CUDA peak counters before its
optimizer window. The original comparison therefore mixed incompatible memory
scopes, even if both models used the same primary sequence length.

## Memory scopes

- `peak_cuda_allocated_bytes` and `peak_cuda_reserved_bytes`: **primary training
  process**, captured after `torch.cuda.reset_peak_memory_stats()` following
  model/setup/diagnostic work. These are PyTorch allocator metrics, not a proof
  of physical VRAM residency. `peak_cuda_metric_scope` is mandatory for a
  qualified R18 comparison.
- `fit_probe_peak_cuda_allocated_bytes` and `fit_probe_peak_cuda_reserved_bytes`:
  maximum observed values among synthetic, isolated fit-probe workers. They
  are separate from the primary training counters. The source reports max
  probes regardless of FIT/OOM status, so this is a diagnostic frontier peak,
  not a successful-shape memory claim.
- `peak_nvml_used_bytes`: where actual NVML sampling exists, a sampled process-
  adjacent device-used observation rather than a continuous peak or a model
  residency attestation; QLoRA reports `UNAVAILABLE_NOT_ESTIMATED`, not an
  invented value. WDDM residency remains unverified without a separate physical
  probe.
- `peak_cpu_rss_bytes`: MeZO samples process RSS during training; QLoRA's
  helper reports Windows process-lifetime peak working set, or current RSS on
  platforms without that field. Those figures are explicitly **not** symmetric
  optimizer-window CPU peaks and should not be compared as such.

## Equal-time comparability

The comparison retains each arm's strict optimizer-window monotonic start,
stop and deadline, actual full optimizer steps, unique examples and total
presentations, useful forward/backward tokens, padded forward/backward tokens,
supervised target tokens, and external process stage times. Fit-probe wall time
and memory do not count as training-time optimizer work. No interpretation is
valid if a step exceeds the deadline, if host standby invalidates timing, or if
MeZO and QLoRA optimize different loss weighting. The existing validator
requires both correct/corrupt arms to consume at least 85% of the preregistered
budget, cross-method correct arms to be within 5%, and all eight controls.

These are necessary source gates, not proof of behavioral improvement or equal
physical compute. The R18 531-second MeZO receipt (19,813.562 seconds observed)
remains `TIME_BUDGET_OVERRUN_INVALID`; it cannot be repaired by subtraction of
Modern Standby elapsed time after the fact. No QLoRA comparison or sealed
held-out score is qualified by this patch.
