# Qwen3.5 low-memory device/host telemetry — source-only candidate

**Scope:** `trainingday.lowmem_budget_telemetry` is a pure import, with a
testable capture helper intended for the separately protected NF4 LoRA kernel.
The physical kernel still exits at `require_physical_launch_authority()`
before importing PyTorch, reading model bytes, or initializing CUDA. This
change does **not** authorize a GPU lease or run the model.

When a real independent host broker eventually authorizes the kernel, it
captures three synchronized observations on the *same CUDA device*:

1. Before model loading;
2. After quantized model, adapter, and optimizer preparation, before the
   step-only peak reset;
3. After forward/backward and AdamW update.

Each observation records `cuda.mem_get_info()` free/total (device-wide at
sample time), PyTorch allocated/reserved and high-water counters
(process-local), `/proc/meminfo:MemAvailable` (Linux/WSL2 currently available
RAM), and `/proc/self/status:VmRSS` (this process's resident RAM). All fields
must be finite nonnegative integers with coherent bounds. Missing phases,
changed device capacity, malformed `/proc` fields, and conflicting CUDA
counters abort the witness rather than silently presenting a fit.

**These metrics have different scopes.** Do not sum PyTorch reserved bytes
with device-wide used bytes as if they were disjoint. Three samples do not
bound all transient peaks; PyTorch maxima do not measure another process's
allocations or driver activity. Device free memory observed after a step is
not a guarantee that the next step will fit. Without a host-owned exclusive
GPU lease, official weight custody, and protected evaluation admission,
even a physically collected timeline remains
`MEMORY_TELEMETRY_FIELDS_VALIDATED_FIT_UNQUALIFIED` with
`four_gib_training_fit=UNKNOWN` and `training_authorized=false`.

The source preserves two distinct strategies:

- NF4 + PEFT QLoRA on a single GPU, which can train only if an actual
  authorized complete optimizer run fits; and
- PyTorch FSDP2 `CPUOffloadPolicy`, which has shown CUDA/CPU state mechanics
  on a tiny synthetic Lappy test but is not yet fitted to official 4B weights.

Hugging Face's `device_map="auto"` is a big-model *inference* path, not an
admitted low-memory training fallback. A CPU-offload recipe must actually
perform gradients and optimizer steps with its supported distributed/offload
backend, not merely move inference layers between devices.

## Reproduce source tests without torch or GPU

```bash
python -m unittest -v \
  tests.test_trainingday_lowmem_budget_telemetry \
  tests.test_lowmem_experiment
```

These are dependency-injected counters and source inspections. They verify
shape/order/negative rejection, **not an actual CUDA capture**. The actual
peak VRAM/RAM of official Qwen3.5-4B remains unknown.

Authoritative backend documentation:

- [PyTorch FSDP2 fully_shard CPUOffloadPolicy](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [Hugging Face Transformers FSDP2](https://huggingface.co/docs/transformers/fsdp)
- [Hugging Face Big Model Inference](https://huggingface.co/docs/transformers/main/models)
