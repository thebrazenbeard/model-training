# Qwen3.5-4B low-memory experiment — source candidate only

Status: **HARD HOLD**. This adds no authorization, physical training, model
download, adapter checkpoint, GPU allocation, evaluator access, or bypass to
the existing R18 architecture. Run `python -m unittest tests.test_lowmem_experiment`.

## Experiment A — smaller NF4 LoRA synthetic fit

The unexecuted `one_step_kernel` proposes BF16/NF4 double quantization with
rank 4 on `q_proj`/`v_proj`, microbatch 1, sequence 64, checkpointed activations,
and text-only Qwen3.5 weights. It now has an **in-tree source-only** offline
`trainingday.pinned_base` verifier, `qwen_text_checkpoint` tensor-key mapping,
loading-shape verification, and protected physical-launch hard hold. The
byte hashes are caller-supplied evidence, **not** authenticated host authority.
It cannot be run by a
CLI flag or environment variable. Any eventual host broker must authenticate
operator, immutable plan, data rights, evaluator custody, host identity, and
exclusive GPU lease independently before dispatching a protected job.

The currently documented public cached checkpoint is the *different*
`rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` revision
`d61dd146c8fd44c9a49cdb7f59f34e17b61902d8`, while this experiment
requires official `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The verifier rejects the
public cached model identity; no model substitution is authorized. An exact
official snapshot and full SHA-256 binding have not been demonstrated here.

Its success criterion, if ever physically permitted, is a nonzero gradient,
completed `optimizer.step()`, *real changed LoRA-B tensor bytes*, and fresh CUDA
peak memory telemetry. Unit tests use **fake** tensor bytes and establish only
that the pure witness check distinguishes actual changes from declarations.

### CUDA memory evidence boundary

The source kernel resets PyTorch CUDA peak counters **before** the quantized
`from_pretrained` load. It captures allocated and reserved peaks for the
model-load/LoRA-preparation phase before the second reset, then captures a
separate optimizer-step phase. The legacy `peak_cuda_*` fields report the
maximum across both phases so a larger transient load peak cannot disappear.
Synthetic unit tests exercise the case where loading exceeds step usage.

These values cover **only the current process's PyTorch allocator**. They do
not include other processes' VRAM, driver allocations, total device use,
host RAM, or the peak of a load that fails before measurement. They are not
an independent 4 GiB fit witness; a protected host-owned process must also
capture device-level GPU and host-RAM telemetry through real repeated steps.

The candidate uses no CPU or disk offload. A 4-bit base that fits for inference
can still OOM on a backward pass. The current Lappy environment remains
unqualified for this experiment. Qwen3.5 4-bit QLoRA also has a reported
quantization-accuracy concern in the Unsloth model-family guidance: an
optimizer-step fit would not establish that the resulting adapter learns well.

## Experiment B — DeepSpeed ZeRO-3 CPU offload hypothesis

`proposed_zero3_config` defines the configuration *shape* for BF16 with CPU
parameter/optimizer offload. This is **not a runnable implementation** and has
no verified Qwen3.5 compatibility, RAM-fit result, or protected DeepSpeed
engine integration. Any implementation must use `engine.backward(loss)` and
`engine.step()`; calling a plain optimizer outside ZeRO is insufficient.
Generic `device_map="auto"` is for inference placement and is specifically
not accepted as trainable offload. On a low-free-RAM host, CPU offload can make
the situation worse rather than better.

Neither candidate establishes training quality, legal data admission, an
exclusive GPU lease, or capacity for repeated optimizer steps. They provide
isolated hypotheses that can be rejected independently without changing R18.

Official relevant documentation:

- https://huggingface.co/docs/transformers/main/model_doc/qwen3_5
- https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
- https://huggingface.co/docs/peft/developer_guides/quantization
- https://www.deepspeed.ai/tutorials/zero-offload/
