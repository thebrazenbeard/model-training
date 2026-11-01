# Qwen3.5-4B BF16 / DeepSpeed ZeRO-3 CPU-offload source candidate

Status: **SOURCE-ONLY / PHYSICAL HARD HOLD / NOT TRAINING-QUALIFIED**.

The prospective `lowmem_zero3_engine.py` is separate from R18's frozen
NF4/QLoRA benchmark. The first statement of its physical entrypoint calls the
existing unconditional host-authority hard hold. Importing this module or
running its source-only CLI never reads base weights, starts DeepSpeed/CUDA,
loads protected evaluation data, or creates a model checkpoint.

The candidate implements a real *shape* of a training-capable backend: it
creates a `HfDeepSpeedConfig` before the official pinned Qwen3.5 text-only
`from_pretrained` load; installs rank-2 LoRA; initializes DeepSpeed ZeRO-3
with CPU parameter and optimizer offload; then uses
`engine.backward(loss)` and `engine.step()` for three synthetic microbatch-1,
64-token updates. It gathers ZeRO-owned adapter bytes and gradients, checks
their continuity, and attempts GPU allocator/CPU RSS telemetry. No inference
`device_map="auto"` is involved.

This is **not** proof the APIs interoperate on the pinned Qwen3.5-4B
multimodal-to-text checkpoint. The exact DeepSpeed, PEFT, Transformers and
Gated DeltaNet backward-kernel versions are unverified. `HfDeepSpeedConfig`
loading and ZeRO-gathering may fail at runtime; the source contract tests do
not conceal that. The proposed 16 GiB free-RAM floor is a conservative policy,
not a measured fit guarantee, and is above the historical Lappy availability.
With one GPU there is no multi-GPU sharding benefit; CPU offload incurs
substantial RAM and transfer pressure. Three steps cannot prove an asymptotic
memory plateau or useful generalization. There is no save/reload, external
`nvidia-smi` peak witness, protected evaluator read, or accepted training corpus.

Before a real execution, an independently installed host broker must bind
actual process identity, immutable run plan, owner-reviewed data rights,
protected evaluator custody, authenticated NAS route, exclusive GPU lease and
revocation through the *actual* worker process. A Windows Job containing
`wsl.exe` alone does not prove revocation of Linux GPU workers. Passing a
Python token or changing this source cannot replace the broker.

Source-only focused test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests \
  python -m unittest -q tests.test_lowmem_zero3_engine
```

Official mechanism documentation:

- https://huggingface.co/docs/transformers/en/main_classes/deepspeed
- https://huggingface.co/docs/peft/accelerate/deepspeed
- https://www.deepspeed.ai/docs/config-json/
- https://docs.pytorch.org/docs/stable/notes/cuda.html

In particular, Hugging Face warns that ZeRO-3 plus QLoRA with CPU offload
can increase GPU memory usage; that is a reason this route proposes BF16 LoRA
instead of treating QLoRA as a memory-fit guarantee.
