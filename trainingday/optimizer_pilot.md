# Qwen3.5-4B low-memory optimizer preflight and synthetic replay (source only)

Run: `python -m unittest -v tests.test_optimizer_pilot tests.test_lowmem_experiment`

This slice adds **no physical run path**. `physical_optimizer_launch()` always
hits the independent `physical_launch_authority` hard hold before model/GPU
effects. A source declaration, test witness, or synthetic checkpoint is not a
host broker grant. Physical pretrained optimizer steps observed: **zero**.

## Official base contract and candidate

- Model: `Qwen/Qwen3.5-4B`, immutable revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Official checkpoint architecture: `Qwen3_5ForConditionalGeneration`;
  the separately checked text projection targets `Qwen3_5ForCausalLM`
  with the official `Qwen3_5TextConfig`. `trainingday/qwen_text_checkpoint.py`
  owns complete index-to-text namespace and loaded-state validation.
- Proposed algorithm: NF4/double-quantized frozen pretrained text base, BF16
  compute, text-attention LoRA on paired `q_proj`/`v_proj`, rank 4, batch 1,
  context 64, three steps, with activation checkpointing. `one_step_kernel`
  remains the protected *unexecuted* one-step GPU candidate. This source-only
  replay deliberately uses tiny CPU float32 tensors and toy SGD, not QLoRA.
- With 4 GiB total GPU VRAM, a 512 MiB reserve leaves only 3584 MiB
  *hypothetical* headroom. This is **not** evidence that Qwen3.5 quantized
  model load or optimizer step fits; observed free VRAM, allocator load/step
  peaks, host RAM and fragmentation are unverified. The ZeRO-3 CPU-offload
  object in `lowmem_experiment.py` is a configuration hypothesis, not a
  runnable fallback.

`validate_official_plan` rejects a mismatched official source manifest,
revision, tokenization hash declaration, text loader, architecture, foreign
visual/MTP module paths, or unpaired attention targets. **These checks operate
on caller-supplied source declarations**: they cannot authenticate local file
bytes, actual module structure, frozen base, signer, data rights, or lease.
Before any future launch, the independent host gate and actual pinned snapshot
validator must verify these facts; simply passing a source validator cannot
qualify a model or start training.

## Deterministic, clearly negative replay evidence

`replay_synthetic_adapter()` uses a fixed seed, two declared LoRA-B tensor
fixtures and three CPU-only toy SGD updates. It checks finite positive
gradient norms, nonzero byte changes per step, ordered before/after SHA-256,
exclusive checkpoint creation, readback byte equality, and a distinct final
adapter SHA-256. The fixture file is created only after the memory declaration
passes; existing paths and symlinks are refused. Callers should use an
isolated temporary directory for disposable fixtures; the code never deletes
or overwrites caller files.

Receipt distinguishes `synthetic_updates = 3` from
`physical_pretrained_optimizer_steps = 0`; records source revision,
checkpoint/text architecture, *unverified declared* tokenizer SHA, seed,
synthetic provenance, rank/context, adapter hashes, save/reload and expected
resource estimate. `gpu_lease_id = null`, `rights_verified = false`,
`base_bytes_verified = false`, `tokenizer_bytes_verified = false`,
`official_graph_verified = false`, `physical_training_authorized = false`,
`model_quality_proved = false`. Never promote these records to physical
TRAININGDAY optimizer, learning, evaluation, or memory-fit evidence.

## Admission blockers

The latest handed-off authenticated broker observation is
`HOLD/HOST_COMPONENT_PRESENT_BUT_UNQUALIFIED`: no issued GPU lease, no
verified data rights, no NAS witness, zero authorized launches, and no
observed training. The current source also always denies physical launch.
No local verified official model weights/tokenizer or host-owned exclusive
lease are available in this lane. Consequently **no GPU work is attempted**,
and no claim of official pretrained optimization or generalization is made.
