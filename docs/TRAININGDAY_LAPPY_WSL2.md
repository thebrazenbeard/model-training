# TrainingDay: local Lappy WSL2 CUDA execution route

**Execution target:** Lappy's existing Ubuntu 26.04 WSL2 distribution, using
Lappy's physically attached NVIDIA GeForce RTX 3050 Laptop GPU through the
Windows NVIDIA driver/WSL2 CUDA interface. There is no remote compute host,
network GPU proxy, or Windows training-process hop in this route.

This neural-weight training route is **separate from the zero-cost public
identity bootcamp** described in the root README. The bootcamp does not write
model weights. Nothing in this page makes private training data publishable or
issues training authority.

## Historical initial baseline (2026-11-02, before WSL environment setup)

Read-only Lappy observation established:

- `wsl.exe --list --verbose`: `Ubuntu-26.04` installed as WSL2;
- WSL-specific `/usr/lib/wsl/lib/nvidia-smi` succeeded and identified the RTX
  3050 Laptop GPU, **4096 MiB VRAM**, NVIDIA driver **617.42**;
- WSL system Python: **3.14.4**, no verified `pip` or PyTorch training stack;
- WSL available RAM at probe: about **4.9 GiB** (variable with workload).

These were the **initial device observations**, not the final state of Lappy
later that day and not a measured 4 GiB Qwen LoRA training fit. See
[the subsequent WSL CUDA runtime observations](TRAININGDAY_WSL2_RUNTIME_EVIDENCE_20261102.md)
before treating the initial Python/RAM figures as current.
WSL2 GPU access still uses the Windows host driver. A visible CUDA device does
not imply physical GPU exclusivity or grant the source-data rights required by
the frozen TrainingDay gates.
Keep the NVIDIA driver on the Windows host; **do not install a separate Linux
NVIDIA display driver inside WSL2**. See NVIDIA's
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

## Reproduce the local preflight

From Windows PowerShell, without installing anything:

```powershell
wsl.exe --list --verbose
wsl.exe --distribution Ubuntu-26.04 --exec /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

From **inside Lappy's Ubuntu WSL2 shell**, in a checkout of this repository:

```bash
python3 -m trainingday.lappy_wsl2_preflight
```

The preflight uses only Python's standard library. It checks that the running
kernel is WSL2, invokes exactly `/usr/lib/wsl/lib/nvidia-smi` with a five-second
timeout, and emits one JSON record. A zero process exit means **NVIDIA GPU
telemetry only**. It does **not** prove that PyTorch CUDA can initialize,
allocate tensors, or perform backward/optimizer steps.
Both success and failure records hardcode
`data_rights_granted=false`, `gpu_lease_granted=false`, and
`training_authorized=false`; this preflight never starts training.

Verify this source on any Python 3.8+ host without requiring NVIDIA hardware:

```bash
python3 -m unittest -v tests.test_lappy_wsl2_preflight
```

## Neural-training work still required

Before attempting new qualified Qwen3.5-4B Vera LoRA training, the separately
controlled neural runner must be integrated and admitted for that job. An
isolated Python/CUDA/quantization
stack has now passed **tiny synthetic optimizer** probes on Lappy
([measured evidence](TRAININGDAY_WSL2_RUNTIME_EVIDENCE_20261102.md)).
That does not establish full Qwen3.5-4B loading, 4 GiB GPU fit,
training-capable offload, exact weight/tokenizer provenance, protected
evaluator custody, a data-rights grant, or real host GPU authorization.
Do not weaken those checks to obtain
a superficial "started" result.

The intended artifact flow, once authorized and genuinely run, is:

1. Load the verified trainable base and tokenizer in the **same WSL2 process**
   using a measured low-memory configuration.
2. Train and persist a versioned LoRA adapter with checkpoints and receipts.
3. Independently evaluate held-out tasks and compare against controls.
4. For inference through KoboldCPP, export/quantize an appropriate GGUF after
   a separately validated merge/conversion workflow; a GGUF runtime file is
   not itself a conventional PEFT training checkpoint.

None of these four neural-weight steps is implemented by the preflight module.
Separately, [R16 Draft PR #140](https://github.com/thebrazenbeard/model-training/pull/140)
records earlier **physical Qwen3.5 NF4 correct/corrupt eight-step LoRA runs**
with adapter hashes and checkpoint custody. Those PR receipts are a different
evidence class from this preflight and from the subsequent **tiny synthetic**
WSL probes; they do not establish authorized promotion or generalization.

## Experimental neural source, still held

The separate `trainingday.lowmem_experiment` source candidate contains a
reviewable Qwen3.5-4B NF4/LoRA rank-4, sequence-64 optimizer-step kernel and
an unrunnable ZeRO-3 CPU-offload *hypothesis*. Its command-line form prints
only a hard-HOLD plan:

```bash
python3 -m trainingday.lowmem_experiment
```

This code cannot launch a training job: the physical authority check raises
before any model, CUDA, data, or output effects. The source tree now includes
`trainingday.pinned_base` and `trainingday.qwen_text_checkpoint` as separately
testable, fail-closed dependencies for local offline byte-hash checks and
multimodal-to-text tensor-name projection. These are **source-only helpers**:
the neural R18 trainer, protected host broker, integration with the isolated
Transformers/PEFT/CUDA environment, and actual full-checkpoint loading or
optimizer fit remain unqualified. Small CUDA/LoRA kernels were separately
measured in the [dated runtime evidence](TRAININGDAY_WSL2_RUNTIME_EVIDENCE_20261102.md).

**Do not interchange the two upstream model identities.** The new
`docs/artifacts/qwen35-4b-public-base-20261102.json` describes a cached public
`rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` checkpoint at revision
`d61dd146c8fd44c9a49cdb7f59f34e17b61902d8`. This experimental R18
source specifically binds official `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The public cached manifest
does **not** meet the R18 binding; a separate official snapshot, byte hashes,
tokenizer, validated license/provenance, and signed authority are still needed.
WSL2 NVIDIA **visibility alone** does not lift the guard or prove PyTorch CUDA
support. Subsequent tiny PyTorch probes did confirm CUDA operations, but still
do not establish Qwen3.5-4B fit in 4096 MiB of VRAM.
