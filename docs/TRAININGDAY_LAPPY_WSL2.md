# TrainingDay: local Lappy WSL2 CUDA execution route

**Execution target:** Lappy's existing Ubuntu 26.04 WSL2 distribution, using
Lappy's physically attached NVIDIA GeForce RTX 3050 Laptop GPU through the
Windows NVIDIA driver/WSL2 CUDA interface. There is no remote compute host,
network GPU proxy, or Windows training-process hop in this route.

This neural-weight training route is **separate from the zero-cost public
identity bootcamp** described in the root README. The bootcamp does not write
model weights. Nothing in this page makes private training data publishable or
issues training authority.

## Physically observed baseline (2026-11-02)

Read-only Lappy observation established:

- `wsl.exe --list --verbose`: `Ubuntu-26.04` installed as WSL2;
- WSL-specific `/usr/lib/wsl/lib/nvidia-smi` succeeded and identified the RTX
  3050 Laptop GPU, **4096 MiB VRAM**, NVIDIA driver **617.42**;
- WSL system Python: **3.14.4**, no verified `pip` or PyTorch training stack;
- WSL available RAM at probe: about **4.9 GiB** (variable with workload).

These are **device observations**, not a measured 4 GiB LoRA training fit.
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

Before attempting Qwen3.5-4B Vera LoRA training, the separately controlled
neural runner must be source-integrated, and a compatible Python environment,
GPU-enabled PyTorch/quantization/PEFT stack, exact weight/tokenizer provenance,
local RAM+VRAM fit, protected evaluator custody, data-rights grant, and real
host GPU authorization must be verified. Do not weaken those checks to obtain
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
