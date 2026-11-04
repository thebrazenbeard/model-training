# Official Qwen3.5-4B GGUF conversion: offline capacity screen

**SOURCE_ONLY / HARD HOLD.** This adds a numeric *rejection* step before any
host-owned conversion. It does not open model files, hash weights, run the
converter, load a model or approve an exclusive process lease. No new model
file is included in Git.

The base identity is `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
not the separate `-Base` checkpoint. The conversion toolchain source pin comes
from `official_qwen35_gguf_workflow` (llama.cpp commit
`149f5641b931aaf33697d70f9abc90c3e5300247`).

## Dated host-capacity falsifier, November 4, 2026

The separately reported VN1 probe had **15.45 GiB physical RAM, 6.90 GiB
available RAM, 62.56 GiB free disk**, with other host workloads present.
These are source-reported *rounded* observations; use fresh host-owned values
before deciding physical workload admissibility. The two official source
safetensors alone total **9,319,828,096 bytes (~8.68 GiB)**. A BF16 GGUF and
quantization will require additional memory and scratch. The upstream pinned
llama.cpp quantization documentation warns that whole-model loading may be
required. Thus **6.90 GiB is below the source-size proxy by ~1.78 GiB**;
available disk does not erase the RAM problem.

This screening proxy is **not** a measured BF16 GGUF peak-RSS bound; a
converter might stream or mmap, while other steps could exceed it. Passing the
screen never means adequate RAM, safe disk headroom, executable compatibility,
or owner-idle process permission. The possible Q4_K_M resulting file is
smaller, but its size does not qualify the prerequisite BF16 conversion step.

The available third-party Q3_K_M GGUF candidate is a **different unverified
derivative** and cannot be treated as an official post-trained HF→GGUF output,
nor as proof the trained correct8 adapter is attached.

The pinned Windows `llama-cli.exe --version` still faulted on VN1 in the
independently reported November 4 probe (access-violation exit code
`0xC0000005`). A separate pinned **VN2 native Linux** version-only binary
probe passed. That makes VN2 the better bounded *runtime-executable* candidate,
not a verified model-serving or BF16-conversion host. VN2's previously reported
7.6 GB total memory also gives no comfortable basis for whole-BF16 quantization.
Keep the physical owner/lease and resource gates separate.

## Offline reproduction

```powershell
python -m unittest -q tests.test_qwen35_gguf_capacity_screen
python -m trainingday.qwen35_gguf_capacity_screen --available-ram-bytes 7408818586 --free-scratch-bytes 67173317734
```

The CLI always returns **exit 1**, even with exaggerated capacity inputs, and
emits only inert argument arrays for an eventual host-owned HF→BF16,
BF16→Q4_K_M and correct8 LoRA→GGUF conversion. Do not pipe them into a shell
or infer provenance from them. They intentionally require local pinned input
paths and exclude remote downloads or caller-selected runtime executables.

Before any real conversion, verify source and adapter byte custody, pinned
transformers/torch/gguf dependency closure, the Windows/Linux runtime binary,
RAM and disk fit with a safely enforceable budget, exclusive process ownership,
and independent output hash/readback. Follow the separate
`OFFICIAL_QWEN35_GGUF_SOURCE_WORKFLOW.md` source-status distinctions. Real
Qwen3.5 hybrid 32-block conversion and adapter graph compatibility must be
confirmed from the resulting GGUF and actual bounded model load; tests of this
screen cannot establish them.
