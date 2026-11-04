# TRAININGDAY official Qwen3.5-4B → GGUF Q4_K_M → Vera: source-only workflow

**Status: HARD HOLD.** This document and
`trainingday/official_qwen35_gguf_workflow.py` are reviewable source contracts,
not an installed model, a verified GGUF derivative, owner permission to start a
process, or an inference receipt. `python -m
trainingday.official_qwen35_gguf_workflow --json` exits **1** intentionally.
It does not read private model files, execute converters or spawn servers.

## Authenticated-source targets, separated from runtime claims

The official **post-trained** checkpoint is
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (not the
distinct `Qwen/Qwen3.5-4B-Base`). An independent 2026-11-04 VN1 custody
report says its *physical* restricted-store snapshot contained **26 files,
9,342,822,940 bytes**, including the **11 canonical content files below**.
The private independent sidecar's SHA-256 is
`06e730744a578cc2168d1b587711f82a8b7e691545182ff1b814dff95e450fca`.
This source module pins the dated report; it does **not** repeat its live hash
measurements or independently verify the sidecar's signer/immutability.

| Canonical file | Bytes | Independent report SHA-256 |
| --- | ---: | --- |
| `chat_template.jinja` | 7,756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `config.json` | 3,161 | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `model.safetensors-00001-of-00002.safetensors` | 5,329,398,688 | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` | 3,990,429,408 | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| `model.safetensors.index.json` | 76,196 | `cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer.json` | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | 16,710 | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

The other **15 inventory names** are 11 Hugging Face download metadata
entries, `.cache/huggingface/.gitignore`, `CACHEDIR.TAG`, one cached tree JSON,
and `TRAININGDAY_PUBLIC_SOURCE_RECEIPT.json`. These are separately inventoried
in source, but their individual hashes are **not pinned**. The 11 canonical
files total 9,342,816,694 bytes; the remaining 15 account for 6,246 bytes.
The source-only check never treats those auxiliary files as independently
verified simply because the total matches.

The *correct-label* R16 adapter is
`r16-v2-correct8-20261030-b`: `adapter_model.safetensors` is **32,531,344
bytes**, SHA-256
`322d91e034cd2b2cf63b077e98061559478a4271c0bf5d4abd8520dca195fc11`;
`adapter_config.json` has SHA-256
`d9c52b766f0eea6bb6e1cbce52ecafd024385822b0e75644ec4dd50b11024816`.
Never relabel the corrupted-label control as trained Vera.

## Toolchain identity and genuine runtime admission

- Upstream `ggml-org/llama.cpp` source commit
  `149f5641b931aaf33697d70f9abc90c3e5300247`, release **b12098**.
- Official Windows x64 CPU ZIP SHA-256
  `8d142d409515ce7a0f873654170ef0071d945acd1918fd774d374840917bf211`;
  extracted `llama-server.exe` SHA-256
  `a34c390204c49a5df796580c7e08962411f3f52f3187d7ef4353717b8ecdb767`;
  `llama-quantize.exe` SHA-256
  `3ede9649dcea3020df0fe1b1f2a1d436e68acdf33d7277c7d6cdf1b3e12555df`.
- Official Linux x64 CPU release tar SHA-256
  `7e6f56a0940e5db3744a766ba5480ab5e0ba30b61b656b682f3919666da6aae7`.
- The private Windows ZIP was independently extracted and **all 52 extracted
  file hashes verified**. This does not qualify executable startup. The official
  GitHub release is marked *prerelease* and *mutable*; retain immutable local
  digests, don't trust a tag alone.

**Historical failure (2026-11-04):** On VN1, the pinned Windows
`llama-server.exe --version` exited `0xC0000005` in `MSVCP140.dll` version
14.32 before stdout or a listener. The official Windows release used VS2026;
VN1's installed C++ redistributable was older. A runtime mismatch is a strong
hypothesis, not proven causation. The separate Ubuntu WSL2 binary had an
unresolved `libgomp.so.1` dependency on first static inspection. Do not turn
either failure into automatic permission to install system-wide packages.

## Bounded conversion sequence — illustrative, **not executed**

Only after a genuine owner-idle admission, physical RAM and IO headroom,
trusted dependency closure, immutable private source access, and explicit
process-tree custody are established:

1. From the exact authenticated 26-file snapshot, use the **pinned**
   `convert_hf_to_gguf.py` with `--outtype bf16 --no-mtp` to create a
   high-precision **text** GGUF. The original BF16 tensors should not be
   unnecessarily cast to F16. `--no-mtp` suppresses speculative **NextN**
   tensors; it does **not** mean “strip the vision subsystem.” Pinned
   `conversion/qwen.py` supports the conditional-generation architecture's
   text projection. Separate multimodal encoders/projectors are needed for
   vision/audio.
2. Rehash the original sources and new BF16 output independently. Record
   converter source commit, exact command, peak physical RAM, CPU/IO use,
   timestamps and exit status; a zero exit is not identity proof.
3. With a separately startup-qualified pinned binary, quantize **BF16 GGUF →
   Q4_K_M GGUF** using `llama-quantize`; record the output digest and a second
   independent readback after transferring it to restricted VN1 storage.
4. Convert only the SHA-pinned **correct8 R16** PEFT adapter using the same
   official HF base and pinned `convert_lora_to_gguf.py --base ... --outtype
   bf16`; verify real tensor compatibility and output digest. Historical
   upstream LoRA bugs were fixed before this converter revision, but that
   does **not** prove this physical adapter converts or loads.
5. In separately owned and bounded fresh processes, compare three genuinely
   distinct states: **base** (Q4 GGUF alone), **trained** (same Q4 plus converted
   correct8 adapter), and **interface** (trained state through Vera's local
   conversational entry point). Record model/adapter digests, tokenizer and
   chat-template behavior, latency, context and stop tokens, peak RAM, tool
   environment, response traces and failures. All current runtime flags in
   the source plan remain FALSE.

No caller-provided executable path, port, HTTP health response, byte-hash
declaration, or success flag may substitute for independent provenance or
admission. The prior hostile review found arbitrary-executable injection,
model-file check/use races, unrelated loopback listener spoofing, and orphaned
process trees. A future runtime controller must bind an independently pinned
executable and private GGUF file handles to an owner-held process tree and
verify the listener's actual PID. The current `probe()` remains an
**unconditional HOLD**; the new `execute()` is also an unconditional HOLD.

## Reproducible source-only check

```bash
python -m unittest -q tests/test_official_qwen35_gguf_workflow.py
python -m trainingday.official_qwen35_gguf_workflow --json
```

The first command runs negative tests. The second deliberately exits **1**
and emits `SOURCE_ONLY_HARD_HOLD`. Neither command accesses model weights.
Source/unit tests are not runtime, scientific-learning, host, or human review.
