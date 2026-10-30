# TrainingDay R27-01: public CPU checkpoint portability rehearsal

Status: **SOURCE + LOCAL LINUX CPU FIXTURE ONLY / CROSS-HOST UNKNOWN / RELEASE HOLD**.

Owner: native ChatGPT specialist `/root/r27_cross_os_checkpoint_portability`; branch
`trainingday/r27-cross-os-checkpoint-portability-20261030`.

## Scope and evidence boundary

`trainingday.r27_checkpoint_portability` provides a strict, bounded, no-torch
standard-library reader and writer for a **synthetic F32 safetensors subset**:

- 8-byte little-endian header length, UTF-8 JSON header, `F32` tensor fields
  `dtype`, `shape`, `data_offsets`, and packed little-endian finite FP32 values.
- Maximum 1 MiB fixture, 128 KiB header, 32 tensors, 65,536 elements per
  tensor, full contiguous coverage, no duplicate JSON keys, trailing bytes,
  nonfinite values, or unsupported dtype. Hash validates exact whole-file bytes.
- Deterministic serialization independent of map insertion order and an actual
  **second local Python process** that parses a fresh fixture from disk.
- Virtual relative path normalization rejects obvious Windows/Unix ambiguous
  names, casefold collisions, dot traversal, absolute/UNC/drive/ADS paths,
  Win32 COM/LPT device aliases (including superscript digits ¹²³),
  invalid UTF-8/Unicode normalization, and optimizer/RNG pickle checkpoints.
- Tensor shapes and values require exact built-in list/tuple types, not mutable
  subclasses with dishonest length/iteration; emitted F32 byte count must
  match the declared tensor shape.
- Logical Lappy / VERA-NODE-1 / VN2 SHA declarations can only produce
  `DECLARED_DIGEST_MATCH_ONLY`, never `actual_cross_host_verified=true`.

This subset is *not* a full safetensors implementation, serializer for Qwen
4B tensors, or Torch checkpoint restorer. It never deserializes arbitrary
Python pickles, optimizer states, RNG states or bytecode. Unsupported real
optimizer/RNG resume explicitly remains disabled.

## Host and artifact matrix

| Claim | Evidence class | State |
| --- | --- | --- |
| Synthetic F32 encode/decode + hash | Local source/CPU tests | TESTED locally |
| Same fixture fresh-process parse | Local process probe | TESTED locally |
| Windows Lappy fixture readback | Not executed against host | UNKNOWN |
| Windows VN1 fixture readback | Not executed against host | UNKNOWN |
| Linux VN2 fixture readback | Not executed against host | UNKNOWN |
| Actual R18/Qwen 4B adapter byte equivalence | No fresh 4B process | UNKNOWN |
| Real optimizer/RNG checkpoint resume | Unsupported/prohibited | HOLD |
| Neural improvement/quantized release | No behavioral/host custody | HOLD |

The local fixture's exact SHA, Git commit/tree, test count and independent
review must be bound in the lead's evidence record after the final head is
verified; do not invent remote SHA or physical host receipt. Report host CPU,
OS, PATH and PyTorch/safetensors version fingerprints only after actually
capturing them; any future three-host test requires a separate authorized
artifact-staging path with isolated resource leases and no shared model writes.

## Hostile limitations

1. Hashes from three caller-provided dictionaries are self-declared, not
   independently authenticated file reads, Windows ACLs, or Executor leases.
2. Local `Path` symlink precheck is not a cross-platform atomic path-open
   guarantee. These probes read only temporary/public fixtures.
3. The actual R18 run consumed ~19,813.562 seconds against a 531-second
   budget, overlapping Modern Standby. This historical run is
   `TIME_BUDGET_OVERRUN_INVALID` for fair equal-time comparison regardless
   of public safetensors integrity.
4. No private evaluation data, secrets, real model adapter, original runtime
   checkpoints, CUDA state, deployed Vera service, OS power settings or
   external GitHub state is changed by this source candidate.

No new neural-learning, graduation, deployment or merge authority is granted.
