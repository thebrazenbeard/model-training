# GGUF ancestry triage (metadata only, source candidate)

**EVIDENCE CLASS: UNTRUSTED_GGUF_HEADER_METADATA_ONLY.** This is a bounded,
read-only inspection aid for an existing GGUF candidate, not an inference run,
adapter load, integrity hash, or upstream lineage certificate.

The standard GGUF v3 header provides — among other fields — model architecture,
text name and potentially author-supplied source metadata. It does **not** by
itself cryptographically bind a GGUF to an immutable Hugging Face source
revision or prove that a quantized model matches the trained LoRA's original
unquantized weights. A filename like `model_q5_k_s.gguf` and an embedded model
name can be changed without changing a tensor; the reported values are not
independent custody evidence.

## Safely inspect an existing file

On a machine with an actual local Python 3.12+ interpreter and an owner-approved
idle read-only slot, run:

```powershell
py -3 -m trainingday.gguf_header_triage "D:\VERA\models\gguf\model_q5_k_s.gguf"
```

or on Linux:

```bash
python3 -m trainingday.gguf_header_triage /exact/path/model.gguf
```

Run it from the checked-out `model-training` repository or with the package
installed in a private environment. It has no network calls and uses only the
standard library. It opens one existing file read-only, refuses non-regular
files, and parses **GGUF v3 metadata only**, without advancing into tensor
descriptors or tensor data. Hard limits:

- 64 MiB maximum metadata offset/scan window.
- 16 MiB maximum actual bytes read.
- 20,000 metadata entries, 1,000,000 total array elements, and nesting depth 3.
- 512-byte selected display strings; tokenizer lists and other large arrays
  are skipped, not printed or retained.
- Selected fields allow-list only; reject malformed length, duplicate key,
  unknown type, truncated header and out-of-window seek.

On systems supporting `O_NOFOLLOW`, symlink final components are refused;
Windows reparse/ancestor custody remains an external host preflight. GGUF
headers may be larger or differ in format; a parser refusal is `UNKNOWN`, not
evidence that the model is bad. Untrusted metadata itself can also lie. Never
automatically trust or import metadata as a trained-base manifest.

## What would make the local Q5 an eligible *candidate* for R13?

1. **File identity passed (W48):** the host custodian independently hashed the
   existing Q5 file and matched the full SHA-256 and byte count to a public HF
   LFS pointer. Recover the creation/download manifest and exact upstream
   repository/revision, original source-shard SHA and tokenizer custody.
2. Match those external manifests to R13's **actual** `adapter_config.json`
   donor and `D:\VERA\models\latest-trained\base` shard custody, not just the
   model family or name. Historical R13 donor is
   `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive@d61dd146c8fd44c9a49cdb7f59f34e17b61902d8`.
3. Independently verify the GGUF quantizer/source chain is derived from that
   donor and preserves compatible tokenizer, tensor mapping and LoRA targets.
4. Only after owner-exclusive host resource/lease approval, proceed to a
   bounded technical load and separately blinded conversational comparison.

Until all these are satisfied, R13+standard Q4 is **HARD NO-GO**, and the
preexisting Q5 has **verified GGUF binary identity but UNKNOWN trained-base
compatibility / NOT QUALIFIED for R13**. The separate official
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` + R16
lineage is a different experiment; its 496-tensor correct8 LoRA remains
unconverted/unqualified for a GGUF adapter at the cited checkpoints.

### Verified GGUF binary identity, not trained-base compatibility

The public repository at
[`tinyopsec/Qwen3.5-4B-Uncensored-Aggressive-GGUF`](https://huggingface.co/tinyopsec/Qwen3.5-4B-Uncensored-Aggressive-GGUF)
describes a GGUF quantization of the `rodrigomt` Aggressive checkpoint.
Its exact [`model_q5_k_s.gguf` file page](https://huggingface.co/tinyopsec/Qwen3.5-4B-Uncensored-Aggressive-GGUF/blob/main/model_q5_k_s.gguf)
reports rounded **3.07 GB**, SHA-256
`15150f534dc90ee15c82320ccc014463db7985205a7161228cff89b925fd1216`,
Xet hash `85203e0c3b62cd99aa40a671c40d9d711a81dfb999f678bbeee0ba52c72aad67`.
The primary [Git LFS raw pointer](https://huggingface.co/tinyopsec/Qwen3.5-4B-Uncensored-Aggressive-GGUF/raw/main/model_q5_k_s.gguf)
exposes exact `size 3072993056` and the same SHA-256 OID.
The **narrative model card contradicts** its file listing: it says Q5_K_S is
`~2.5 GB` and "Qwen2 (28 layers)", whereas the automatically shown GGUF
summary says `qwen35`. W48's **separate live host custodian readback** hashed
the actual preexisting Lappy file
`D:\VERA\models\gguf\model_q5_k_s.gguf`: **3,072,993,056 bytes**, SHA-256
`15150f534dc90ee15c82320ccc014463db7985205a7161228cff89b925fd1216`,
using a bounded, idle-priority read-only PowerShell operation lasting 48.18 s.
The file's binary digest and length therefore **match the HF LFS pointer**.
This evidence is `LIVE_HOST_FILE_SHA256 + PUBLIC_REMOTE_LFS_POINTER`, not a
training-source tensor comparison. The GGUF publisher's claim of deriving
from `rodrigomt` does not establish exact donor revision `d61dd146...`,
checkpoint tensor mapping, tokenizer equivalence, LoRA target compatibility,
or evaluated conversational benefit. No GGUF load, inference, file transfer,
download, or model mutation was performed. This source-only inspector itself
has not yet been exercised on the physical Q5 file.

Reference format specification: [ggml GGUF v3](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
and [gguf.h](https://github.com/ggml-org/ggml/blob/master/include/gguf.h).
