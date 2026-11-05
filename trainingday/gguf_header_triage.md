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

1. Host custodian reads back the exact file and its external download/creation
   manifest. Record actual source URL, immutable upstream repository/revision,
   original source-shard SHA receipts and tokenizer revision.
2. Match those external manifests to R13's **actual** `adapter_config.json`
   donor and `D:\VERA\models\latest-trained\base` shard custody, not just the
   model family or name. Historical R13 donor is
   `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive@d61dd146c8fd44c9a49cdb7f59f34e17b61902d8`.
3. Independently verify the GGUF quantizer/source chain is derived from that
   donor and preserves compatible tokenizer, tensor mapping and LoRA targets.
4. Only after owner-exclusive host resource/lease approval, proceed to a
   bounded technical load and separately blinded conversational comparison.

Until all these are satisfied, R13+standard Q4 is **HARD NO-GO**, and the
preexisting Q5 is **UNKNOWN / NOT QUALIFIED**. The separate official
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` + R16
lineage is a different experiment; its 496-tensor correct8 LoRA remains
unconverted/unqualified for a GGUF adapter at the cited checkpoints.

### Public provenance lead (not a match receipt)

The public model card at
[`tinyopsec/Qwen3.5-4B-Uncensored-Aggressive-GGUF`](https://huggingface.co/tinyopsec/Qwen3.5-4B-Uncensored-Aggressive-GGUF)
describes a GGUF quantization of the `rodrigomt` Aggressive checkpoint and
lists a generic `model_q5_k_s.gguf` at roughly **3.07 GB**. The unrelated
Lappy filename is identical and its **3,072,993,056-byte** file size is near
that rounded figure. This is a **SEARCH LEAD ONLY**: file names and rounded
sizes do not prove matching SHA-256, exact upstream revision, conversion
parameters, or tokenizer. In fact, the card's prose says "Qwen2 (28 layers)"
while its GGUF model summary says `qwen35`, a contradiction that prevents
treating the card's architecture details as reliable proof. No model or file
was downloaded, moved or hashed by this source investigation.

Reference format specification: [ggml GGUF v3](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
and [gguf.h](https://github.com/ggml-org/ggml/blob/master/include/gguf.h).
