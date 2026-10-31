# R75 — offline R18 Qwen3.5-4B loader source preflight

This is a **read-only, standard-library-only diagnostic**. It does not
import `transformers`, `torch`, or PEFT, instantiate a model, read adapters,
schedule compute, change loader selection, unseal an evaluation bank, or
authorize training.

The official pinned Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` declares the full
`Qwen3_5ForConditionalGeneration` architecture with `qwen3_5` configuration,
nested `qwen3_5_text`, and a vision configuration. Current R18 MeZO and
QLoRA sources request `Qwen3_5ForCausalLM`, a class documented against the
text-only configuration. This **declaration mismatch does not prove** either
a successful or failed actual `from_pretrained()` call.

Primary source pointers:

- [Exact official model config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
- [Transformers Qwen3.5 model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)

## Invocation

First retrieve and independently authenticate the SHA-256 of the exact
official `config.json` bytes for the pinned revision **outside the producer's
own base-binding JSON**. Do not simply copy `files.config.json` from the
binding to the `--expected-config-sha256` argument and call it independent.

```bash
python -m trainingday.r75_qwen_loader_preflight \
  --base-binding /path/to/existing/base-binding.json \
  --expected-config-sha256 64-lowercase-hex-official-config-sha256 \
  --source-root /path/to/model-training-checkout
```

The audit requires the expected revision, ID, loader, a regular bounded local
`config.json`, a matching producer-declared SHA, a matching separately
provided SHA, and an exact observed full-model architecture/config family.
It parses each source AST to require one literal, unaliased
`from transformers import Qwen3_5ForCausalLM` statement in an unconditional
module/function suite and one direct `Qwen3_5ForCausalLM.from_pretrained`
model call with `local_files_only=True` in both R18 arms. It fails closed
on static shadow bindings, misleading class aliases, and conditional/nested
import or loader statements. This is deliberately narrower than general
Python loader syntax; alternate valid source layouts require a new review.
It reports the two source SHA-256 values, installed distribution metadata if
available, and the source-declaration mismatch. It does not print local file
paths or read model weight shards.

`DECLARED_MISMATCH_UNKNOWN_RUNTIME` is a **HOLD** classification even on a
successful exit. An exit zero means only that the expected mismatch was
reproduced from the pinned local files. The CLI returns nonzero on malformed
or contradictory source, config, or digest declarations.

## Qualification boundary

The supplied independent SHA is not authenticated *by this program*;
`external_hash_authority_verified` remains false. Source AST inspection does
not establish the effective loader implementation in a running process.
Distribution version metadata does not prove a compatible import. A later
host-authorized, fresh-process test must capture exact package versions,
actual class/config behavior and warnings, model shard load, placement and
memory, and a finite text-only forward loss. That test must happen under an
exclusive documented resource lease and cannot be inferred from this report.

No R18 optimizer or model-serving implementation is changed here. Switching
to a multimodal class before compatibility, PEFT target coverage, and VRAM
fit are separately tested would be an unqualified change.
