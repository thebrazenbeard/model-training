# R202 official Qwen3.5-4B tokenizer-only profile — 2026-11-04

**Evidence class:** fresh local CPU tokenizer runtime probe. **NO** weights,
optimizer, GPU lease, model inference, training, evaluator unseal or learning
qualification. This is one bounded measurement on the R202 candidate at
`model-training@85deb6064d4138aa7078b2490d160a00b94ea148`, not a claim
about any later corpus revision.

## Source and device custody

- Device: Lappy, Ubuntu 26.04 WSL2 via Executor Lappy Dedicated.
- Interpreter: existing `/home/vera/.local/share/vera-trainingday/.venv/bin/python`;
  `tokenizers==0.23.3`; no package installation.
- Official tokenizer path:
  `D:\VERA\scratch\trainingday-official-qwen35-4b-851bf6e-20261101\tokenizer.json`.
- Observed size: **12,807,982 bytes**. Fresh SHA-256:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
  matching the separately reported official revision
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Isolated host probe:
  `D:\VERA\scratch\trainingday-r202-tokenizer-20261104-w3\r202_tokenize_lappy_w3.py`,
  SHA-256 `a2316a68bf0e43323659514b74321157d62fdfdfddb343e4546ce902061b384b`.
  This source loader verified exact SHA-256 of the two public GitHub modules at
  pinned commit `85deb606...` **before** running them in memory.
- Frozen public R202 1,536-TRAIN-row JSONL SHA-256:
  `3e9c67be9e690588c47e979af0f03981175fb4fc6c22cefcc5df8782c805726a`.

`Tokenizer.from_file` loaded only the 12.8 MB JSON and called
`encode(text, add_special_tokens=False)`. Two text representations were
profiled: prompt alone, and the literal prompt + newline + answer. No
chat/system template, BOS/EOS, loss mask, batching or runtime overhead is
included; adding these generally increases the token requirement.

## Observed full-corpus counts

| Scope | min | median | p95 (nearest rank) | max | >64 | >128 | >256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only, 1,536 | 29 | 58 | 79 | 82 | **512** | 0 | 0 |
| Prompt + answer, 1,536 | 31 | 64 | 86 | 91 | **757** | 0 | 0 |

| TRAIN family (256 each) | Prompt >64 | Prompt+answer >64 | Max prompt+answer |
| --- | ---: | ---: | ---: |
| `math_logic` | 0 | 0 | 40 |
| `language_tracking` | 0 | 0 | 49 |
| `tool_selection` | 0 | **244** | 74 |
| `dependency_planning` | **256** | **256** | 77 |
| `error_correction` | **256** | **256** | 91 |
| `evidence_uncertainty` | 0 | 1 | 65 |

**Conclusion: FAIL for a strict 64-token sample budget.** More than half of
the answers would not fit in a naive fixed-length `seq64` prompt+answer
representation; truncation could remove supervision and bias experiments.
All measured bare prompt+answer strings fit a **128-token** budget, but a
deployed chat template, EOS/system overhead, and low-memory activation costs
have **not** been measured. This is neither a 128-token optimizer-fit PASS nor
a recommendation to launch Qwen training on the 4 GiB GPU.

### Additional genuine default chat-template measurement

A second CPU-only run used the same pinned public R202 generator and local
official snapshot, but now loaded **only** tokenizer/config metadata via
`transformers==5.22.1`, `AutoTokenizer.from_pretrained(...,
local_files_only=True, trust_remote_code=False)`. For every row it rendered
one `user` message containing the prompt and one `assistant` message
containing the answer with `apply_chat_template(tokenize=True,
add_generation_prompt=False)`, then counted the actual `input_ids`.

| Default one-user/one-assistant chat template | min | median | p95 | max | >64 | >128 | >256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,536 R202 examples | 44 | 77 | 99 | 104 | **1,024** | 0 | 0 |

- Diagnostic script:
  `D:\VERA\scratch\trainingday-r202-tokenizer-20261104-w3\r202_chat_template_lappy_w3.py`;
  fresh local/remote matching SHA256
  `9a662113beed56b679c968fbbb3710d3512c4cc47d5232f534f2fd8adc4a5119`.
- A **rejected preliminary output** reported exactly two tokens for every
  row because `BatchEncoding.__len__()` counts its **two fields**, not its
  token IDs. The measurement was fail-closed, instrumented, and rerun by
  explicitly extracting `input_ids` and rejecting unexpected data types.
- This authenticates a **specific** default two-message chat-template length
  calculation with the pinned files. It does **not** establish the actual
  optimizer's training-input contract, additional system context, model
  architecture loading, label-mask/EOS placement, dynamic padding/packing,
  gradient activation memory, or Q6/host-owned training lease. Physical
  `seq128` Qwen training fit therefore remains **UNKNOWN**, even though these
  1,536 measured default-template examples are each below 128 tokens.

## Independent contamination and shortcut checks

The wave3 source tests regenerated 1,024 R201 public TRAIN prompt strings
from the frozen `calc_curriculum._parameters()` source helper and compared
against all 1,536 R202 rows. **Zero exact or case/punctuation-normalized
overlaps** were found. This is a text-level fact, **not** semantic-equivalence
proof or a protected DEV review.

Prior hostile review found that wave2's class allocation was fully exposed by
the row ID modulo 2/3. Wave3 switched to a public SHA-ranked balanced
allocation and content-hashed IDs. The old ordinal-modulo predictors now
match only 128/256 mathematical labels and 84/256 uncertainty labels.
This removes the specific *trivial* shortcut but not the ability to enumerate
the entire published generator. Tool intent still follows lexical cues;
correct arguments and unseen paraphrases remain untested capabilities.
Error-correction data now includes clean examples and both positive and
negative errors. No result here demonstrates learned generalization.

## Safety gate / next experiment

Keep R202 in `SOURCE_ONLY_UNAUTHENTICATED_HOLD`. Before including it in the
physical 64-token low-memory recipe, shorten and re-freeze the long families,
rerun this **exact** pinned-tokenizer profile, and require zero over-budget
examples **including** the actual conversation template and supervised
special tokens. Alternatively validate a genuinely memory-safe `seq128` host
recipe with an owner-owned process lease and recovery before launching a run.

The probe is reproducible with:

```bash
python -m trainingday.r202_tokenizer_probe --tokenizer /absolute/path/tokenizer.json
```

No artifact was deleted, copied, modified, or promoted by the tokenizer
measurement. The isolated diagnostic script on Lappy is small, contains no
credentials or model weights, and can be removed after custody review.
