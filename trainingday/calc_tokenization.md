# Calc → pinned Qwen text-token seam (source-only HOLD)

`trainingday.calc_tokenization.compile_calc_train_tokens` converts the
**public 1,024-row R201 Calc polynomial derivative TRAIN candidate** into
causal language-model examples. Every record must contain exactly `id`,
`prompt`, `answer`, `split`; all rows must be `train`, have unique IDs/prompts,
and carry exact signed-integer answers. The separate manifest is checked against
the exact Calc commit (`17e12d0f659b42732fb98e5730732aba9ff1b488`),
the symbolic polynomial oracle family and a canonical TRAIN JSONL SHA-256.

The source contract also freezes the exact `calc.symbolic` SHA-256
`aba705f0d4370afa35538280f3db00ecbe36da1b3f3ca083ee29ebc292113b0b`
and the complete 1,024-row TRAIN JSONL SHA-256
`fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b`.
A caller cannot create arbitrary new rows and self-hash a manifest to pass this
consumer. This is still a source pin, not an authenticated identity grant.

The tokenizer is a caller-provided object exposing `encode`, `decode` and an
integer `eos_token_id`. With `add_special_tokens=False`, each row uses
`Q: <prompt>\nA: <answer><EOS>`. Only the answer and EOS have non-`-100`
labels; the prompt is masked. The concatenated separately encoded tokens must
decode to the exact original text. Lengths above 64 **raise**, never truncate
the answer. The output is Python lists; PyTorch, Transformers, GPU state,
model weights and private DEV data are not accessed by this module.

The *model target* is bound separately from the model-agnostic Calc manifest.
Only `Qwen/Qwen3.5-4B` at R18's exact revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` is admitted as a declaration.
The distinct `Qwen/Qwen3.5-4B-Base` and Aggressive variants are rejected.
The caller must separately establish the verified tokenizer and weight bytes
using the host-owned pinned-base boundary before any protected operation.

**Important limitations**

- Matching the frozen manifest and content hashes is **structural consistency**, not authenticated
  rights, source custody, independent mathematical verification or a GPU lease.
- No hidden DEV rows, answers or nonce enter the tokenizer. The module refuses
  non-TRAIN IDs and positive rights/GPU/evaluator/training flags.
- We have not exercised this seam with the real Qwen tokenizer or full Calc
  corpus on Lappy. Tokenizer-specific merges may trigger a safe rejection.
- The source-only output cannot launch training and reports zero optimizer
  steps. The physical TRAININGDAY broker remains an unconditional HARD HOLD.
- This exact-integer polynomial family says nothing about Calc's approximate
  derivatives/integrals, nonsmooth points or numerical quadrature aliasing.

Run `python3 -m unittest -v tests.test_trainingday_calc_tokenization`.
