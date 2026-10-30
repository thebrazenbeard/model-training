# VERA::TRAININGDAY — R15 matched numeric question pilot

This is a **development experiment**, not a Vera model release. Branch parent:
`model-training#131@02a3bb4daa830f2e89d0ea5dfe3809d0df7c25c6`.
Bus coordination: `thebrazenbeard/chat-communication-bus#404`.

## Controlled change
The same `R15_STRICT_INTEGER_V1` numeric-only suffix is applied by
`trainingday.numeric_prompt.numeric_question` in training *and* heldout
evaluation. R14 trained raw arithmetic questions while its V3 evaluator
added this suffix; the negative comparison was valid but format-confounded.
Changing instructions now establishes a **new** development pilot, never
a retroactive R14 PASS.

## Exact proposed run
- Corpus: v2 manifest SHA256
  `90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836`.
- Labyrinth source PR #59 head:
  `1d38e4f25b5663bf8a84ff5e139875d4152e3680`.
- Train: verified_reasoning only; code-source-only and eval excluded.
- Frozen official Qwen3.5-4B base `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Matched zero/correct/corrupt arms, seed41, batch1/GA8, 8 steps,
  64 selected cases; record actual optimizer steps and tensor hashes.
- Use separate v2 structural development eval, independent oracle,
  plus retention controls before any scaling or GGUF promotion.

## Evaluator custody gate

The R15 evaluator now rejects any manifest whose bytes do not match the
exact v2 SHA-256 above. For adapter evaluation, it additionally verifies the
saved adapter and base-binding hashes, requires the precise R15 recipe and
`R15_STRICT_INTEGER_V1` prompt contract, and matches the training receipt's
corpus manifest and Labyrinth producer head to the evaluator arguments.
Generic R14 adapter *reload* verification remains supported, but an R14
adapter is not admitted to the R15 comparison.

Output metadata records an R15 development protocol and the R14 selection
namespace retained to compare exactly the same case IDs. The R14 V3 bank was
previously used for development, so these case IDs are **reused, not fresh,
sealed, or final**. A separate structurally unique frozen bank must qualify
transfer before any release decision. The reported answer NLL also includes
EOS and should not be described as numeric-token-only likelihood.

**Limits:** the runner remains an eight-step pilot without resume or
consumer-side proof of producer Git head. New numeric formatting does not
prove reasoning transfer, nor do the 77 held-out primitive-operation
sequences constitute novel branching or program semantics.
