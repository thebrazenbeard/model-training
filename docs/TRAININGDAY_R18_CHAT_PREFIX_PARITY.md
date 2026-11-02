# R18 strict Qwen chat-prefix parity — source-only HOLD

The official R18 plan pins `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Both the R18 QLoRA
completion-only label builder and the private evaluator now use the same
`r18_generation_prefix` function. It calls the tokenizer's chat template with
`tokenize=False`, `add_generation_prompt=True`, and `enable_thinking=False`.

Previously, the trainer's borrowed V10 helper caught `TypeError` and retried
without `enable_thinking=False`, while the R18 evaluator required that keyword.
This could let the trainer format a prefix the evaluator would refuse, or let
two different thinking-mode contexts pass as equivalent. R18 now rejects an
incompatible callback rather than silently changing templates.

This does **not** authenticate an actual tokenizer, model snapshot or serving
GGUF runtime. It is an offline callback-contract regression with synthetic
characters; the protected host must independently attest the tokenizer bytes,
official revision and actual generation prefix before training and deployment.
The Labyrinth cognitive multiple-choice `Answer:` source format is a distinct
TRAIN lane and is not rewritten or silently imported into R18 arithmetic.

No protected DEV content, model weights, GPU or physical launch path is touched.
`trainingday.physical_launch_authority` remains an unconditional HARD HOLD.
