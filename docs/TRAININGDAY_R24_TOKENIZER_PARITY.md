# R24 R18 MeZO / QLoRA tokenizer parity witness

This is a source/CPU-only witness. Neither the R19 token-weighted
cross-entropy correction nor agreement on a scalar loss proves that the
two arms supervise identical token IDs or positions.

Observed source paths at the independently reviewed PR #149 tree
`2f977515ff6571b0e6bf8debb886597e842a1d85`:

- `r18_mezo.tokenize_cases()` tokenizes the chat-generation prefix and
  `response + EOS` in separate tokenizer invocations, then concatenates IDs.
- `r18_qlora._tensorize()` tokenizes the prefix and the complete joined
  `prefix + response + EOS`, checks that prefix IDs remain stable, then
  supervises the suffix. Prefix stability alone does **not** guarantee the
  separately tokenized suffix equals the suffix tokenization in context.

`trainingday.r24_tokenizer_parity.inspect_tokenizer_parity()` runs those exact
preparers against the **same caller-supplied public/admitted examples** and
tokenizer instance. It requires equal, ordered input IDs, labels and
supervised-target counts; both no-truncation conditions must hold. No example
text or token IDs are returned. It always returns
`authorizes_training: false`.

Synthetic adversarial fixtures demonstrate:

1. A byte/character-local tokenizer can pass, with equal IDs and labels.
2. A prefix-stable but context-sensitive completion tokenizer can make the
   two runners supervise different token IDs and must HOLD.
3. A tokenizer that changes the prefix at the generation boundary is rejected
   by QLoRA and must HOLD.
4. A joined sequence that exceeds the admitted token limit must HOLD even if
   MeZO's separately encoded sequence fits.
5. Duplicate selections, invalid rows, and missing responses must HOLD.

## Physical qualification still required

An actual Qwen3.5-4B acceptance run would need a frozen base-tokenizer
revision/file identity, independently bound admitted corpus IDs and content,
the exact same rendered prompts and labels for **every** admitted case,
identical initial adapter, sampling schedules and repetitions, and an
independently witnessed, exclusive host lease. This source witness does not
attest any of those, measure GPU memory, open the protected evaluation bank,
or authorize either training runner. The historical R18 time-budget overrun
remains invalid comparison evidence.
