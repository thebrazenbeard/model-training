# W42 real-Q3 dialogue replay: evidence, not a new qualification

This is an **archival, source-labeled audit** of six real-model replies reported in [Bus #404 comment 6646210315](https://github.com/thebrazenbeard/chat-communication-bus/issues/404#issuecomment-6646210315). It runs no neural model, loads no weights, and produces no new model measurements. Do not report its passing unit tests as a successful Vera conversation. It is intentionally separate from the seven-case synthetic transfer casebook in `conversation_transfer.py`; those cases have not been given to this physical Q3 candidate as a published test result.

## How to reproduce the archival audit

From the repository root:

```sh
python -m trainingday.historical_conversation_replay trainingday/fixtures/w42_q3_prompt_only_real_probe.json
python -m unittest -v tests.test_historical_conversation_replay
```

The JSON fixture names the exact Bus source comment, Q3 GGUF SHA256, profile Git blob, runtime and two physical process receipts; it preserves all three identical user turns and their six reported assistant replies, plus a labeled **internal** model review. Each review quotation must be a substring of the cited turn. The fixture SHA256 is reproducible canonical JSON, not a signature or independent authentication. The audit validates internally consistent transcription and stops unsupported upgrades of the evidence class; verification against the live Bus source is still an independent review responsibility.

| Subject | Evidence | Boundary |
|---|---|---|
| Six replies | HISTORICAL_REPORTED_REAL_MODEL_PROBE, VN2 `llama.cpp` process receipts in Bus | Historical physical probe report; not rerun here |
| Two first-turn sarcasm misses | INTERNAL_NATIVE_MODEL_REVIEW; raw quote evidence | Not a blind external or human review |
| Corrections after user names sarcasm | The reported turn-2/turn-3 replies | Does not demonstrate proactive pragmatic recognition |
| Prompt delta effect | Same GGUF, settings, user turns; one appended system-instruction delta | Single example; no demonstrated benefit; no causal effect on weights |
| Vera identity, durable memory, deployment | No qualifying evidence from this fixture | Not established; historical source's current history is not persistent memory |

Reported observations: in both arms the model initially took a sarcastic refrigerator complaint as admiration. The baseline proposed diagnostics; the delta invented a compressor hum. Both repaired when directly corrected. Both already emitted a first-line bracketed cue without the delta. See the raw replies in the JSON rather than applying a model-generated aggregate score. No hidden holdout was used here.

> **Hostile falsifier:** A malicious editor could change model text, edit its review quotes, recompute the SHA256 and pass the audit. A content hash is not a source signature. Cross-read the referenced immutable Bus comment and, ideally, original terminal receipts. Neither fixture validation nor approving a PR can turn a limited prompt-only comparison into verified `BASE_VS_ADAPTER` effect or proof of self-ID/voice/memory.

## Next real-model experiment, when host and exact base are qualified

1. An authenticated host owner supplies an explicit process lease (CPU, memory, swap, workload exclusions) and readback; do not load if the host is occupied or its tunnel times out.
2. Verify **exact trained-base weight/tokenizer/adapter lineage** before an R13 study. R13's `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` source and the available `lmstudio-community/Qwen3.5-4B-GGUF` Q4 are **not proven equivalent**. No meaningful adapter A/B until corrected.
3. Freeze a held-out, multi-turn conversational set before inference. Include unprompted correct self-identification, dry humor that doesn't dominate, sarcasm with no explicit label, recovery from social mistakes, the ability to distinguish a fact from a guess, current-session reference, and honest denial of durable memory it has not received. A separate reviewer scores complete raw transcripts against prepublished criteria with model/seed/prompt/runtime/host/timing and resource receipts.
4. Compare a matched true base versus base+adapter using identical inference settings and observed history; evaluate prompt-only variants separately. Use paired statistics over sufficient prompts/seeds, not one anecdote. Never substitute CI fixtures, a successful LoRA file load, or a synthetic casebook for physical inference.

Until those gates exist, the meaningful live finding remains **no established satisfactory conversational behavior**, with an observed historical failure in sarcasm recognition. This branch preserves an unembellished baseline, not a trained personality.
