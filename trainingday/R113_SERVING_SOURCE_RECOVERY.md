# R113 serving-envelope source recovery

This is a source-only recovery candidate based on the open Qwen loader-preflight
Draft PR #174 (`d19e2a5172d44dd31ed8536672c1587f6ac68503`). The R25
synthetic-custody foundation was reintroduced from the earlier local source
commit `b26cbd124dfb33145941013636ef20eee320e256` and then hardened.

The base model architecture field must match the pinned Qwen3.5-4B config's
`Qwen3_5ForConditionalGeneration`. That is **not** proof that either R18
training runner's `Qwen3_5ForCausalLM` loader will load weights correctly.

The in-memory serving envelope admits only exact built-in JSON-like containers,
primitive field names and values, structural LoRA A/B metadata, and matching
caller-declared source/quantizer/reload hashes. It explicitly refuses to grant
deployment or model-promotion authority. The hashes do not establish tensor
bytes, independent quantizer execution, genuine fresh-process reload, release
rollback, host fitness, or neural behavioral gain.

Source tests are deliberately CPU-only with adversarial container/key spoofing.
This branch has no physical model loading, GPU use, production configuration,
protected evaluation access, merge, deployment, or release authorization.
