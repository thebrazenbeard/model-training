> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# VERA Model Training — Zero-Cost Identity Bootcamp

This repository hosts an **optional external training workbench** for native ChatGPT Project identities. The native Project does not depend on this repository for authority, memory, or ordinary operation.

## Current execution protocol

Current repository work follows `PROTOCOL_V2_CURRENT.md`. That correction prevents redundant lease/permission recursion for already-assigned isolated reversible work while preserving this repository's real privacy, zero-cost, source-grounding, qualification, and protected-effect gates.

## Hard invariant: zero monetary cost

The bootcamp must not require metered model APIs or paid compute.

The default engine runs an open-weight GGUF model locally on a standard GitHub-hosted runner in this public repository. The workflow fails closed if the repository is private or if `ZERO_COST` is not `true`.

No OpenAI API key, Gemini key, Hugging Face inference key, Supabase paid compute, or other paid inference service is required. There is no paid fallback.

## What a run does

A single bootcamp run performs externally:

1. bounded acquisition of user-selected public HTTPS sources;
2. source-grounded knowledge synthesis;
3. source-support filtering of the proposed capability map;
4. ten or more Trainer → Student → Examiner proxy rounds;
5. adversarial exercises and false-premise checks;
6. two independently generated transfer exercises;
7. conservative distillation into a compact identity capsule;
8. a package record, regression set, and detailed one-day audit artifact.

The target native ChatGPT conversation does **not** need the full training transcript.

This is not neural-weight training. The local Student is a proxy used to exercise and refine the training material. The transferable product is an **auditable training capsule**, not a claim that the native identity magically inherited the local model's scores.

## Qualification boundary

The external workbench may establish that a **training package is ready**. It may not declare the native ChatGPT identity qualified.

Trainer, Student, and Examiner use isolated prompt contexts but the same small local model, so proxy grades are diagnostic evidence only. Capabilities with only one proxy observation are explicitly marked provisional. The source set is always treated as bounded rather than exhaustive.

A fresh native ChatGPT cold/transfer evaluation remains required for PASS, CONDITIONAL PASS, or FAIL of the actual native identity.

## Trigger

Open an issue whose title starts with:

`[BOOTCAMP]`

and whose body is YAML:

```yaml
identity: SB
function: Supabase Platform Specialist
rounds: 10
target: practitioner
zero_cost: true
sources:
  - https://supabase.com/docs/guides/database/overview
  - https://supabase.com/docs/guides/database/postgres/row-level-security
```

Only the repository owner can trigger a bootcamp. The issue is the one-turn bridge: ChatGPT can create it, the external workflow performs the multi-round work, and ChatGPT can retrieve and audit the result without placing every training round into the target conversation.

## Privacy boundary

This repository is public to preserve the zero-cost execution model. **Do not place private chat transcripts, secrets, personal data, credentials, confidential source text, or sensitive identity material in bootcamp issues or source URLs.** Pass only a function specification and public sources.

For private/sensitive material, this public execution mode is not appropriate.

## Source and task guards

- HTTPS sources only.
- Literal loopback/private/link-local source addresses are rejected before fetch; HTTP redirects are refused before following any destination.
- **Remaining network boundary:** the hostname DNS resolution is not pinned to a validated public address, so DNS rebinding/private-target resolution is not yet excluded. Do not treat the source acquisition path as an SSRF-proof crawler. A future integration should use Ingest's validated-address connection-pinning transport.
- Capabilities that lack lexical support in the bounded source material are removed before training.
- Weak generic hidden rubrics are replaced with capability-specific and unsupported-claim checks.
- Adversarial tasks receive an explicit false-premise/unsafe-shortcut check.
- Newer runs supersede obsolete in-progress production runs.

## Output

Each successful run creates:

- `IDENTITY_CAPSULE.md`
- `QUALIFICATION.json`
- `REGRESSION_SET.json`
- `TRAINING_AUDIT.json`
- `SUMMARY.md`

The detailed workflow artifact is retained for one day. The compact summary is posted to the triggering issue.

## Default local inference

- `ggml-org/Qwen3-1.7B-GGUF`
- `Qwen3-1.7B-Q4_K_M.gguf`
- executed through a checksum-pinned `llama.cpp` server
- CPU-bounded prompt/output profile for ordinary GitHub-hosted Linux runners

Both the model file and llama.cpp archive are checksum-pinned in the workflow.

## Intended native flow

```text
native ChatGPT request
        ↓
create owner-only [BOOTCAMP] issue
        ↓
zero-cost external multi-round workbench
        ↓
retrieve package + one-day audit
        ↓
native ChatGPT audits material claims against primary sources
        ↓
compact audited capsule goes to target identity
        ↓
fresh native cold / adversarial / transfer qualification
```

That final native test, not the local model congratulating itself, is the competence gate.
