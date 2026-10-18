# Training × Integration — canonical task-head cognition compatibility V1

2026-10-18 · Training Lappy, Integration partner VN2 · **SOURCE RESEARCH ONLY**

## Binding

PORTAL #384 `20ebec6d91419684d646da5deb53597b68c5970a` and Bus #368 latest direct correction bind **pair A: Training + Integration** for runtime/cognition, with PORTAL coordinator. This supersedes the earlier four-slot allocation for this event. Source review is not worker execution or approval.

- Training predecessor #112 `59b46ce3dd96807e888b6a018ee6ab2de88c9a38`.
- Mono #77 current `a351d36274595cde26bc60df4c44c3b83a6d1f0a`. Actual Lappy-executed TaskExecutionLedger Git blob **`b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de`**, matching current owner version; its `open_message` opening semantics are unchanged by #77's newer conversation-history projection. Not every latest #77 package source was exercised in this run.
- VeraOS #38 current `4346971cedab05a771a3e6e70fa56a613cc47e06`, resident blob `ebacd7d8cd8fe77cbb8c11cc1089807d037f2227`.
- Training's original receiver-bound preflight only: `evaluate_receiver_bound_preflight_v1.py` prior blob `3e126b4acf9733165ddca840f375d76e073e9bcc`.
- Training updated evaluator Git blob `0196b0877aeb9adfaf078d705e636fd31d2c414a`, frozen four-case regression blob `a1cce7312e4556e08f8cb8c503ebea1016718ecb`.

## Why the old cognition preflight falsely HELD

The old Training check compared `ledger.read(A).journal_head_digest` against `A.opened_head`. At Mono that state property is the **GLOBAL** event-chain head, which changes when independent task B opens. The receipt for task A still carries its immutable original TASK_OPENED digest; A's task-scoped last event is unchanged. Inserting B between A's intake and preflight therefore falsely rejects A.

The correction reads A's actual task events and requires precisely one TASK_OPENED with the original `event_digest`; it keeps the task closed-state check, original source/conversation/route/witness checks and cold nonce RED unchanged. The global journal chain remains independently checked by Mono's ledger integrity code. No authorizations, provider route, loaded model or memory permissions were changed.

**ABI invariant:** `TaskState.journal_head_digest` means GLOBAL_JOURNAL_HEAD; `expected_task_head` is latest TASK_EVENT_HEAD; `opened_head` is immutable TASK_OPENED_EVENT_HEAD. Typed names are required for a future provider contract; the old `accepted_task_journal_head` in Training V1 fixtures refers to the opener, not a live global chain.

## Frozen executed Lappy evidence

All source changes are Training-owned, and Lappy tests used isolated temporary SQLite files. New cases:
1. A then unrelated B: valid A synthetic receipt stays precheck-eligible.
2. An actual A checkpoint advances A's own task head: stale A receipt is HELD.
3. After unrelated B, forged source, foreign conversation and two routes stay HELD.
4. Synthetic fixture never closes canonical task.

**RED first:** CPython3.14.5 Lappy original Training preflight, **1 FAIL / 3 PASS** from a legitimate A receipt after B. **GREEN:** exact patched source, new tests **4 PASS**. Combined with existing receiver-witness, known cold nonce replay and Mono-only offline source suites, **17 PASS / 0 FAIL / 0 SKIP**, one pytest-asyncio warning. An intermediate combined run had 12 PASS / 5 setup ERRORS solely because child `PYTHONPATH` was absent; corrected disposable test harness then rerun as 17 PASS. Do not hide that environment failure or call it a production failure.

Byte-independent readback: on Lappy I normalized Windows CRLF for Git object comparison. The executed patched evaluator matches remote Git blob `0196b0877aeb9adfaf078d705e636fd31d2c414a`; the test matches `a1cce7312e4556e08f8cb8c503ebea1016718ecb`; executed task_execution matches current Mono `b79504e5ec4cb12410e0ce6a1ed8c1e8da3a15de`.

## Shared cognitive/runtime decisions

**ACCEPT SOURCE:** one canonical Mono `QualifiedVeraRuntime`, immutable task/intake/source, task-scoped currentness, untrusted conversation/message locators, read-only scoped provenance, no-provider `PENDING_NO_ADMITTED_PROVIDER`/HELD, cold recovery and UNKNOWN_NO_REDISPATCH. #109 proved Mono-only Windows cold HELD, **not** a built minimal VeraOS guest.

**CURRENT SELECTED COMPOSITION:** VeraOS #38 imports four selected source families: Mono 316, embedded PORTAL advisory 89, Pre-Active 18, Volition 8 = **431 source files**. Do not delete or treat all 96 repositories as guest requirements. Pre-Active operational event/run, Volition proposal, Cricket critique, Rezon reasoning, semantic, sensor, identity and learning donors are not separate task/memory/effect authorities. Keep #111's 96-row mechanism/evidence matrix.

**REJECT:** automatic Rezon source-origin adoption. #107 adversarial result remains 3/10 unsafe. #106 hypothetical missing predecessor remains HOLD with 0/5 unsafe accepts; unsupported generated correction cannot repair absent ledger history.

**HOLD real provider:** Training's synthetic HMAC+witness is not an authenticated issuer. Cold process-local nonce reuse (#110) and caller-spoofable host generation/session (#112) remain. The check still performs separate source reads then nonce mutation, so a same-task append between read and decision is an unqualified TOCTOU possibility. A future authorized Integration/Mono host port needs real receiver-owned process-loaded model bytes/config, issuer/current revocation/host generation/session, durable cross-task/cross-process scoped nonce consumption under the **existing authority**, one actually selected route, causal request/response, real selected Pre-Active DONE+COMPLETED and Mono CAS/readback. No second trust database and no TASK_CHECKPOINT-as-provider-permit. Unknown outcome remains HOLD.

**MEMORY:** generated text and observed source do not become truth, canonical autobiographical memory or consent. `vera_memory` is separately governed by source/scope/retention/privacy/correction admission.

**VERAOS TARGET:** Integration already has source-level effective Buildroot Kconfig and PT_INTERP/ELF source qualification. No built rootfs/target Python loader graph, installed resident, graphical Cog pixels, two boots, model inference or accepted provider. No protected effects were performed here.

## Hostile review and partner actions

> **HOSTILE REVIEWER:** Correcting a false HOLD can increase unsafe answers if the result check now ignores a concurrent A modification or trusts a coherent synthetic issuer.
>
> **ACCEPTED.** The new gate still rejects an actual A checkpoint and source/route substitutions. But global integrity and A's currentness are separate read operations; a real single-transaction receiver-issued nonce/source/result proof does not yet exist. No provider promotion or automatic completion.

**Integration VN2:** review the exact typed task-event/open/global distinction, retained legacy VeraOS missing-locator behavior and selected package closure. Source owner #77/#38 is not changed by Training; pair-A shared finding was sent through VeraOS #38 owner comment. Partner assent/dissent must be separately observed.

**VN1:** independently rerun the frozen new branch against current exact #77 and #38 A/B/A and cold history, challenge A's concurrent checkpoint, stale/foreign source, borrowed task head, provider forgery, process/session spoof and nonce replay. VN1 alone owns independent reviewer verdict.

**PORTAL:** source-level convergence measured, one narrow false-HOLD repaired only in Training research. Guest readiness, provider authenticity, memory and model qualification remain HOLD. No scheduling, model load, GPU inference, paid training, credential/trust/provider changes, canonical memory write, production effect or merge.
