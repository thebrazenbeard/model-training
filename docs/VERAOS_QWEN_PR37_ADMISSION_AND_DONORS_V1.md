# VeraOS Qwen3 4B resident admission and donor triage V1

Status: Training L2/L3 source-only; no real model executed, installed or selected.

## Primary VeraOS #37 finding

Exact PR37 e134dd7881a68b719b27e3c0fdd627ee3dad241c
desktop.py 91b68d5adb49751dd6eca6e1a6b9c5d84c120f80
supports POST /api/conversation. An in-process caller can supply
conversation_template and a verifier callback returning literal
SYNTHETIC_SOURCE_FIXTURE_ONLY. Existing native_cognition admits
technically ready OPENAI_COMPATIBLE_HTTP providers via live
probe_model_target and a network-capable OpenAICompatibleAdapter.
Thus the synthetic-only token can reach a real local HTTP service if
an in-process caller supplies that endpoint. This is a source safety
hazard, NOT observed installed access or real inference. Default
resident CLI has no provider and remains HELD.

Historical PR37 head e134dd7881a6 returned HTTP 200 status
COMPLETED for an injected callback with run_id=None/final_text=None.
Exact-source independent RED reproduction is in
test_veraos37_http_source_optional.py with ephemeral fake HTTP, no
model. PORTAL subsequently advanced PR37 to exact 59a4d8f0d26c00104cf1ea6d87249d94fc5a58ad.
Its desktop.py blob ee20af7fb5de2b48d179bdfa22f518b3b619202b
now rejects missing terminal run/text (HTTP 502 OUTCOME_UNKNOWN)
and its resident worker cross-checks completed claims against the
existing Pre-Active Store. Training independently tests the new exact
HTTP source as GREEN in the same optional test file. Real transport
authentication and injected fixture isolation remain separate HOLDS.
Training source resident_http_completion_gate_v1.normalize_http_result
rejects missing event/run IDs, empty/oversized text, wrong provider,
tampered route, noncompleted result with text and attempted claimed
authority. Positive remains TEST fixture source only, with
model_authenticated=false, display_authorized=false,
canonical_task_completed=false and effect_authority=false.

PORTAL-owned minimal remedy:
1. Remove synthetic callback token path from resident.main production
   branch; confine fixture to a non-network, not-installed test adapter.
2. For real bounded development admission, use existing Vera Mono
   OutboundTrustRegistry PROVIDER currentness and independently
   installed verifier key, Lappy enrolled generation/session,
   receiver-issued one-time nonce, independently observed loaded
   Ollama blob/manifest/endpoint/capacity and Integration task-head CAS.
3. OS-minted task-scoped lease MUST be independently checked before
   native_cognition.admit_local_text_route; no caller PASS, inventory,
   synthetic token, or arbitrary Python callback confers trust.
4. At HTTP completion require a committed, readback-verified PreActive
   run/event/message and canonical Mono task binding; source-only
   structural gate alone cannot authorize display or claim Qwen identity.
5. Keep text-only, max context/output, empty tools and honest UNKNOWN
   response when effects or observations remain ambiguous.

Training port resident_verifier_port_v1 already exercises simulated
signed scoped requests and real Mono-compatible PROVIDER currentness.
Its real-host mode is deliberately HOLD because no installed verifier,
host-attested loaded model or durable one-use nonce has been demonstrated.
The source PREACTIVE readback from Training PR106 is synthetic-only too.
Full production PKI and Vera-trained model promotion can be deferred.

Placement: keep host Ollama Qwen3 4B Q4_K_M on Windows Lappy and the
network-isolated QEMU guest behind private host relay/AF_UNIX/VirtIO
loopback 127.0.0.1:18766/v1. VN1, Lappy and guest have different
identities. Previously verified Lappy stored Ollama manifest file
SHA256 0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0,
and model blob file SHA256 85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9
(2497280480 bytes). Stored file hash does NOT authenticate what
the future Ollama serving process will load. Guest RAM 768MiB is
not the memory holding those weights.

## Secondary read-only donors (do not affect Qwen milestone)

All paths under D:/VERA/models/donors/huggingface; all source worktrees
clean. Exact HEADs, README front matter, JSON configuration, tokenizer,
and bounded safetensors headers observed; no weights loaded and no
repository Python code executed. Previously verified full weight SHA256
from original move. Model evaluations NOT run.

Open-Reasoner-Zero-0.5B: HEAD a0720dbf6af0d56300c938cc5c9df82b143441bb,
988097824-byte safetensors, 494032768 BF16 declared elements, 290
tensors, Qwen2ForCausalLM/Qwen2Tokenizer, 24 layers, 896 hidden,
151936 vocab, chat template, no custom Python files. README indicates
MIT, upstream base rights still require checking. KEEP for later
isolated small-model reasoning research; Qwen2 is not Qwen3 4B and
format compatibility does not prove quality or authority.

GPT-X3-150M: HEAD 46407499ecaf5f0e81f2798aa302c5641f36d4b1,
597655008 bytes, 149400930 F32 elements, 464 tensors,
GPTX3ForCausalLM custom auto_map to configuration_gptx3.py and
modeling_gptx3.py, vocab 65536, no chat template. README Apache-2.0
and benchmark claims NOT independently qualified. QUARANTINE custom
code, no trust_remote_code; keep read-only architecture reference
pending independent code and dependency review, not resident provider.

uncensored-stage1-hacker: HEAD c7231ab736cc27203a651b47b600a66f8ef902be,
1192135096 bytes, 596049920 BF16 elements, 310 tensors,
Qwen3ForCausalLM/Qwen2Tokenizer, 28 layers, 1024 hidden,
chat template. README autogenerated placeholders with NO model
license, base weights, datasets or evaluation provenance. HOLD for
training/admission/redistribution until rights and lineage are verified;
no trust or competence inferred from architecture or name.

The prior interrupted donor move left three missing static images
in the Open-Reasoner checkout; they were restored from exact pinned
Git HEAD and all three worktrees now have clean Git status.

HOSTILE REVIEW: source syntactic checks and synthetic ACCEPT do not
authenticate a running OS verifier; live model access must have
independent caller capability isolation inside the resident.
