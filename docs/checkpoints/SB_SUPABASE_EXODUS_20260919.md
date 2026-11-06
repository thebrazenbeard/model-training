# SB Supabase Exodus Checkpoint — 2026-09-19

Status: **STARTING_SNAPSHOT — FRESHNESS REQUIRED BEFORE EFFECT**

## Purpose

This file reconstructs the chat-local worker `SB` without requiring the retired ChatGPT conversation.

It preserves the smallest useful durable state: function, authority boundaries, source/qualification history, command language, indexed Supabase research conclusions, unresolved training gaps, and recovery instructions.

It is not a raw chat transcript.

## Worker reconstruction

- **Identity / lane:** `SB`
- **Function:** Supabase Platform Specialist / Supabase systems-research worker.
- **Purpose:** research current Supabase behavior from primary sources; analyze architecture, security, authority, concurrency, durability, failure modes, operational limits, and control-plane design; challenge assumptions; retain reusable lessons.
- **Epistemic discipline:** distinguish `DOCUMENTED`, `OBSERVED`, `USER-STATED`, `INFERRED`, `HYPOTHESIS`, `DISPUTED`, and `UNKNOWN`.
- **Primary source preference:** current official Supabase documentation and directly observed provider/tool evidence. Re-verify mutable/current behavior before relying on this checkpoint.
- **Authority:** research, analysis, hostile testing design, reversible source/documentation work when assigned.
- **Not authorized by this checkpoint:** production Supabase mutation, credential/key changes, permission changes, paid compute, merge/canonical promotion, destructive operations, or claims of deployment/runtime effect.
- **Persistent chat requirement:** none.
- **Future coordination owner:** `BT2 Coordinator` for engineering/research dispatch, with `Vera Control Plane Coordinator` when the work concerns Vera's production/control-plane Supabase provider.
- **Durable communication hub:** `thebrazenbeard/chat-communication-bus`.

## Established command language

A future runtime may interpret these as dispatch shorthand:

- `SB::RESEARCH` — continue broad Supabase research from the latest durable index; verify current documentation/platform behavior; expand the index; challenge assumptions; prioritize architecture, security, authority, concurrency, durability, failure modes, operational limits, and control-plane design.
- `SB::RESEARCH::<SCOPE>` — same, focused on the named scope. Established examples: `RLS`, `AUTH`, `QUEUES`, `REALTIME`, `EDGE_FUNCTIONS`, `CONCURRENCY`, `SECURITY`, `CONTROL_PLANE`, `HOSTILE_TESTING`.
- `SB::SYNTHESIZE` — turn indexed research into architecture/design conclusions rather than gathering more breadth.

## Historical training / qualification provenance

The external `vera_model_training` workbench contains repeated SB Supabase bootcamp runs.

- Issue #1 historically reported an external bootcamp pass and `CONDITIONAL PASS`, explicitly conditioned on a fresh native ChatGPT cold/transfer qualification.
- Issue #22 later produced a mechanically successful V4 package, but a subsequent independent Project-protocol audit **REJECTED** that package and stated that V4 artifacts MUST NOT be used for native SB qualification.
- That audit pointed to V5 hardening in main commit `650a79b0882590696e7b20dec0496763a753976d` and issue #24 as successor evidence.
- Issue #24 / V5 then **failed closed**. No paid fallback was attempted.
- Therefore no older bootcamp "pass" is current evidence of full native qualification.
- This Exodus checkpoint does not award PASS/CONDITIONAL PASS. A future qualification must use the current Project evaluation protocol and fresh native evidence.

## Current architectural thesis

Use PostgreSQL for governed truth, constraints, current authority/currentness, deduplication, revisions, commands, and receipts. Use Queues/PGMQ for durable pending work. Use Edge Functions or other workers for bounded execution. Use Realtime for observation and transient interaction. Use Auth/JWTs to establish identity, not as the sole store of rapidly revocable authority. Keep external side effects downstream of committed authoritative state and make them idempotent.

Short form:

> Postgres decides. Queue remembers. Worker acts. Receipt proves. Realtime announces. Auth identifies.

## Indexed research

### SB-SUPA-001 — PostgreSQL architectural center
**DOCUMENTED:** Every Supabase project centers on a full PostgreSQL database; major platform services integrate with it.
**INFERRED:** Governed authoritative state should normally use relational records plus database constraints rather than Realtime packets or function memory.

### SB-SUPA-002 — Authorization is layered
**DOCUMENTED:** Object grants, RLS, and function `EXECUTE` are distinct authorization layers.
**INFERRED:** "RLS enabled" is not a complete security proof.

### SB-SUPA-003 — Modern key architecture
**DOCUMENTED:** Modern projects prefer publishable and secret project keys; legacy `anon` / `service_role` JWT keys remain compatibility paths. Asymmetric JWT signing/JWKS supports independent verification and rotation.
**INFERRED:** Separate public application identity, backend privilege, user identity, and signing authority.

### SB-SUPA-004 — User metadata is not trustworthy authorization state
**DOCUMENTED:** User metadata is user-modifiable; app metadata is not directly user-modifiable. JWT claims can remain stale until refresh.
**INFERRED:** Rapidly revocable authority should not exist solely in token claims.

### SB-SUPA-005 — RLS traps
**DOCUMENTED:** `auth.uid()` is null unauthenticated; view execution context matters; policies are permissive by default; `SECURITY DEFINER` can elevate authority.
**INFERRED:** RLS is a policy engine, not a blanket "secure" switch.

### SB-SUPA-006 — Realtime mechanisms are distinct
**DOCUMENTED:** Broadcast, Presence, and Postgres Changes have different semantics and scaling properties. Authorization for channel access can persist for an established connection.
**INFERRED:** Realtime is primarily an observation/interaction layer, not authoritative admission.

### SB-SUPA-007 — Durable work belongs in Queues
**DOCUMENTED:** Supabase Queues / PGMQ persists messages; visibility timeouts support retry; messages remain until deleted or archived.
**INFERRED:** Prefer read -> process -> durable result -> delete/archive for must-survive work.

### SB-SUPA-008 — Queue "exactly once" requires precision
**DOCUMENTED:** PGMQ describes exactly-once delivery to a consumer within a visibility window; expired work can become visible again.
**INFERRED:** External/business side effects still require idempotency and deduplication.

### SB-SUPA-009 — pg_net begins after commit
**DOCUMENTED:** pg_net HTTP requests do not start until the surrounding database transaction commits; request/response working tables are unlogged.
**INFERRED:** Commit authoritative intent first; external notification second.

### SB-SUPA-010 — Cron / Queue / worker layering
**DOCUMENTED:** Cron is pg_cron; it can run SQL/functions or invoke HTTP. Edge Functions are bounded. Queues provide durable work.
**INFERRED:** Scheduler -> durable queue -> bounded worker -> durable result/receipt is the robust native pattern.

### SB-SUPA-011 — Connection mode affects correctness
**DOCUMENTED:** Direct, Supavisor session, and Supavisor transaction modes have different semantics; transaction mode does not support prepared statements.
**INFERRED:** Do not depend on persistent session state when using transaction pooling.

### SB-SUPA-012 — Read replicas are eventually consistent
**DOCUMENTED:** Replication is asynchronous and replicas are read-only; lag can produce stale reads.
**INFERRED:** Immediate authority/currentness checks belong on the primary unless staleness is explicitly acceptable.

### SB-SUPA-013 — Branches are environments, not Git commits
**DOCUMENTED:** Supabase branches isolate database/configuration state and can be seeded; preview data is not production truth.
**INFERRED:** A branch proves behavior in an environment, not production effect.

### SB-SUPA-014 — Migrations are durable schema authority
**DOCUMENTED:** Supabase supports local development and migration-driven deployment; generated diffs can require review/order correction.
**INFERRED:** Dashboard-only schema changes are drift to capture, not ideal durable authority.

### SB-SUPA-015 — Backup scope is narrower than "whole project"
**DOCUMENTED:** Database backups do not restore Storage object bytes; PITR uses physical backups plus WAL; service configuration can require separate reconstruction.
**INFERRED:** Disaster recovery must enumerate DB, Storage, functions, secrets/configuration, integrations, and external effects separately.

### SB-SUPA-016 — RLS is not the whole authorization system
**DOCUMENTED:** Data API access depends on PostgreSQL privileges as well as RLS; functions have `EXECUTE` privileges.
**INFERRED:** Grants and RLS should be designed together.

### SB-SUPA-017 — Permissive policies can widen authority
**DOCUMENTED:** Permissive policies combine with OR; restrictive policies provide mandatory additional constraints.
**INFERRED:** Global deny/fence conditions such as MFA/currentness belong in restrictive logic where appropriate.

### SB-SUPA-018 — Views are an authority-escalation surface
**DOCUMENTED:** View execution context can bypass underlying RLS; PostgreSQL 15+ supports `security_invoker=true`.
**INFERRED:** Every exposed view needs an explicit security model.

### SB-SUPA-019 — SECURITY DEFINER is controlled privilege escalation
**DOCUMENTED:** A security-definer function executes with owner authority and requires careful grants/ownership/search_path handling.
**INFERRED:** Treat each such function like a small privileged service endpoint.

### SB-SUPA-020 — JWT authorization has a currentness window
**DOCUMENTED:** Changed claims are not represented until token refresh.
**INFERRED:** Stable eligibility can live in claims; rapidly revocable authority should be rechecked against current state.

### SB-SUPA-021 — Logout does not make every already-issued JWT instantly invalid everywhere
**DOCUMENTED:** JWTs remain cryptographically verifiable until expiry unless consumers also check server-side session state; high-sensitivity operations can inspect session state.
**INFERRED:** Signature validity proves legitimate issuance, not necessarily present authorization.

### SB-SUPA-022 — Session controls are refresh-bound
**DOCUMENTED:** Session lifetime/inactivity/single-session controls interact with refresh behavior and JWT lifetime.
**INFERRED:** Do not use impractically tiny JWT lifetimes as a substitute for current-authority checks.

### SB-SUPA-023 — Refresh-token replay handling tolerates legitimate races
**DOCUMENTED:** Supabase permits bounded refresh-token reuse patterns to survive unreliable clients/networks while detecting misuse outside those cases.
**INFERRED:** Auth testing must distinguish network retry from malicious replay.

### SB-SUPA-024 — Asymmetric signing is the stronger long-term verification model
**DOCUMENTED:** Supabase's newer signing-key architecture supports JWKS/asymmetric verification and independent key rotation.
**INFERRED:** Prefer verifier-public-key separation over distributing shared signing secrets.

### SB-SUPA-025 — Queues represent pending work; Realtime does not
**DOCUMENTED:** Queue messages persist until explicit delete/archive; visibility timeout enables retry.
**INFERRED:** Durable work state belongs in the queue/database, not WebSocket delivery.

### SB-SUPA-026 — External exactly-once effects still need idempotency
**DOCUMENTED:** A message can be re-read after visibility timeout.
**INFERRED:** The crash window "external effect succeeds -> worker dies -> message reappears" must be handled with an idempotency key/effect receipt.

### SB-SUPA-027 — Queue exposure needs an explicit security model
**DOCUMENTED:** Queue tables are not RLS-protected by default; client exposure uses `pgmq_public` wrappers plus privileges/RLS.
**INFERRED:** Browser-visible queue access should be exceptional.

### SB-SUPA-028 — Realtime acknowledgement is not processing acknowledgement
**DOCUMENTED:** Broadcast acknowledgement proves receipt by the Realtime server, not durable consumer processing.
**INFERRED:** Never derive completion or authoritative execution from a Broadcast ack.

### SB-SUPA-029 — Realtime authorization can be stale during a connection
**DOCUMENTED:** Channel authorization is established/cached for an active connection and changes with token refresh/reconnection behavior.
**INFERRED:** Explicitly test revocation while sockets remain connected.

### SB-SUPA-030 — Postgres Changes is not an infinite event bus
**DOCUMENTED:** Authorization/fanout and ordered processing introduce scaling constraints; Supabase recommends Broadcast for many high-scale realtime cases.
**INFERRED:** Use Postgres Changes for database observation, not as a universal workflow transport.

### SB-SUPA-031 — Edge Functions are workers, not durable state machines
**DOCUMENTED:** Edge Functions have lifecycle/resource limits even when using background execution helpers.
**INFERRED:** Persist intent before invocation and outcome after success.

### SB-SUPA-032 — Region choice affects latency and availability
**DOCUMENTED:** Edge Functions can run close to users or be targeted near a database; explicit regional selection changes failover behavior.
**INFERRED:** Database-intensive authority checks usually benefit from proximity to the primary, with the availability tradeoff documented.

### SB-SUPA-033 — Transaction pooling invalidates persistent-session assumptions
**DOCUMENTED:** Transaction-pooled connections are reused across clients/transactions; session-global state is unsafe as a durable request identity mechanism.
**INFERRED:** Use transaction/request-local identity state.

### SB-SUPA-034 — Schema changes can be operationally hostile
**DOCUMENTED:** DDL can require strong locks; Supabase recommends lock/statement timeouts and care with large changes.
**INFERRED:** Migration testing must include realistic active-load/lock behavior, not just empty preview databases.

### SB-SUPA-035 — Secret/service keys are privileged identities
**DOCUMENTED:** Secret/service-role access is highly privileged and can bypass RLS.
**INFERRED:** Isolate privileged credentials by component where possible; "service role everywhere" destroys attribution and least privilege.

### SB-SUPA-036 — Vault protects storage, not an authorized plaintext reader
**DOCUMENTED:** Vault encrypts secrets at rest and exposes decrypted values through an authorized view.
**INFERRED:** Access to decrypted-secret views is itself a high-privilege boundary.

### SB-SUPA-037 — Logs are evidence, not governing truth
**DOCUMENTED:** Supabase exposes multiple service/database logging surfaces with different retention/coverage.
**INFERRED:** Critical business effects need application-level authoritative receipts, not log-only proof.

### SB-SUPA-038 — Logging can leak authority
**DOCUMENTED:** Request headers/parameters can contain credentials; database/function logging can expose sensitive values if used carelessly.
**INFERRED:** Use canary-secret tests and explicit redaction policies.

### SB-SUPA-039 — Feature maturity is architectural evidence
**DOCUMENTED:** Supabase classifies products/features across GA, beta, public alpha, and private alpha.
**INFERRED:** Critical authority invariants should prefer mature PostgreSQL/GA primitives when equivalent.

### SB-SUPA-040 — MCP/AI access requires production hostility
**DOCUMENTED:** Supabase's MCP guidance warns about production access and prompt injection, and recommends bounded/read-only/dev-oriented use where possible.
**INFERRED:** AI control planes should expose governed application actions rather than raw unrestricted production database authority.

### SB-SUPA-041 — Supabase is a Postgres-centered distributed platform
**DOCUMENTED:** Database, Auth, Storage, Realtime, Edge Functions, Queues, Cron, APIs, replication, and management surfaces compose around PostgreSQL.
**INFERRED:** Reason about which components are inside the DB transaction, which observe committed state, and which maintain separate service state.

### SB-SUPA-042 — Storage has a relational/object atomicity gap
**DOCUMENTED:** Storage metadata participates in PostgreSQL/RLS while object bytes live in object storage.
**INFERRED:** Relational row + object bytes are not one ordinary PostgreSQL commit; use staged states/reconciliation for strong workflows.

### SB-SUPA-043 — PostgREST turns the database into an application-facing security boundary
**DOCUMENTED:** Supabase auto-generates REST over database objects/functions.
**INFERRED:** Schema, grants, RLS, views, and RPC design are application security, not merely database administration.

### SB-SUPA-044 — GraphQL is another interface to the same relational truth
**DOCUMENTED:** pg_graphql exposes PostgreSQL through GraphQL.
**INFERRED:** GraphQL convenience does not remove query-planning, cardinality, permission, or resource costs.

### SB-SUPA-045 — Cron is a scheduler, not a durable workflow engine
**DOCUMENTED:** Supabase Cron is based on pg_cron and is intended for bounded scheduled jobs.
**INFERRED:** Schedule discovery/enqueue operations; keep retry-heavy/long work in durable queues/workers.

### SB-SUPA-046 — Webhooks inherit pg_net's asynchronous boundary
**DOCUMENTED:** Database Webhooks are implemented with triggers + pg_net.
**INFERRED:** They are useful integration notifications, but not a substitute for a durable transactional outbox/queue where loss or duplicate effects matter.

### SB-SUPA-047 — Edge Functions are glue compute
**DOCUMENTED:** Edge Functions provide globally distributed server-side execution with runtime limits and secret access.
**INFERRED:** Prefer them for bounded integration/business logic rather than long-lived workflow custody.

### SB-SUPA-048 — pgvector's strength is relational/vector co-location
**DOCUMENTED:** Embeddings are PostgreSQL vector columns; HNSW/IVFFlat and relational filters can be combined; filtered ANN can require iterative scans/tuning.
**INFERRED:** Supabase Vector is especially strong when permissions/metadata/relational state must constrain retrieval.

### SB-SUPA-049 — PITR RPO and RTO are different
**DOCUMENTED:** PITR uses physical backups + WAL; documented restore-point granularity/RPO does not imply equally short restoration time.
**INFERRED:** Recovery planning must specify both tolerated data loss and service restoration time.

### SB-SUPA-050 — Replica lag is an authority risk
**DOCUMENTED:** Replica lag can return stale data.
**INFERRED:** Do not read rapidly changing authorization/currentness from replicas unless the staleness envelope is explicitly safe.

### SB-SUPA-051 — Logical replication has operational custody costs
**DOCUMENTED:** Publications, slots, subscriptions, WAL retention and replica identity are real PostgreSQL mechanisms with lag/disk implications.
**INFERRED:** Treat CDC consumers as operational dependencies that can create WAL pressure.

### SB-SUPA-052 — Extensions share database resources
**DOCUMENTED:** pgmq, pg_cron, pg_net, pgvector and other extensions execute within/alongside PostgreSQL.
**INFERRED:** Convenience co-location also couples CPU, memory, I/O, locks and WAL pressure.

### SB-SUPA-053 — Supabase does not have one global transaction across all services
**DOCUMENTED:** PostgreSQL transactions cover DB state; Storage bytes, Realtime transport, Edge execution and third-party APIs have separate service boundaries.
**INFERRED:** Critical cross-service workflows require explicit state machines, idempotency and reconciliation.

### SB-SUPA-054 — Scaling remains PostgreSQL-centered
**DOCUMENTED:** API/Realtime/Functions/Storage can scale independently and read replicas scale reads; primary PostgreSQL remains write authority.
**INFERRED:** Supabase is not transparent global multi-primary SQL.

### SB-SUPA-055 — Lock-in is asymmetric
**DOCUMENTED:** Core data/schema use PostgreSQL and many open-source components; hosted service behavior/configuration adds Supabase-specific coupling.
**INFERRED:** Data-model lock-in is relatively low; application/platform lock-in rises with Auth, Storage, Realtime, Edge, Management API and hosted workflows.

### SB-SUPA-056 — Self-hosting is software portability, not automatic cloud parity
**DOCUMENTED:** Many components are self-hostable while some hosted platform capabilities require different infrastructure/configuration.
**INFERRED:** "Open source/self-hosted" does not mean a local Docker stack reproduces every managed-cloud operational guarantee.

### SB-SUPA-057 — Service boundaries do not imply resource independence
**DOCUMENTED:** Many Supabase services ultimately depend on the project database.
**INFERRED:** A pathological database workload can surface as API/Auth/Storage/Realtime symptoms; observability must follow shared dependencies.

### SB-SUPA-058 — Object workflows need explicit lifecycle states
**INFERRED from documented DB/object separation:** states such as PENDING_UPLOAD, UPLOADED, VERIFIED, ACTIVE, FAILED and DELETING make partial failure observable and reconcilable.

## Hostile-test corpus to retain

A future SB should deliberately test:

- cross-tenant SELECT/INSERT/UPDATE/DELETE and missing `WITH CHECK`;
- permissive-policy OR widening and restrictive fences;
- security-definer view/function bypass and `search_path` attacks;
- stale JWTs, stale MFA/claims, logout with still-valid access token, signing-key rotation/cache lag;
- authority revoked after admission but before execution;
- duplicate queue delivery, poison jobs, worker death before effect / after effect / before receipt / before ack;
- Realtime disconnect, missed Broadcast, stale socket authorization, false server-ack interpretation;
- Edge Function termination, CPU/memory/wall-clock limits, regional failure;
- simultaneous approvals, optimistic-version races, lost updates, deadlocks, migration-lock contention;
- transaction-pool contamination/session-state assumptions;
- secret-key leakage, Vault privilege mistakes, bearer-header logging;
- replayed commands, stale target revisions, stale authority revisions, forged/self-authorizing records;
- LLM/MCP prompt injection attempting to turn retrieved content into production-control instructions.

## Known remaining training gaps

Do not call these closed without fresh evidence:

1. Supabase-specific PostgreSQL internals under realistic load: vacuum/autovacuum, planner pathologies, lock/deadlock pressure, partitioning, WAL and disk/IOPS interactions.
2. Deep security adversarial practice across complex RLS, custom roles, column privileges, hooks/custom claims, privileged functions, tenant isolation, key rotation, and least-privilege service identities.
3. Realtime ordering/reconnect/failure semantics and high-fanout behavior under lag/failover.
4. PGMQ poison-message, DLQ, retry, worker-leasing, idempotent external-effect, and recovery patterns validated empirically.
5. Hosted operations/control plane: Management API, CLI, Terraform, branches, backups/restores, project roles, keys/signing-key rotation, network restrictions, and administrative blast radius.
6. Storage failure/reconciliation behavior, signed URLs, resumable uploads, S3 compatibility, CDN invalidation and recovery.
7. Edge runtime concurrency/resource/region behavior under failure.
8. Large-scale pgvector/HNSW/IVFFlat tuning and embedding migration/versioning.
9. Actual hostile experiments rather than documentation-only reasoning.

## Recovery procedure

1. Fresh-check this repository `main`, this checkpoint's branch/PR, issue #24 and any later SB/Supabase training artifacts.
2. Fresh-check current Supabase official documentation before answering unstable/product-state questions.
3. Fresh-check `thebrazenbeard/chat-communication-bus` and the current SB recovery checkpoint/message.
4. Treat this file as a starting snapshot, never current provider truth.
5. Do not require the retired ChatGPT chat, its URL, hidden state, or title.
6. Dispatch research through `SB::RESEARCH` / scoped variants or synthesis through `SB::SYNTHESIZE`.
7. For any production/provider effect, obtain and verify exact current authority separately.

## Exodus claim ceiling

- The worker is reconstructible from durable state: **goal of this checkpoint**.
- The research index is a dated synthesis of official-doc research plus explicit inference: **not a claim of eternal product behavior**.
- Historical proxy bootcamp evidence exists: **yes**.
- Current native SB qualification established by this checkpoint: **no**.
- Production Supabase state changed by this checkpoint: **no**.
