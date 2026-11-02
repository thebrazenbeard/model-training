# TrainingDay Windows host broker: missing authority boundary

Status: **DESIGN / BLOCKED — NOT A LAUNCH GRANT**. This is a source-only
deployment contract. No Windows service, NAS replay witness, GPU reservation,
rights authorization, or physical training authority is installed by this file.

## Existing source contracts

- `physical_launch_authority.py` rejects every physical launch, even with
  caller-provided environment flags. Keep this unconditional hold until a
  separately reviewed host-native implementation owns the entire launch.
- `r140_host_custody_service.py` specifies reservation, immutable snapshot,
  commit, and abandon, but explicitly cannot authenticate its injected backend
  or establish an independent replay witness.
- `r140_windows_custody_contract.py` compares untrusted observations; it is
  deliberately incapable of issuing a usable filesystem capability.
- `r140_monotonic_authority.py` models transitions with an in-memory witness,
  not a cross-process, antirollback NAS authority.
- The independent R144 host-launch declaration in Draft #184 supplies source,
  rights, plan, nonce, expiry, lease, and witness *claims*. Its canonical hash
  is useful as a request identifier, never proof that those claims are true.

Adding a Python verifier that accepts a signed-looking response from a
caller-supplied backend would create a new source-controlled authorization
shortcut. The next implementation belongs to a Windows service installed and
administered **outside** the Git checkout and Python caller process.

## Minimal service ownership contract

The externally installed broker is the **only** process that can grant and
execute a physical training operation. The Python client can submit a
declaration and display a diagnostic receipt; it must not acquire a new
`can_launch`, `authorized`, or equivalent launch decision.

1. **Protected local IPC endpoint.** Install a named-pipe or local-RPC server
   under a dedicated Windows service identity, with a restrictive DACL,
   local-only access, and exclusive server-instance creation. Clients must
   verify that they connected to the expected installed Windows service,
   not a same-name pipe hosted by an untrusted process. A string containing
   a SID, process ID, service name, or host name is insufficient.
2. **Real peer/process identity.** For each message, the server authenticates
   the connection with Windows token-based identity (for example, successful
   named-pipe client impersonation followed by token inspection) and holds
   kernel process handles while checking PID and creation-time bindings.
   Failed impersonation or a low/anonymous identification level rejects the
   request; a claimed `caller_principal` must equal the observed identity.
   Capture and pin the permitted executable identity as an independently
   approved installation policy, not a path/string from the request.
3. **Exact, immutable workload.** Recompute the canonical declaration SHA-256
   and verify against an administrator-approved policy: source commit/tree,
   corpus manifest and producer, exact selected training data hash,
   independently adjudicated rights for that exact data, official Qwen
   model/revision/hash, R15/R16 mode and full run-plan parameters, and zero
   arm/control digest. All file access must use broker-held stable handles
   or immutable protected snapshots: no check-then-open user pathnames.
4. **Exclusive GPU ownership.** Broker atomically reserves the selected GPU
   and host route under a persistent, host-owned lease ID and strictly
   increasing epoch. It must independently reconcile competing GPU processes
   and reservations; a named mutex or client-provided `lease_id` is not a
   hardware-exclusivity proof. Revalidate the lease immediately before
   dispatch and continuously enforce process lifecycle and cleanup.
5. **Durable anti-replay / deadline.** Allocate a broker-side unique request
   and nonce record, tied to the authenticated caller, host boot instance,
   exact request hash, GPU lease epoch, and expiration under trusted host
   time. Atomically consume each nonce before any irreversible launch effect;
   reject duplicates, expiry, clock regressions, reboots without renewed
   authority, and ambiguously committed retries. Do not let client clocks or
   nonce claims substitute for server custody.
6. **Independent NAS witness.** An independently administered NAS authority
   must durably compare-and-append the request hash, lease epoch, nonce,
   authorized rights decision, and dispatch transition, with a prior-head
   check and authenticated service identity. The broker must verify its
   response from that authority (not a caller-provided `witness_head`).
   Witness timeout, rollback, split brain, or uncertain commit yields HOLD
   or UNKNOWN and must not trigger an automatic replay.
7. **Broker-owned launch and outcome.** Only after the above transaction is
   durably committed may the broker create a pinned executable/process in a
   Windows Job object with appropriate termination-on-owner-loss policy.
   The broker retains the run process, GPU route, and snapshot handles,
   records observable lifecycle transitions, and releases resources only
   after a witnessed terminal state. Client disconnect or source updates do
   not delegate permission to Python or silently free ambiguous leases.
8. **Fail closed.** Missing, unreachable, stale, unauthenticated, version-
   incompatible, or partially upgraded service/NAS means HOLD. No fallback
   to source `Protocol` objects, caller-provided JSON, environment variables,
   self-signed receipts, or boolean switches. Failures must not open data,
   load weights, initialize CUDA, or dispatch workers.

## Proposed protocol (for host-side design review, not callable here)

*Request*: version; canonical immutable launch declaration bytes; declared
SHA-256; client challenge nonce (not an authority nonce); opaque correlation
identifier. Server obtains peer identity and authoritative current policy
itself. Do not add client-supplied `approved=true` fields.

*Response*: protocol version; correlation identifier; request hash; broker
instance and boot epoch; authoritative observed caller/process binding;
reservation ID and GPU lease epoch; host-owned dispatch ID/job identity;
NAS witness epoch/head and durable receipt reference; state (`HOLD`,
`UNKNOWN`, `DISPATCHED`, `TERMINAL`); reason code; broker-authenticated response
binding. A client may display this response but **must never use its state
to unlock the current source launcher**. The broker launches, not the client.

The protocol must specify exactly-once transaction recovery for every crash
point between reservation, witness append, job creation, and job observation;
`UNKNOWN` forbids blind replay. Both endpoints need independent protocol and
installed-binary version validation. Signatures or message MACs are only
meaningful when their keys and verification root are unavailable to the
caller; copying keys into the checkout is prohibited.

## Required hostile validation before host deployment

| Test | Mandatory behavior |
| --- | --- |
| Service absent, wrong instance, fake pipe server, remote pipe | HOLD before host/weight access |
| Anonymous, low impersonation, false PID/SID, PID reused, unauthorized image | HOLD; audit actual token/process |
| Changed source tree, data hash, base revision, right decision, any run-plan field | HOLD; no partial matching |
| Unknown rights owner or unsigned/unreviewed rights decision | HOLD irrespective of request fields |
| Duplicate/reordered/expired nonce, host clock rollback, service reboot | HOLD or UNKNOWN; never second dispatch |
| Two concurrent applicants, lease epoch ABA, foreign GPU process | At most one physically owned launch; no optimistic mutex-only proof |
| NAS offline/timeout, forged witness, stale CAS head, split-brain reply | HOLD or UNKNOWN; no launch/replay |
| Symlink/hardlink/reparse swap, rename during verification, changed file after hash | Immutable handle mismatch rejects |
| Crash after nonce consume, after NAS append, during CreateProcess, before job attach | Auditable UNKNOWN / deterministic reconciliation; no duplicate launch |
| False `DISPATCHED`/signed-looking JSON injected into Python test backend | `physical_launch_authority.py` still rejects all physical launches |

Acceptance requires independent Windows host integration tests with real
service ACL/token impersonation checks, NAS consistency failure injection,
and host-observed GPU exclusivity, followed by an owner-approved rights and
model-data inspection. Linux CPU unit tests can validate syntax and negative
cases only; **they cannot establish Windows host ownership**.

## Reference API behavior

Microsoft documents named-pipe security descriptors, impersonation failure
hazards, and Job objects; implementation review should use those primary
references:

- https://learn.microsoft.com/en-us/windows/win32/ipc/named-pipe-security-and-access-rights
- https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-impersonatenamedpipeclient
- https://learn.microsoft.com/en-us/windows/win32/ipc/named-pipes
- https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
