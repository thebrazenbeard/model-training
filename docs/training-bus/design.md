# Training Bus Terminal Design

Date: 2026-08-12
Status: USER-APPROVED DESIGN BASELINE
Scope: Native-function training transport between external training workbenches and native ChatGPT specialist identities

## Objective

Provide a reusable, release-pinned transport layer that carries an independently audited training package from an external training workbench to the correct native ChatGPT specialist identity without making the finished training system depend on GitHub, Qwen, APIs, or other persistent external infrastructure.

The system transports the package, not the chat. A target identity such as SB only enters the process when it consumes an admitted package in its native ChatGPT context and undergoes native qualification.

## Architecture

The canonical route is:

`Training School -> Audit Depot -> Native Cargo Store -> Project Terminal -> Named Stop -> Native Specialist -> Qualification`

The preferred Native Cargo Store is ChatGPT Library when available.

### Training School

A temporary external workbench may gather primary sources, generate curricula, run proxy practice, perform adversarial testing, and produce a candidate release. The current GitHub/Qwen bootcamp is one implementation of this role.

The school has no authority to mark a native ChatGPT identity qualified.

### Audit Depot

Candidate output is independently reviewed before admission. The depot verifies source provenance, release manifest, checksums, scope claims, known failures, qualification material, and regression provenance.

Admission produces an immutable admission receipt bound to the exact release fingerprint. The receipt records at minimum the release ID, intended stop/function, package fingerprint, audit outcome, and supersession/currentness relationship known at admission time.

Only an admitted release with a matching admission receipt may be routed to a native specialist.

### Boarding Package

Each admitted release is immutable and self-contained. A package contains at minimum:

- release ID
- intended stop / function
- manifest and package fingerprint
- audit admission receipt or its exact embedded representation
- identity capsule
- exact source anchors and provenance
- operating invariants and known failure modes
- remaining gaps
- native qualification profile or test set
- regression candidates with epistemic status
- supersession / rollback metadata

A package must not require GitHub or another external service to interpret its authoritative contents after admission.

The package does not become authoritative merely because it exists. Boarding requires the Project Terminal to name the same admitted release and expected fingerprint, and the package must contain the matching audit admission receipt.

### Native Cargo Store

The preferred cargo store is ChatGPT Library when available because it allows admitted package files to be reused natively in later chats.

Cargo storage is not authority. The immutable package contents, admission receipt, and fingerprint establish package identity, while the Project Terminal identifies which admitted release is current for a stop.

Automatic assistant-side Library retrieval is an optimization only, because universal autonomous retrieval by every future specialist chat is not established as a guaranteed product contract.

### Project Terminal

One small Project-native source acts as the route board and stop registry, provisionally named `TRAINING_BUS_TERMINAL.md`.

It contains routing metadata only, not bulky training knowledge. For each named stop it records:

- stop ID
- function
- current admitted release ID
- expected package filename or logical identity
- expected package fingerprint
- expected admission-receipt identity or fingerprint
- qualification profile
- status
- immediately previous admitted release for rollback when available

The terminal is the native current-route registry, not a substitute for the audit receipt. A terminal entry cannot make an unadmitted package authoritative. If the terminal, admission receipt, and package identity do not agree exactly, boarding fails closed until currentness is resolved.

### Named Stops

Stops are primarily declarative configuration, not bespoke scripts. Example stop IDs include `SB`, `SECURITY`, and `PROJECT_ENGINEER`.

A stop answers: which function is this destination, which admitted release is current, which exact package bytes are expected, and which native qualification profile applies?

Special stop handlers are permitted only when the generic boarding contract cannot express legitimate specialist-specific behavior.

### Generic Bus Protocol

The bus is generic orchestration logic. It performs the same bounded sequence for every stop:

1. Resolve the named stop.
2. Resolve the exact current admitted release.
3. Verify release ID, intended stop, manifest, fingerprint, admission receipt, admission status, and currentness.
4. Reject stale, wrong-passenger, malformed, unadmitted, or mismatched packages.
5. Make the exact admitted package available to the native specialist.
6. Require the specialist to consume the package without promoting proxy scores or model-authored claims beyond their stated epistemic status.
7. Run native cold and transfer qualification against fresh tasks.
8. Record PASS, CONDITIONAL PASS, or FAIL for that native identity and release.
9. On failure, route to remediation rather than silently promoting the release.

A canonical boarding command may take the form `BUS::BOARD::<STOP_ID>`.

## Boarding Modes

### Automatic Boarding

If the native runtime exposes reliable Library search/retrieval, the specialist may resolve the exact package from the terminal and consume it directly.

Automatic pickup is never required for correctness.

### Manual Fallback

If autonomous retrieval is unavailable, the specialist reports the exact required package identity and fingerprint. The user attaches that exact package from ChatGPT Library when available, or uploads the exact admitted package file from another user-controlled copy. Boarding then continues from the same verification step.

The fallback therefore depends on possession of the exact fingerprinted package, not on a particular ChatGPT UI feature.

## Authority and Trust Rules

- External training evidence is not native qualification.
- Same-model trainer/student/examiner scores are proxy evidence only.
- The route board cannot make an unadmitted package authoritative.
- A package cannot self-authorize merely because it exists in Library or another cargo store.
- Audit admission is bound to exact package bytes through the admission receipt and fingerprint.
- `latest` is not a valid release selector for boarding; boarding is release-pinned.
- Wrong-stop packages fail closed.
- Fingerprint, manifest, or admission-receipt mismatch fails closed.
- Superseded releases do not silently override the terminal's current admitted release.
- Rollback is explicit and may target only a previously admitted release.
- GitHub, Qwen, and other workbench infrastructure disappear from the native runtime dependency graph after admission.

## State Model

Recommended release / route states:

`CANDIDATE -> AUDIT_PENDING -> ADMITTED -> READY_FOR_BOARDING -> NATIVE_QUALIFICATION -> QUALIFIED`

Failure and maintenance states:

`REJECTED`, `REMEDIATION_REQUIRED`, `SUPERSEDED`, `ROLLBACK_ELIGIBLE`, `RETIRED`, `CONFLICTED`.

Only `ADMITTED` or `READY_FOR_BOARDING` packages may be presented for initial native boarding. Only a native qualification result may produce `QUALIFIED` for a target identity.

## Rollback

Each stop should expose the current admitted release and, when available, the immediately previous admitted release. Rollback is explicit and fingerprint-pinned. Keeping unlimited historical releases in the terminal is unnecessary; historical packages may remain in Library or user-controlled archival storage without cluttering native routing context.

## Error Handling

Boarding fails closed on:

- missing stop
- missing release
- missing or invalid admission receipt
- unadmitted release
- stale terminal entry
- wrong intended identity/function
- package fingerprint mismatch
- malformed manifest
- conflicting currentness evidence
- absent required native qualification profile

Failure must identify the narrowest blocking condition and must not substitute an older or newer release automatically.

## Testing Strategy

The transport layer requires deterministic tests for:

- correct stop -> correct release resolution
- wrong passenger rejection
- stale route-board rejection
- altered package / checksum rejection
- forged, missing, or mismatched admission receipt rejection
- unadmitted package rejection
- current + rollback resolution
- automatic retrieval success path
- manual attachment/upload fallback path
- no external-runtime dependency after package admission
- native qualification remains separate from proxy qualification
- explicit remediation after qualification failure

A future end-to-end acceptance test should use one real specialist stop, initially SB, and prove the complete route from admitted package through native cold/transfer qualification without relying on project-chat recollection as authority.

## Initial Physical Layout

Conceptual external repository layout:

```text
school_bus/
  bus.py
  stops/
    sb.yaml
    security.yaml
    project_engineer.yaml
  handlers/
    default.py
  releases/
    <RELEASE_ID>/
      MANIFEST.json
      ADMISSION_RECEIPT.json
      IDENTITY_CAPSULE.md
      SOURCE_ANCHORS.json
      QUALIFICATION_SET.json
      REGRESSION_SET.json
```

Native Project surface:

```text
TRAINING_BUS_TERMINAL.md
```

Preferred native cargo surface:

```text
BUS_<FUNCTION>_<RELEASE>.md   # or an equivalent immutable package artifact
```

The concrete package format may be one file or a compact bundle. The design requirement is self-contained immutable identity and admission evidence, not a particular serialization format or storage UI.

## Explicit Non-Goals

This design does not:

- modify ChatGPT model weights
- claim that proxy training directly trains a specialist chat
- allow an external model to grant native qualification
- require a permanent GitHub service at runtime
- assume one custom Python script per specialist
- assume autonomous cross-chat message injection
- assume that every specialist can always retrieve arbitrary Library files without user attachment
- make ChatGPT Library mandatory for correctness

## Success Criteria

The design succeeds when a new specialist can be added primarily by declaring a stop and admitting a package, not by cloning transport logic; the exact package routed to that specialist is verifiable and current; the specialist can board through automatic native retrieval or bounded manual attachment/upload; and native qualification remains the only evidence that the actual target identity successfully absorbed the training release.
