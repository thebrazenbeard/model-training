# TRAININGDAY Contract B — host-derived input evidence (source candidate)

**STATUS: HOLD. This does not install a host broker or grant permission to run an optimizer.**

The existing native `HostOwnedProbeReservation` fixture proves that a Windows
file lock, flushed append, monotonic epoch, nonce consumption, and unresolved
post-crash reservation can be exercised on a native Windows test runner. It is
not itself protected against a privileged journal rollback or authenticated
for the actual Lappy service.

`Tier1BHostEvidenceHold` adds one bounded native integration experiment:

- Hash **actual bytes** from three fixed names under a unique *test-only* root:
  `plan.json`, `manifest.json`, and `kernel.bin`. No request-provided digest can
  replace those observations. Each bounded file handle remains held exclusively
  until the fixture is disposed, and its bytes can be rechecked through the
  same handle before any hypothetical future effect.
- Generate a 256-bit nonce using the OS CSPRNG, hash it, and show a
  `HostOwnedProbeReservation` can durably reserve the nonce and actual plan
  hash with a host-owned epoch. Reopening an unfinished reservation remains
  a hard HOLD. A new nonce cannot silently clear an unresolved reservation.
- Reject relative fixture roots, missing/reparse or oversized fixture files,
  and concurrent writes to held inputs. A change after release is detected
  when the host opens and hashes the new bytes.
- A red-first Windows regression demonstrated that a final-directory reparse
  check misses a junction in an **ancestor**. The candidate now rejects an
  ancestor junction, UNC/device roots, nonfixed drives, and oversized plans.
  This remains **pre-open checking, not post-open kernel file-ID or ACL
  attestation**: a malicious administrator could race path resolution, swap
  content using privileged filesystem access, or roll back the journal.
- Prove the remaining rollback counterexample in an isolated fixture: deleting
  the journal and allowing `OpenOrCreate` resets its epoch to zero and lets an
  old nonce be reserved again. A live Tier1 journal must require an
  owner-provisioned existing protected file and independently anchored
  monotonic state. The source explicitly reports rollback resistance false.

The exact source is an **inert test helper**, not integrated into the running
`VeraTrainingDayHostBroker` or `Tier1BGrantCandidate` in draft PR #309. The
caller-selectable fixture root and its journal must **never** be used as a
live host trust root. The future production host must use a fixed protected
directory and verified Windows ACL, file identity and open-handle lifetime,
with an authenticated owner signing certificate (public cert on broker,
private key retained outside broker trust), cryptographic signature, atomic
grant admission, and protected durable monotonic anti-replay journal. A local
plain hash chain does **not** defeat rollback by an administrator; external
replay anchoring or stronger independent trusted monotonicity remains UNKNOWN.

The test also does **not** authenticate the full official Qwen3.5-4B snapshot
(four metadata byte pins are still UNKNOWN; physical shards are not locally
hashed), enforce an actual WDDM GPU lease, establish targeted WSL2 Linux child
custody, prove 4 GiB fit, or authorize Tier2 R201/DEV/NAS Q6. Every output
remains HOLD, with authorization flags false. No positive service request
handler, launch command, model download, private key or protected data exists.

## Evidence boundary

The repository's new Windows workflow compiles exact checked-out source with
.NET Framework 4.8 `csc.exe` and runs the harmless native file/journal negative
test. Green CI proves only this synthetic Windows-hosted fixture. The actual
owner host, service identity, ACLs, signer custody, crash restart, real WSL2,
model bytes, driver resources and optimizer remain independently unqualified.
