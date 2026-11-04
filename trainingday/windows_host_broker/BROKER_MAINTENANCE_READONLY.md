# Contract B — broker maintenance preflight (source-only)

This is an **inspection-only** procedure. Neither the script nor the hosted
Windows workflow installs, signs, imports, rotates, enrolls, restarts, backs
up, removes, or grants anything on Lappy. The installed LocalSystem service
is not the signed Contract B broker; its currently observed file is unsigned.

## Evidence and review boundaries

1. A Windows CI runner compiles the deny-only source from the exact checked-out
   commit and prints the Git tree, source SHA-256 digests and candidate binary
   SHA-256. The installed broker is **not** thereby rebuilt or attested.
2. The read-only script checks candidate hash and Authenticode signer pin,
   service identity, installed SHA against a separately observed baseline,
   installation-directory write ACL, available GPU memory, and a conservative
   inventory of processes whose disruption could affect other work.
3. It returns `maintenance_state: HOLD`, even when no individual blocker is
   detected. No result is installation approval, a signed owner grant, an
   enforceable WDDM memory cap, or WSL child-custody qualification.
4. The signing **private** key must stay owner-controlled, outside the
   LocalSystem broker's machine certificate store. An owner-signed PFX must
   not be imported into `LocalMachine\My`; a broker may ultimately hold only
   separately pinned public DER verification material.

## Reproducible inspection request

After a separate owner-controlled signature and independent fingerprint
verification, an administrator may *read* a candidate using PowerShell:

```powershell
$knownInstalledSha = '<64-hex SHA256 read from installed image in the same window>'
$independentlyApprovedSigner = '<40-hex thumbprint, verified out-of-band>'
$candidateSha = '<64-hex SHA256 of candidate build>'
./trainingday/windows_host_broker/Assert-BrokerMaintenancePreflight.ps1 `
  -CandidateExecutable 'D:\VERA\trainingday-staging\VeraTrainingDayHostBroker.exe' `
  -ExpectedCandidateSha256 $candidateSha `
  -ExpectedInstalledSha256 $knownInstalledSha `
  -ExpectedSignerThumbprint $independentlyApprovedSigner
```

The example path is a **proposed staging location**, not proof that it exists.
Do not put real keys or certificates in the repository.

## Protected installation and rollback (not automated)

Before an actual replacement, a human owner must inspect the exact compiled
binary, verified signer, service ImagePath, restricted ACLs, source HEAD,
active WSL and GPU processes, startup mode, independent owner key custody,
retained old binary and hash, and a dated rollback plan. A maintenance window
must be separately confirmed *at execution time*. Any missing or changed
evidence is HOLD. An operator must explicitly approve service interruption
and ensure active user workloads will not be terminated.

Restore is not safe simply because the old EXE exists: it requires preserved
service configuration, original binary, restrictive ACL, boot semantics,
fresh status readback and recovery checks. No silent or automatic rollback is
implemented. Retaining an unsigned old binary is historical evidence, **not**
endorsement of redeploying it.

Physical Qwen3.5-4B training stays blocked by host-derived grant authority,
authenticated model files, replay journal/NAS rights, GPU capacity, and WSL2
orphan cleanup qualification. Tier2 R201 and protected DEV remain excluded.
