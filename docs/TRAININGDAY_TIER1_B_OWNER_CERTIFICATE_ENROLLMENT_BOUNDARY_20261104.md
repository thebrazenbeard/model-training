# TRAININGDAY Tier1 Contract B — owner certificate separation (source candidate)

**Status:** SOURCE-ONLY / NOT ENROLLED / NO LAUNCH. Applies only to Lappy,
official `Qwen/Qwen3.5-4B`, public synthetic Tier1 Contract B. Protected
R201/DEV/NAS Q6 remains outside this scope. This runbook is a proposed manual
maintenance procedure, **not** an instruction to execute it unattended.

## Two different trust domains

| Boundary | Allowed certificate material | Forbidden |
| --- | --- | --- |
| Patrick's independently controlled signer (`CurrentUser\\My` or owner hardware key) | A non-exportable RSA signing **private key**, paired public X.509 certificate | Private key export, Git commits, world-readable paths, service-account custody |
| Lappy broker verifier (`LocalMachine\\My`) | The **public certificate DER only** with exact SHA-256 of its full DER pinned in the separately reviewed broker binary | `.pfx`/`.p12`, any machine-store private key, runtime-selected fingerprints, request-supplied signer, broad CA trust |

The private signing key must **not** be present in the Windows service's
LocalSystem identity or `LocalMachine\\My` store. A certificate with
`HasPrivateKey == true` in the machine verifier store is now a hard source
candidate `HOLD` (`OWNER_PRIVATE_KEY_IN_MACHINE_STORE`), even if its DER matches.
The native test uses an in-memory certificate carrying a private key and a
separate public-only DER copy to exercise both branches without host enrollment.

## Gate before touching Lappy

1. Obtain a fresh `STATUS_V1` readback, service `ImagePath`/hash/signature,
   certificate-store inventory **without** exporting any secrets, and an owner
   identity check. The current installed broker may be unsigned; do not silently
   replace or restart it.
2. Confirm a controlled owner-attended maintenance window and a rollback
   snapshot of existing service binary/configuration and its exact ACLs. If
   Blender, PreActive, Edge GPU work, or WSL jobs remain active, defer any
   restart or GPU experiment. Contract B permits coexistence in principle;
   it does not imply adequate VRAM or grant authority to interrupt users.
3. Verify that a *newly generated* owner key is non-exportable, independently
   user-/hardware-controlled, RSA 3072 or 4096, SHA-256 signing capable, with
   `DigitalSignature` KeyUsage, a non-CA basic constraint, and a short explicit
   expiry. Do not reuse an existing private key on speculation. If the owner
   needs to sign off-host, keep the private key off-host entirely.
4. Export **only public DER** from that signer after independent inspection;
   compute the full DER SHA-256 out-of-band and record its provenance and owner
   approval in a protected deployment receipt. DER fingerprint is public; it
   is not a bearer secret.
5. Import **only the DER `.cer`** into `Cert:\\LocalMachine\\My` using an
   owner-attended elevated session. Read it back and verify exact DER digest,
   certificate validity, key algorithm/size, `DigitalSignature`, non-CA, and
   `HasPrivateKey == false`. A PFX import is prohibited. Do not rely on a
   display-name or thumbprint alone.
6. Independently review a candidate broker build with the **compiled** DER
   SHA-256 anchor, SHA-256 of every source/build input, Authenticode signer,
   protected installed binary path, unchanged ACLs, and exact service identity.
   Only a separately authorized, rollback-capable maintenance deployment may
   replace the installed service. The current source literal is `UNENROLLED`.
7. Recheck the broker's local authenticated `STATUS_V1` and enforce unconditional
   `HOLD`. No certificate by itself may make a launch possible. A positive Tier1
   path needs separate protected replay custody, verified model/tokenizer
   bytes, host-derived immutable plan hashes, actual live GPU headroom, WSL2
   descendant termination across broker crash, and independent native host
   qualification before **any** training run.

## Stop conditions and rollback

- Any store with a private key where only a verifier copy belongs, missing
  expected DER, key shorter than RSA 2048, invalid time or usage, untrusted
  operator/session, mismatched binary hash, absent protected rollback, or
  live shared workloads => **STOP**. Keep existing deny-only broker running.
- Preserve the pre-maintenance service executable, metadata, ACLs, and full
  transcript of operator decisions. Restore only the exact previously verified
  binary if an authorized deployment fails. Do not uninstall unrelated certs,
  rotate signer keys, terminate WSL distros, or clear training journals.
- Enrollment must not be construed as a GPU lease, protected corpus admission,
  NAS grant, evaluator authority, or a checkpoint. No positive run is authorized
  by this source-only change.

**Evidence limit:** This document and the hosted native test can prove a
source-level rejection of the private-in-machine-store mistake. They cannot
prove the actual Lappy certificate stores, signing key non-exportability,
compiled service trust, WSL process custody, or device fit.
