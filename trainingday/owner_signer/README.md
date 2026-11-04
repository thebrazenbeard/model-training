# Tier1 B owner signer — manual VN1 candidate (HOLD)

This is a **source-only, owner-attended candidate**, not an installed cert,
Lappy broker grant, protected model-training authorization, or validated VN1
runtime. It exists separately from the ephemeral GitHub Actions proof in
[PR #321](https://github.com/thebrazenbeard/model-training/pull/321) and the
public-only Lappy verifier boundary in
[PR #318](https://github.com/thebrazenbeard/model-training/pull/318).

## Boundary

- Log in **locally and interactively** to VN1 as `VERA-NODE-1\Patrick`. An
  Executor session running `VERA-NODE-1\Vera`, an administrative session,
  WinRM, an unattended task, and GitHub Actions are not qualified owner sessions.
- Only after exact live status/owner review, run
  `trainingday/owner_signer/New-Tier1BOwnerPublicCertificateVN1.ps1` from
  an independently reviewed source tree in that *Patrick* session. The script
  requires typed owner consent before creating a permanent, nonexportable CNG
  signing key. Do not send an account password or protection UI PIN to Executor.
- It refuses an existing owner certificate subject or public output path rather
  than silently replacing/reusing the owner identity.
- It requests **Microsoft Software Key Storage Provider**, RSA3072,
  SHA256-RSA, CNG signing-only, **NonExportable**, **ProtectHigh**, 120-day
  validity, `DigitalSignature` only, `CA=false`, and a single **critical**
  purpose `2.25.281124513629607016020108740092472226515`.
- The EKU is encoded as raw ASN.1 hex because the Windows certificate API's
  `{text}` OID parser silently truncated the full 128-bit `2.25` arc in the
  first hosted CI attempt. The corrected DER `301606146983A6FECEA99ABA92BCAF8AA4B9BB9BEBD9E553`
  passed the subsequent hosted Windows-runner proof. Never replace it with
  textual EKU syntax without a new native proof.
- It validates exact certificate shape, provider/export/UI policy, and a
  local random sign/verify challenge. It writes only a public DER `.cer`
  under Patrick's `%LOCALAPPDATA%\Vera\TrainingDay`, with no clobber.
- The private key is not exported and must stay under Patrick's
  `Cert:\CurrentUser\My`. The public DER fingerprint must be independently
  recorded/verified before **any future separately authorized** Lappy
  machine-store public import or verifier pin. Importing a PFX or any private
  key to Lappy is forbidden.
- If creation succeeds but later checks fail, the script leaves the newly
  nonexportable key/cert for manual owner inspection. It does not automatically
  destroy unique signer material or attempt a replacement.

## Evidence and limits

The hosted CI job on this branch parses the script and exercises the actual
wrong-host refusal **without creating a key**. PR #321 independently proves
the DER EKU/algorithm/public-only export shape on an ephemeral Windows runner;
it does **not** prove `ProtectHigh` UI behavior or the owner profile on VN1.

Before Patrick runs anything, verify Windows version, PowerShell PKIClient,
console interactivity and original owner identity on VN1; `ProtectHigh` support
and UI prompts there remain **UNKNOWN**. Stop if an expected prompt fails.
There is no automatic trust enrollment, HostBroker install/restart, signed
launch grant, Qwen optimizer step, GPU lease or NAS scope in this artifact.

Primary references:

- [Microsoft PKIClient New-SelfSignedCertificate](https://learn.microsoft.com/en-us/powershell/module/pki/new-selfsignedcertificate?view=windowsserver2025-ps)
- [Microsoft PKIClient Export-Certificate](https://learn.microsoft.com/en-us/powershell/module/pki/export-certificate?view=windowsserver2022-ps)
- [Microsoft CngKey UIPolicy](https://learn.microsoft.com/en-us/dotnet/api/system.security.cryptography.cngkey.uipolicy)
- [Microsoft CngUIProtectionLevels](https://learn.microsoft.com/en-us/dotnet/api/system.security.cryptography.cnguiprotectionlevels)
