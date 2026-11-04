# MANUAL, OWNER-ATTENDED ENROLLMENT CANDIDATE — NOT HOST QUALIFICATION.
# Run locally in an interactive PowerShell session signed in as
# VERA-NODE-1\Patrick on VN1. Do not run via Executor, WinRM, a service,
# scheduled task, GitHub Actions, or the VERA-NODE-1\Vera account.
# This creates a private signing key *only* under Patrick's CurrentUser\My;
# its only file output is the PUBLIC certificate DER (.cer). No PFX/export
# of the private key, no LocalMachine store edits, and no Lappy installation.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedMachine = 'VERA-NODE-1'
$expectedOwner = 'VERA-NODE-1\Patrick'
$purposeOid = '2.25.281124513629607016020108740092472226515'
# DO NOT change to {text}<OID>: hosted Windows PKIClient silently truncated
# this 128-bit 2.25 OID to 2.25.9223372036854775807. This is an ASN.1
# SEQUENCE OF OID, encoded with the entire intended base-128 arc.
$purposeEkuDerHex = '301606146983A6FECEA99ABA92BCAF8AA4B9BB9BEBD9E553'
$subject = 'CN=VERA TrainingDay Tier1 B Owner Patrick v1'
$store = 'Cert:\CurrentUser\My'

# This validation deliberately comes BEFORE any filesystem/cert-store write
# and cannot be bypassed with a script parameter.
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if ($env:GITHUB_ACTIONS -eq 'true' -or
    [Environment]::MachineName -cne $expectedMachine -or
    $identity.Name -ine $expectedOwner -or
    -not [Environment]::UserInteractive -or
    [Console]::IsInputRedirected -or
    [Console]::IsOutputRedirected -or
    (Get-Process -Id $PID).SessionId -le 0) {
    throw ('HOLD: owner-attended VN1 local console required as {0}; this process is {1} on {2}' -f
        $expectedOwner, $identity.Name, [Environment]::MachineName)
}

$cmd = Get-Command -Name New-SelfSignedCertificate -CommandType Cmdlet -ErrorAction Stop
foreach ($requiredParameter in @('KeyProtection', 'KeyExportPolicy', 'KeyUsageProperty', 'TextExtension')) {
    if (-not $cmd.Parameters.ContainsKey($requiredParameter)) {
        throw ('HOLD: this Windows PKIClient does not support required {0}' -f $requiredParameter)
    }
}

$directory = Join-Path $env:LOCALAPPDATA 'Vera\TrainingDay'
$publicDerPath = Join-Path $directory 'Tier1B-Patrick-Owner-PUBLIC.cer'
if (Test-Path -LiteralPath $publicDerPath) {
    throw ('HOLD: existing public DER must be independently reconciled; no overwrite: {0}' -f $publicDerPath)
}
$before = @(Get-ChildItem -Path $store -ErrorAction Stop |
    Where-Object { $_.Subject -ceq $subject })
if ($before.Count -ne 0) {
    throw 'HOLD: exact owner subject already exists in CurrentUser\My; do not create a second owner key'
}

Write-Host 'STOP/REVIEW: this is a one-time nonexportable private signing key creation on VN1.'
Write-Host "Owner: $expectedOwner; certificate subject: $subject"
Write-Host 'KSP: Microsoft Software Key Storage Provider; RSA3072; SHA256; ProtectHigh; valid 120 days.'
Write-Host 'Private signing material stays in Patrick CurrentUser; only a public .cer is written.'
Write-Host 'No Lappy import, installation, service restart, grant, or model training follows.'
$answer = Read-Host 'To create this new key type EXACTLY: ENROLL VN1 PATRICK TIER1B'
if ($answer -cne 'ENROLL VN1 PATRICK TIER1B') { throw 'HOLD: owner declined enrollment' }

# Create only inside this owner's LOCALAPPDATA, never under shared/public paths.
if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
    $null = New-Item -Path $directory -ItemType Directory -ErrorAction Stop
}
if (Test-Path -LiteralPath $publicDerPath) {
    throw 'HOLD: public output appeared during confirmation; refusing overwrite'
}

$issued = $null
try {
    $notBefore = [datetime]::UtcNow.AddMinutes(-2)
    $notAfter = [datetime]::UtcNow.AddDays(120)
    # If ProtectHigh is unsupported or prompts are unavailable, fail closed;
    # never retry with Protect/None, change KSP, or use exportable policy.
    $issued = New-SelfSignedCertificate -Type Custom -Subject $subject `
        -CertStoreLocation $store -Provider 'Microsoft Software Key Storage Provider' `
        -KeyAlgorithm RSA -KeyLength 3072 -HashAlgorithm SHA256 `
        -KeyExportPolicy NonExportable -KeyProtection ProtectHigh `
        -KeyUsage DigitalSignature -KeyUsageProperty Sign `
        -TextExtension @(
            ('2.5.29.37={critical}{hex}' + $purposeEkuDerHex),
            '2.5.29.19={text}'
        ) -NotBefore $notBefore -NotAfter $notAfter -ErrorAction Stop

    if (-not $issued -or $issued.Subject -cne $subject -or $issued.Issuer -cne $subject -or
        -not $issued.HasPrivateKey -or $issued.SignatureAlgorithm.Value -ne '1.2.840.113549.1.1.11') {
        throw 'HOLD: certificate subject/issuer/private key/SHA256-RSA signature mismatch'
    }
    $after = @(Get-ChildItem -Path $store |
        Where-Object { $_.Subject -ceq $subject })
    if ($after.Count -ne 1 -or $after[0].Thumbprint -ine $issued.Thumbprint) {
        throw 'HOLD: owner certificate store has ambiguous/mismatched subject'
    }
    if ($issued.PublicKey.Oid.Value -ne '1.2.840.113549.1.1.1') { throw 'HOLD: public key is not RSA' }
    $privateRsa = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($issued)
    try {
        if (-not $privateRsa -or $privateRsa -isnot [Security.Cryptography.RSACng] -or
            $privateRsa.KeySize -ne 3072) { throw 'HOLD: private key not 3072-bit CNG RSA' }
        if ($privateRsa.Key.Provider.Provider -cne 'Microsoft Software Key Storage Provider') {
            throw 'HOLD: signer key is not in Microsoft software KSP'
        }
        if ([int]$privateRsa.Key.ExportPolicy -ne 0) { throw 'HOLD: private key unexpectedly exportable' }
        if ([int]$privateRsa.Key.KeyUsage -ne [int][Security.Cryptography.CngKeyUsages]::Signing) {
            throw 'HOLD: private key usage is not restricted to signing'
        }
        $ui = $privateRsa.Key.UIPolicy
        if (-not $ui -or
            (([int]$ui.ProtectionLevel -band [int][Security.Cryptography.CngUIProtectionLevels]::ForceHighProtection) -eq 0)) {
            throw 'HOLD: CNG key did not read back mandatory ProtectHigh UI policy'
        }

        $oids = @($issued.Extensions | ForEach-Object { $_.Oid.Value })
        if (@($oids | Sort-Object -Unique).Count -ne $oids.Count) {
            throw 'HOLD: duplicate X509 extension OIDs'
        }
        $ekus = @($issued.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.37' })
        if ($ekus.Count -ne 1 -or -not $ekus[0].Critical) {
            throw 'HOLD: dedicated EKU is missing, duplicated or noncritical'
        }
        $purposeExtension = [Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]$ekus[0]
        $purposes = @($purposeExtension.EnhancedKeyUsages | ForEach-Object { $_.Value })
        if ($purposes.Count -ne 1 -or $purposes[0] -cne $purposeOid) {
            throw ('HOLD: unexpected EKU purpose count/values: {0} / {1}' -f $purposes.Count, ($purposes -join ','))
        }
        $basics = @($issued.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.19' })
        if ($basics.Count -ne 1 -or $basics[0].CertificateAuthority -or $basics[0].HasPathLengthConstraint) {
            throw 'HOLD: issuer is not a plain non-CA end-entity cert'
        }
        $usages = @($issued.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.15' })
        if ($usages.Count -ne 1 -or
            $usages[0].KeyUsages -ne [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature) {
            throw 'HOLD: key usage not exactly DigitalSignature'
        }
        $period = ($issued.NotAfter.ToUniversalTime() - $issued.NotBefore.ToUniversalTime()).TotalDays
        if ($issued.NotAfter.ToUniversalTime() -le [datetime]::UtcNow -or $period -le 0 -or $period -gt 121) {
            throw 'HOLD: owner certificate validity outside bounded 120-day window'
        }

        # Prove actual sign/verify under the owner prompt policy (no secret data).
        # This may trigger the ProtectHigh user-consent UI; cancellation is HOLD.
        $challenge = New-Object byte[] 32
        $random = [Security.Cryptography.RandomNumberGenerator]::Create()
        try { $random.GetBytes($challenge) } finally { $random.Dispose() }
        $signature = $privateRsa.SignData($challenge,
            [Security.Cryptography.HashAlgorithmName]::SHA256,
            [Security.Cryptography.RSASignaturePadding]::Pkcs1)
        $publicRsa = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPublicKey($issued)
        try {
            if (-not $publicRsa.VerifyData($challenge, $signature,
                    [Security.Cryptography.HashAlgorithmName]::SHA256,
                    [Security.Cryptography.RSASignaturePadding]::Pkcs1)) {
                throw 'HOLD: owner signing key failed challenge verification'
            }
        } finally { if ($publicRsa) { $publicRsa.Dispose() } }
    } finally { if ($privateRsa) { $privateRsa.Dispose() } }

    # Only public DER leaves CurrentUser\My; Export-Certificate -Type CERT
    # cannot contain private signing key, and -NoClobber prevents overwrite.
    $null = Export-Certificate -Cert $issued -FilePath $publicDerPath -Type CERT -NoClobber -ErrorAction Stop
    $bytes = [IO.File]::ReadAllBytes($publicDerPath)
    if (-not [Collections.StructuralComparisons]::StructuralEqualityComparer.Equals($bytes, $issued.RawData)) {
        throw 'HOLD: public DER export differs from owner certificate RawData'
    }
    $publicOnly = [Security.Cryptography.X509Certificates.X509Certificate2]::new($bytes)
    try {
        if ($publicOnly.HasPrivateKey -or $publicOnly.Subject -cne $subject -or
            $publicOnly.Thumbprint -ine $issued.Thumbprint) {
            throw 'HOLD: public DER readback contains private key or identity mismatch'
        }
    } finally { $publicOnly.Dispose() }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $fingerprint = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '') }
    finally { $sha.Dispose() }
    Write-Output "OWNER_PUBLIC_DER_PATH=$publicDerPath"
    Write-Output "OWNER_PUBLIC_DER_SHA256=$fingerprint"
    Write-Output 'OWNER_PUBLIC_DER_ONLY_PASS; HOLD: independent fingerprint pin/broker install not performed'
} catch {
    # Never silently delete/recreate a newly generated nonexportable key;
    # preserve evidence, require owner-controlled targeted review/cleanup.
    Write-Warning ('HOLD: enrollment did not validate: {0}' -f $_.Exception.Message)
    Write-Warning ('If a key was created, inspect only {0} in Patrick CurrentUser\My. No automatic removal.' -f $subject)
    throw
}
