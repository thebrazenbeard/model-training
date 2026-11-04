# Ephemeral Windows GitHub Actions certificate-shape test ONLY. Never run on
# Lappy/VN1: no owner enrollment, no broker installation, no signing key reuse.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:GITHUB_ACTIONS -ne 'true' -or
    $env:RUNNER_OS -ne 'Windows' -or
    $env:GITHUB_REPOSITORY -ne 'thebrazenbeard/model-training') {
    throw 'Refuse certificate generation outside the expected ephemeral Windows CI repository'
}

$purposeOid = '2.25.281124513629607016020108740092472226515'
$id = [guid]::NewGuid().ToString('N')
$subject = 'CN=VERA-TrainingDay-Tier1B-EPHEMERAL-CI-' + $id
$certStore = 'Cert:\CurrentUser\My'
$publicDerPath = Join-Path $env:RUNNER_TEMP ('tier1b-ci-public-' + $id + '.cer')
$before = @((Get-ChildItem -Path $certStore | Where-Object { $_.Subject -eq $subject }))
if ($before.Count -ne 0) { throw 'Unique CI test subject already exists' }
$notBefore = [datetime]::UtcNow.AddMinutes(-2)
$notAfter = $notBefore.AddHours(2)
$testSucceeded = $false

try {
    # Microsoft PKIClient supports {critical} immediately after OID=, before
    # {text}; 2.5.29.19={text} is the documented end-entity/non-CA form.
    # This software KSP key is explicitly NonExportable and user-scoped.
    $created = New-SelfSignedCertificate -Type Custom -Subject $subject `
        -CertStoreLocation $certStore -Provider 'Microsoft Software Key Storage Provider' `
        -KeyAlgorithm RSA -KeyLength 3072 -HashAlgorithm SHA256 `
        -KeyExportPolicy NonExportable -KeyUsage DigitalSignature `
        -TextExtension @(
            ('2.5.29.37={critical}{text}' + $purposeOid),
            '2.5.29.19={text}'
        ) -NotBefore $notBefore -NotAfter $notAfter -ErrorAction Stop

    if (-not $created -or $created.Subject -cne $subject -or $created.Issuer -cne $subject) {
        throw 'Ephemeral owner certificate subject/issuer mismatch'
    }
    if (-not $created.HasPrivateKey) { throw 'Signer test certificate has no private key' }
    if ($created.PublicKey.Oid.Value -ne '1.2.840.113549.1.1.1') {
        throw 'Owner key algorithm is not RSA'
    }
    $privateRsa = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($created)
    if (-not $privateRsa -or $privateRsa.KeySize -ne 3072) { throw 'Owner RSA size not 3072' }
    try {
        if ($privateRsa -isnot [Security.Cryptography.RSACng]) {
            throw 'Private-key provider is not a qualifying Windows CNG RSA key'
        }
        if ([int]$privateRsa.Key.ExportPolicy -ne 0) {
            throw 'Owner signing key unexpectedly carries export permission'
        }
    } finally { if ($privateRsa) { $privateRsa.Dispose() } }

    $matches = @((Get-ChildItem -Path $certStore | Where-Object { $_.Subject -eq $subject }))
    if ($matches.Count -ne 1 -or $matches[0].Thumbprint -ine $created.Thumbprint) {
        throw 'Store subject is ambiguous or not idempotent'
    }
    $machineMatches = @(Get-ChildItem -Path 'Cert:\LocalMachine\My' |
        Where-Object { $_.Subject -eq $subject })
    if ($machineMatches.Count -ne 0) {
        throw 'Test signer leaked into machine store'
    }

    # Export-Certificate emits public DER; never call Export-PfxCertificate,
    # nor import a PFX anywhere. Readback deliberately holds no private key.
    $null = Export-Certificate -Cert $created -FilePath $publicDerPath -Type CERT -Force
    $raw = [IO.File]::ReadAllBytes($publicDerPath)
    if ($raw.Length -lt 500 -or $raw.Length -gt 8192) { throw 'Unexpected DER size' }
    $public = [Security.Cryptography.X509Certificates.X509Certificate2]::new($raw)
    try {
        if ($public.HasPrivateKey) { throw 'Public-only DER unexpectedly carries a private key' }
        if ($public.Subject -cne $subject -or $public.Issuer -cne $subject) {
            throw 'Public DER subject/issuer mismatch'
        }
        $publicRsa = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPublicKey($public)
        try {
            if (-not $publicRsa -or $publicRsa.KeySize -ne 3072) { throw 'DER RSA key size mismatch' }
        } finally { if ($publicRsa) { $publicRsa.Dispose() } }
        if ($public.NotAfter.ToUniversalTime() -le [datetime]::UtcNow -or
            ($public.NotAfter.ToUniversalTime() - $public.NotBefore.ToUniversalTime()).TotalHours -gt 3) {
            throw 'Test signing cert lifetime is absent or too long'
        }
        $oids = @($public.Extensions | ForEach-Object { $_.Oid.Value })
        if (@($oids | Sort-Object -Unique).Count -ne $oids.Count) {
            throw 'Duplicate X.509 extension OID'
        }
        $ekus = @($public.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.37' })
        if ($ekus.Count -ne 1 -or -not $ekus[0].Critical) {
            throw 'Dedicated EKU missing, duplicated, or not critical'
        }
        $eku = [Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]$ekus[0]
        $purposes = @($eku.EnhancedKeyUsages | ForEach-Object { $_.Value })
        if ($purposes.Count -ne 1 -or $purposes[0] -cne $purposeOid) {
            throw 'EKU includes an unexpected purpose'
        }
        $basics = @($public.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.19' })
        if ($basics.Count -ne 1 -or $basics[0].CertificateAuthority) {
            throw 'Non-CA basic constraint missing or wrong'
        }
        $usage = @($public.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.15' })
        if ($usage.Count -ne 1 -or
            (($usage[0].KeyUsages -band [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature) -eq 0) -or
            (($usage[0].KeyUsages -band [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyCertSign) -ne 0)) {
            throw 'DigitalSignature-only key usage not verified'
        }

        # Idempotent readback: a second parse produces the same DER digest,
        # and does not create or import a second certificate context.
        $rawAgain = [IO.File]::ReadAllBytes($publicDerPath)
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $digest = [Convert]::ToHexString($sha.ComputeHash($raw))
            $digestAgain = [Convert]::ToHexString($sha.ComputeHash($rawAgain))
        } finally { $sha.Dispose() }
        if ($digest -cne $digestAgain) { throw 'Public DER readback digest changed' }
        $testSucceeded = $true
        Write-Output "EPHEMERAL_TIER1_B_PUBLIC_DER_SHA256=$digest"
        Write-Output 'EPHEMERAL_TIER1_B_CERTIFICATE_SHAPE_PASS critical_eku=true rsa_bits=3072 public_has_private_key=false'
    } finally { $public.Dispose() }
} finally {
    # Never delete anything but this run's uniquely named test certificate.
    # The hosted runner itself is ephemeral; this is not proof of global key
    # removal from all providers or machine slots.
    $current = @(Get-ChildItem -Path $certStore |
        Where-Object { $_.Subject -eq $subject })
    foreach ($item in $current) {
        if ($item.Subject -cne $subject -or $item.Issuer -cne $subject) {
            throw 'Refusing cleanup of certificate without exact test identity'
        }
        Remove-Item -LiteralPath ($certStore + '\' + $item.Thumbprint) -Force
    }
    if (Test-Path -LiteralPath $publicDerPath) {
        Remove-Item -LiteralPath $publicDerPath -Force
    }
}
if (-not $testSucceeded) { throw 'Ephemeral test did not complete' }
