[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PreflightScript,
    [Parameter(Mandatory = $true)][string]$AclHelper,
    [Parameter(Mandatory = $true)][string]$UnsignedBrokerExecutable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$file = Get-Item -LiteralPath $UnsignedBrokerExecutable
$before = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash
$baseline = 'D4885718A9D81FBE4D28AF7232CFAD302DA10763741CB214D5BD9BE4443102FB'
$fakeSigner = '1111111111111111111111111111111111111111'
. $AclHelper

$nonAdmin = [Security.Principal.NTAccount]::new('BUILTIN\Users')
$readAce = [pscustomobject]@{
    AccessControlType = [Security.AccessControl.AccessControlType]::Allow
    FileSystemRights = [Security.AccessControl.FileSystemRights]::ReadAndExecute
    IdentityReference = $nonAdmin
}
$writeAce = [pscustomobject]@{
    AccessControlType = [Security.AccessControl.AccessControlType]::Allow
    FileSystemRights = [Security.AccessControl.FileSystemRights]::Modify
    IdentityReference = $nonAdmin
}
$changeDaclAce = [pscustomobject]@{
    AccessControlType = [Security.AccessControl.AccessControlType]::Allow
    FileSystemRights = [Security.AccessControl.FileSystemRights]::ChangePermissions
    IdentityReference = $nonAdmin
}
if (-not (Test-TrainingDayNoUnexpectedWriters ([pscustomobject]@{ Access = @($readAce) }))) {
    throw 'ReadAndExecute for non-admin was wrongly classified as mutating'
}
if (Test-TrainingDayNoUnexpectedWriters ([pscustomobject]@{ Access = @($writeAce) })) {
    throw 'Modify for non-admin unexpectedly passed the ACL preflight'
}
if (Test-TrainingDayNoUnexpectedWriters ([pscustomobject]@{ Access = @($changeDaclAce) })) {
    throw 'ChangePermissions for non-admin unexpectedly passed the ACL preflight'
}

function Invoke-Probe([string]$CandidatePath, [string]$ExpectedCandidateHash, [string]$Signer = $fakeSigner) {
    $lines = @(& $PreflightScript -CandidateExecutable $CandidatePath `
        -ExpectedCandidateSha256 $ExpectedCandidateHash `
        -ExpectedInstalledSha256 $baseline -ExpectedSignerThumbprint $Signer)
    if ($lines.Count -ne 1) { throw 'Expected exactly one JSON receipt line' }
    return ($lines[0] | ConvertFrom-Json)
}

function Assert-Held($result, [string]$reason) {
    if ($result.schema -ne 'VERA_TRAININGDAY_BROKER_MAINTENANCE_PREFLIGHT_V1' -or
        $result.maintenance_state -ne 'HOLD' -or
        $result.broker_launch_authorized -ne $false -or
        $result.certificate_import_authorized -ne $false -or
        $result.service_restart_authorized -ne $false) {
        throw 'Read-only preflight illegally returned readiness or an effect authorization'
    }
    if (@($result.problems) -notcontains $reason) { throw "Expected hold reason $reason" }
}

$unsigned = Invoke-Probe $file.FullName $before
Assert-Held $unsigned 'CANDIDATE_SIGNATURE_NOT_VALID'

$badHash = ('0' * 64)
if ($badHash -ieq $before) { $badHash = ('f' * 64) }
$mismatched = Invoke-Probe $file.FullName $badHash
Assert-Held $mismatched 'CANDIDATE_HASH_MISMATCH'

$invalidSigner = Invoke-Probe $file.FullName $before 'not-a-thumbprint'
Assert-Held $invalidSigner 'INVALID_EXPECTED_SIGNER_THUMBPRINT'

$missing = Invoke-Probe ($file.FullName + '.absent') $before
Assert-Held $missing 'CANDIDATE_NOT_READABLE'

$minimum = @(& $PreflightScript -CandidateExecutable $file.FullName `
    -ExpectedCandidateSha256 $before -ExpectedInstalledSha256 $baseline `
    -ExpectedSignerThumbprint $fakeSigner -MinimumFreeGpuMiB 1) | ConvertFrom-Json
Assert-Held $minimum 'GPU_MINIMUM_BELOW_REVIEWED_BOUND'

$after = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash
if ($after -ne $before) { throw 'Preflight mutated candidate bytes' }
Write-Output 'BROKER_MAINTENANCE_READONLY_NEGATIVE_PASS cases=8 no_service_install=true no_launch=true'
