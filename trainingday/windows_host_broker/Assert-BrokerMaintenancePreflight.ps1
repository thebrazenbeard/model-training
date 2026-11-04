# Source-only, read-only maintenance evidence collection. This script never
# installs binaries, changes certificates, restarts services, or grants launch.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CandidateExecutable,
    [Parameter(Mandatory = $true)][string]$ExpectedCandidateSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedInstalledSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedSignerThumbprint,
    [string]$ServiceName = 'VeraTrainingDayHostBroker',
    [int]$MinimumFreeGpuMiB = 3328
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$problems = New-Object 'System.Collections.Generic.List[string]'
$facts = [ordered]@{
    candidate_sha256 = $null
    candidate_signer_thumbprint = $null
    installed_sha256 = $null
    service_state = 'UNKNOWN'
    service_process_id = $null
    gpu_total_mib = $null
    gpu_free_mib = $null
    potentially_competing_processes = @()
}

if ($ExpectedCandidateSha256 -cnotmatch '^[a-fA-F0-9]{64}$' -or
    $ExpectedInstalledSha256 -cnotmatch '^[a-fA-F0-9]{64}$') {
    $problems.Add('INVALID_EXPECTED_HASH')
}
if ($ExpectedSignerThumbprint -cnotmatch '^[a-fA-F0-9]{40}$') {
    $problems.Add('INVALID_EXPECTED_SIGNER_THUMBPRINT')
}
if ($MinimumFreeGpuMiB -lt 3328) {
    $problems.Add('GPU_MINIMUM_BELOW_REVIEWED_BOUND')
}

try {
    $candidate = Get-Item -LiteralPath $CandidateExecutable -ErrorAction Stop
    if ($candidate.PSIsContainer -or (($candidate.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
        $problems.Add('CANDIDATE_NOT_REGULAR_FILE')
    } else {
        $candidateSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate.FullName).Hash
        $facts.candidate_sha256 = $candidateSha
        if ($candidateSha -ine $ExpectedCandidateSha256) { $problems.Add('CANDIDATE_HASH_MISMATCH') }
        try {
            $sig = Get-AuthenticodeSignature -LiteralPath $candidate.FullName
            if ($sig.SignerCertificate) { $facts.candidate_signer_thumbprint = $sig.SignerCertificate.Thumbprint }
            if ($sig.Status -ne 'Valid' -or -not $sig.SignerCertificate) {
                $problems.Add('CANDIDATE_SIGNATURE_NOT_VALID')
            } elseif ($sig.SignerCertificate.Thumbprint -ine $ExpectedSignerThumbprint) {
                $problems.Add('CANDIDATE_SIGNER_NOT_PINNED')
            }
        } catch { $problems.Add('CANDIDATE_SIGNATURE_CHECK_UNAVAILABLE') }
    }
} catch { $problems.Add('CANDIDATE_NOT_READABLE') }

try {
    $service = Get-CimInstance -ClassName Win32_Service -Filter ("Name='{0}'" -f $ServiceName)
    if (-not $service) {
        $problems.Add('INSTALLED_SERVICE_NOT_FOUND')
    } else {
        $facts.service_state = [string]$service.State
        $facts.service_process_id = [int]$service.ProcessId
        if ($service.StartName -ine 'LocalSystem') { $problems.Add('SERVICE_IDENTITY_CHANGED') }
        if ($service.State -ine 'Running') { $problems.Add('SERVICE_NOT_RUNNING') }
        $quoted = [regex]::Match([string]$service.PathName, '^\s*"([^"]+\.exe)"(?:\s|$)')
        if (-not $quoted.Success) {
            $problems.Add('INSTALLED_SERVICE_PATH_UNQUOTED_OR_AMBIGUOUS')
        } else {
            $installedPath = [Environment]::ExpandEnvironmentVariables($quoted.Groups[1].Value)
            $installed = Get-Item -LiteralPath $installedPath -ErrorAction Stop
            if ($installed.PSIsContainer -or (($installed.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
                $problems.Add('INSTALLED_BINARY_NOT_REGULAR_FILE')
            } else {
                $installedSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $installed.FullName).Hash
                $facts.installed_sha256 = $installedSha
                if ($installedSha -ine $ExpectedInstalledSha256) { $problems.Add('INSTALLED_HASH_CHANGED_SINCE_BASELINE') }
                $acl = Get-Acl -LiteralPath $installed.DirectoryName
                foreach ($ace in $acl.Access) {
                    if ($ace.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) { continue }
                    $writer = [Security.AccessControl.FileSystemRights]::Write -bor
                              [Security.AccessControl.FileSystemRights]::Modify -bor
                              [Security.AccessControl.FileSystemRights]::FullControl
                    if (($ace.FileSystemRights -band $writer) -eq 0) { continue }
                    $sid = try { $ace.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value } catch { '' }
                    $trusted = @('S-1-5-18', 'S-1-5-32-544')
                    if ($sid -notin $trusted -and $ace.IdentityReference.Value -notmatch '^NT SERVICE\\TrustedInstaller$') {
                        $problems.Add('INSTALL_DIRECTORY_HAS_OTHER_WRITER')
                        break
                    }
                }
            }
        }
    }
} catch { $problems.Add('INSTALLED_SERVICE_OR_ACL_INSPECTION_FAILED') }

try {
    $raw = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or @($raw).Count -ne 1 -or [string]$raw -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*$') {
        $problems.Add('GPU_OBSERVATION_UNAVAILABLE_OR_AMBIGUOUS')
    } else {
        $facts.gpu_total_mib = [int]$Matches[1]
        $facts.gpu_free_mib = [int]$Matches[2]
        if ($facts.gpu_free_mib -lt $MinimumFreeGpuMiB) { $problems.Add('GPU_VRAM_PREFLIGHT_FAIL') }
    }
} catch { $problems.Add('GPU_OBSERVATION_UNAVAILABLE_OR_AMBIGUOUS') }

try {
    $activeNames = @('blender', 'msedge', 'python', 'pythonw', 'wsl', 'wslhost', 'vmmemWSL', 'docker')
    $active = @(Get-Process -Name $activeNames -ErrorAction SilentlyContinue |
        Sort-Object -Property ProcessName, Id |
        ForEach-Object { '{0}:{1}' -f $_.ProcessName, $_.Id })
    $facts.potentially_competing_processes = $active
    if ($active.Count -gt 0) { $problems.Add('OTHER_ACTIVE_WORKLOADS_REQUIRE_OWNER_WINDOW') }
} catch { $problems.Add('WORKLOAD_INVENTORY_UNAVAILABLE') }

# Never return a ready-to-install or launch state: passing a snapshot does not
# prove no concurrent changes, signer custody, protected rights, or WSL cleanup.
$result = [ordered]@{
    schema = 'VERA_TRAININGDAY_BROKER_MAINTENANCE_PREFLIGHT_V1'
    maintenance_state = 'HOLD'
    broker_launch_authorized = $false
    certificate_import_authorized = $false
    service_restart_authorized = $false
    problems = @($problems | Sort-Object -Unique)
    observed = $facts
    note = 'Read-only snapshot; requires independent review and a separately controlled owner window.'
}
$result | ConvertTo-Json -Depth 6 -Compress
