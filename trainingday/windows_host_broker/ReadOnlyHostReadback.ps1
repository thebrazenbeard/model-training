# TrainingDay broker evidence collector. Read-only and always HOLD.
# Run locally on Lappy in an administrative PowerShell session. No network,
# secrets, credentials, service changes, enrollment, or trainer execution.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$serviceName = 'VeraTrainingDayHostBroker'
$expectedRoot = 'D:\VERA'
$expectedServiceSid = 'S-1-5-18'  # LocalSystem
$broadSids = @('S-1-1-0', 'S-1-5-11', 'S-1-5-32-545')
$writeRights = [int64](
    [System.Security.AccessControl.FileSystemRights]::Write -bor
    [System.Security.AccessControl.FileSystemRights]::Modify -bor
    [System.Security.AccessControl.FileSystemRights]::FullControl -bor
    [System.Security.AccessControl.FileSystemRights]::ChangePermissions -bor
    [System.Security.AccessControl.FileSystemRights]::TakeOwnership)

function Get-ExecutablePath([string]$commandLine) {
    if ([string]::IsNullOrWhiteSpace($commandLine)) { return $null }
    if ($commandLine -match '^\s*"(?<executable>[^"\r\n]+?\.exe)"(?:\s|$)') {
        return $Matches.executable
    }
    if ($commandLine -match '^\s*(?<executable>[^\r\n]+?\.exe)(?:\s|$)') {
        return $Matches.executable.Trim()
    }
    return $null
}

function Get-DirectoryAclEvidence([string]$path) {
    $out = [ordered]@{
        exists = $false
        inspected = $false
        reparse_point = $null
        inheritance_protected = $null
        broad_write_allow = $null
        error = $null
    }
    try {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { return $out }
        $out.exists = $true
        $item = Get-Item -LiteralPath $path -Force
        $out.reparse_point = [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
        if ($out.reparse_point) { return $out }
        $acl = Get-Acl -LiteralPath $path
        $out.inheritance_protected = [bool]$acl.AreAccessRulesProtected
        $out.broad_write_allow = $false
        foreach ($ace in $acl.Access) {
            if ($ace.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
                continue
            }
            try {
                $sid = $ace.IdentityReference.Translate(
                    [System.Security.Principal.SecurityIdentifier]).Value
            } catch {
                $out.broad_write_allow = $null
                break
            }
            if ($broadSids -contains $sid -and
                (([int64]$ace.FileSystemRights -band $writeRights) -ne 0)) {
                $out.broad_write_allow = $true
            }
        }
        $out.inspected = $true
    } catch {
        $out.error = 'ACL_READ_UNAVAILABLE'
    }
    return $out
}

$report = [ordered]@{
    schema = 'VERA_TRAININGDAY_WINDOWS_READBACK_V1'
    evidence = 'LOCAL_READ_ONLY_PROBE'
    observed_utc = (Get-Date).ToUniversalTime().ToString('o')
    requested_root = $expectedRoot
    broker_service = $serviceName
    service_present = $false
    service_state = $null
    service_start_mode = $null
    configured_service_account = $null
    configured_localsystem = $false
    process_id = $null
    process_owner_sid = $null
    process_sid_is_localsystem = $false
    process_owner_sid_evidence_available = $false
    process_service_pid_stable = $false
    executable_path = $null
    executable_sha256 = $null
    executable_is_reparse_point = $null
    executable_acl = $null
    host_root_acl = $null
    host_state_acl = $null
    service_probe_error = $null
    rights_admitted = $false
    evaluator_custody_authenticated = $false
    gpu_lease_granted = $false
    physical_training_authorized = $false
    status = 'HOLD'
}

try {
    $svc = Get-CimInstance -ClassName Win32_Service -Filter "Name='$serviceName'" `
        -OperationTimeoutSec 5 -ErrorAction Stop
    if ($null -ne $svc) {
        $report.service_present = $true
        $report.service_state = [string]$svc.State
        $report.service_start_mode = [string]$svc.StartMode
        $report.configured_service_account = [string]$svc.StartName
        $report.configured_localsystem = ($svc.StartName -eq 'LocalSystem')
        $report.process_id = [int]$svc.ProcessId
        $path = Get-ExecutablePath ([string]$svc.PathName)
        if ($null -ne $path) {
            $report.executable_path = $path
            $report.executable_acl = Get-DirectoryAclEvidence (Split-Path -Path $path -Parent)
            try {
                $item = Get-Item -LiteralPath $path -Force -ErrorAction Stop
                $report.executable_is_reparse_point = [bool](
                    $item.Attributes -band [IO.FileAttributes]::ReparsePoint)
                if (-not $report.executable_is_reparse_point -and -not $item.PSIsContainer) {
                    $report.executable_sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            } catch {
                $report.service_probe_error = 'BINARY_READ_UNAVAILABLE'
            }
        }
        if ($report.process_id -gt 0) {
            try {
                $process = Get-CimInstance -ClassName Win32_Process `
                    -Filter ("ProcessId={0}" -f $report.process_id) `
                    -OperationTimeoutSec 5 -ErrorAction Stop
                if ($null -ne $process) {
                    $owner = Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid `
                        -OperationTimeoutSec 5 -ErrorAction Stop
                    if ($owner.ReturnValue -eq 0 -and $owner.Sid) {
                        # PID alone can be reused. Recheck the SCM service PID
                        # and the process creation identity after GetOwnerSid.
                        $serviceAfter = Get-CimInstance -ClassName Win32_Service `
                            -Filter "Name='$serviceName'" -OperationTimeoutSec 5 -ErrorAction Stop
                        $processAfter = Get-CimInstance -ClassName Win32_Process `
                            -Filter ("ProcessId={0}" -f $report.process_id) `
                            -OperationTimeoutSec 5 -ErrorAction Stop
                        $stable = ($null -ne $serviceAfter -and $null -ne $processAfter -and
                            [int]$serviceAfter.ProcessId -eq $report.process_id -and
                            [int]$processAfter.ProcessId -eq $report.process_id -and
                            $null -ne $process.CreationDate -and
                            $process.CreationDate -eq $processAfter.CreationDate)
                        $report.process_service_pid_stable = [bool]$stable
                        if ($stable) {
                            $report.process_owner_sid = [string]$owner.Sid
                            $report.process_owner_sid_evidence_available = $true
                            $report.process_sid_is_localsystem = ($owner.Sid -eq $expectedServiceSid)
                        }
                    }
                }
            } catch {
                # Unknown process identity never becomes an authorization grant.
            }
        }
    }
} catch {
    $report.service_probe_error = 'SERVICE_CIM_UNAVAILABLE'
}

$report.host_root_acl = Get-DirectoryAclEvidence $expectedRoot
$report.host_state_acl = Get-DirectoryAclEvidence 'C:\ProgramData\VERA\TrainingDayHostBroker'
$report | ConvertTo-Json -Depth 6
