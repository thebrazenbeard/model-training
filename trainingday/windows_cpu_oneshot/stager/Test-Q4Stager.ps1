#Requires -Version 5.1
# Offline contract tests. Creates only one self-generated temp directory; no network.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Windows host required for ACL/path tests.' }
. (Join-Path $PSScriptRoot 'Q4Stager.Core.ps1')
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Assert-Throws {
    param([scriptblock]$Action, [string]$Label)
    $threw = $false
    try { & $Action | Out-Null } catch { $threw = $true }
    if (-not $threw) { throw ('Negative test unexpectedly passed: ' + $Label) }
}
function New-Fixture {
    param([string]$Path, [string[]]$Entries, [bool]$Symlink = $false)
    $file = [System.IO.FileStream]::new($Path, [System.IO.FileMode]::CreateNew)
    $zip = [System.IO.Compression.ZipArchive]::new($file, [System.IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        foreach ($name in $Entries) {
            $entry = $zip.CreateEntry($name)
            if ($Symlink -and $name -eq 'llama-cli.exe') {
                # Unix type 0120000 (symbolic link) in upper 16 bits.
                # Windows PowerShell 5.1 parses this hexadecimal literal as a
                # signed Int32; use its unsigned decimal value explicitly.
                $attr = [uint32]2684354560
                $entry.ExternalAttributes = [System.BitConverter]::ToInt32([System.BitConverter]::GetBytes($attr), 0)
            }
            $s = $entry.Open()
            try {
                $byte = [byte[]]@(65, 66, 67)
                $s.Write($byte, 0, $byte.Length)
            } finally { $s.Dispose() }
        }
    } finally { $zip.Dispose() }
}

$root = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), 'bt2-q4-unit-' + [guid]::NewGuid().ToString('N'))
[void][System.IO.Directory]::CreateDirectory($root)
try {
    $good = [System.IO.Path]::Combine($root, 'good.zip')
    New-Fixture $good @('pkg/bin/llama-cli.exe', 'pkg/bin/ggml.dll', 'pkg/readme.txt')
    $selected = @(Test-Q4SafeZipEntries -ArchivePath $good)
    if ($selected.Count -ne 2) { throw 'Expected exactly two allowlisted binaries.' }
    $bin = [System.IO.Path]::Combine($root, 'bin')
    [void][System.IO.Directory]::CreateDirectory($bin)
    $hashes = @(Expand-Q4PinnedRunner -ArchivePath $good -BinDirectory $bin)
    if ($hashes.Count -ne 2 -or
        -not [System.IO.File]::Exists([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -or
        [System.IO.File]::Exists([System.IO.Path]::Combine($bin, 'readme.txt'))) {
        throw 'Unexpected extraction allowlist behavior.'
    }
    $sha = (Get-FileHash -LiteralPath ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -Algorithm SHA256).Hash
    [void](Assert-Q4FileIntegrity -Path ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -ExpectedBytes 3 -ExpectedSha256 $sha)
    Assert-Throws { Assert-Q4FileIntegrity -Path ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -ExpectedBytes 4 -ExpectedSha256 $sha } 'size mismatch'
    Assert-Throws { Assert-Q4FileIntegrity -Path ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -ExpectedBytes 3 -ExpectedSha256 ('0' * 64) } 'hash mismatch'

    $invalid = @(
        @('pkg/../llama-cli.exe', 'ggml.dll'),
        @('/llama-cli.exe', 'ggml.dll'),
        @('C:/llama-cli.exe', 'ggml.dll'),
        @('x:stream/llama-cli.exe', 'ggml.dll'),
        @('pkg\llama-cli.exe', 'ggml.dll'),
        @('pkg/CON/file', 'llama-cli.exe'),
        @('x/llama-cli.exe', 'y/LLAMA-CLI.EXE')
    )
    $i = 0
    foreach ($entries in $invalid) {
        $path = [System.IO.Path]::Combine($root, ('bad-{0}.zip' -f $i))
        New-Fixture $path $entries
        Assert-Throws { Test-Q4SafeZipEntries -ArchivePath $path } ('unsafe ZIP case ' + $i)
        $i++
    }
    $symlink = [System.IO.Path]::Combine($root, 'link.zip')
    New-Fixture $symlink @('llama-cli.exe', 'ggml.dll') $true
    Assert-Throws { Test-Q4SafeZipEntries -ArchivePath $symlink } 'symlink'
    Assert-Throws { Assert-Q4IsolatedDestination -Destination ([System.IO.Path]::GetPathRoot($env:USERPROFILE)) } 'drive root'
    Assert-Throws { Assert-Q4IsolatedDestination -Destination $env:USERPROFILE } 'profile root'
    $collision = [System.IO.Path]::Combine($root, 'collision')
    [void][System.IO.Directory]::CreateDirectory($collision)
    $sentinel = [System.IO.Path]::Combine($collision, 'preserve.txt')
    [System.IO.File]::WriteAllText($sentinel, 'owned-by-other-process')
    Assert-Throws { New-Q4ExclusiveLeaf -Path $collision } 'existing leaf collision'
    if ([System.IO.File]::ReadAllText($sentinel) -cne 'owned-by-other-process') {
        throw 'Collision attempt touched pre-existing leaf.'
    }
    $private = [System.IO.Path]::Combine($root, 'private')
    New-Q4ExclusiveLeaf -Path $private
    Set-Q4PrivateDirectory -Directory $private
    $acl = Get-Acl -LiteralPath $private
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    if (-not $acl.AreAccessRulesProtected -or
        $acl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -cne $sid.Value) {
        throw 'Private directory owner or ACL inheritance protection failed readback.'
    }
    $aces = @($acl.GetAccessRules($true, $false, [System.Security.Principal.SecurityIdentifier]))
    $currentAllow = @($aces | Where-Object {
        $_.IdentityReference.Value -eq $sid.Value -and
        $_.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Allow -and
        (($_.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::FullControl) -eq [System.Security.AccessControl.FileSystemRights]::FullControl)
    })
    if ($currentAllow.Count -lt 1) { throw 'Current user lacks explicit private-directory access.' }

    # Forced TOCTOU: preflight sees the absent leaf; another process creates
    # it before the atomic call. Neither bytes nor ACL may be changed.
    $forcedParent = [System.IO.Path]::Combine($env:USERPROFILE, 'bt2-q4-race-test-' + [guid]::NewGuid().ToString('N'))
    [void][System.IO.Directory]::CreateDirectory($forcedParent)
    try {
        $forcedLeaf = [System.IO.Path]::Combine($forcedParent, 'leaf')
        [void](Assert-Q4IsolatedDestination -Destination $forcedLeaf)
        [void][System.IO.Directory]::CreateDirectory($forcedLeaf)
        $forcedSentinel = [System.IO.Path]::Combine($forcedLeaf, 'sentinel.txt')
        [System.IO.File]::WriteAllText($forcedSentinel, 'competitor-evidence')
        $beforeAcl = (Get-Acl -LiteralPath $forcedLeaf).Sddl
        Assert-Throws { New-Q4ExclusiveLeaf -Path $forcedLeaf } 'forced preflight/create collision'
        if ([System.IO.File]::ReadAllText($forcedSentinel) -cne 'competitor-evidence' -or
            (Get-Acl -LiteralPath $forcedLeaf).Sddl -cne $beforeAcl) {
            throw 'Forced collision modified competing leaf data or ACL.'
        }
    } finally {
        [System.IO.Directory]::Delete($forcedParent, $true)
    }
    $race = [System.IO.Path]::Combine($root, 'race')
    $helper = Join-Path $PSScriptRoot 'Q4Stager.Core.ps1'
    $jobs = @()
    try {
        foreach ($n in 1..2) {
            $jobs += Start-Job -ScriptBlock {
                param($Helper, $Race)
                . $Helper
                try {
                    New-Q4ExclusiveLeaf -Path $Race
                    'CREATED'
                } catch {
                    'REFUSED'
                }
            } -ArgumentList $helper, $race
        }
        $results = @($jobs | Wait-Job | Receive-Job)
        if ((@($results | Where-Object { $_ -eq 'CREATED' })).Count -ne 1 -or
            (@($results | Where-Object { $_ -eq 'REFUSED' })).Count -ne 1) {
            throw 'Concurrent creation did not produce exactly one winner and one refusal.'
        }
    } finally {
        foreach ($job in $jobs) {
            if ($null -ne $job) { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue }
        }
    }
    Write-Output 'PASS: offline path, ZIP, allowlist, extraction, hash and size negative tests'
} finally {
    # Only this exact GUID-scoped temp directory was created by the test itself.
    [System.IO.Directory]::Delete($root, $true)
}
