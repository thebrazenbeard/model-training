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
    # Complete minimal b12125 CLI dynamic-link closure. The official release
    # contains additional ggml CPU variants and tools not needed by this test.
    $required = @(
        'llama-cli.exe', 'llama-cli-impl.dll', 'llama-server-impl.dll',
        'llama-common.dll', 'mtmd.dll', 'llama.dll', 'ggml.dll',
        'ggml-base.dll', 'ggml-cpu-x64.dll'
    )
    $good = [System.IO.Path]::Combine($root, 'good.zip')
    $goodEntries = @($required | ForEach-Object { 'pkg/bin/' + $_ }) + @('pkg/readme.txt')
    New-Fixture $good $goodEntries
    $selected = @(Test-Q4SafeZipEntries -ArchivePath $good)
    if ($selected.Count -ne $required.Count) { throw 'Expected exactly the complete minimal CLI binary closure.' }
    $bin = [System.IO.Path]::Combine($root, 'bin')
    [void][System.IO.Directory]::CreateDirectory($bin)
    $hashes = @(Expand-Q4PinnedRunner -ArchivePath $good -BinDirectory $bin)
    if ($hashes.Count -ne $required.Count -or
        -not [System.IO.File]::Exists([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -or
        [System.IO.File]::Exists([System.IO.Path]::Combine($bin, 'readme.txt'))) {
        throw 'Unexpected extraction allowlist behavior.'
    }
    foreach ($name in $required) {
        if (-not [System.IO.File]::Exists([System.IO.Path]::Combine($bin, $name))) {
            throw ('Required CLI dependency not extracted: ' + $name)
        }
    }
    # Negative coverage: all nine closure entries are mandatory, and the
    # archive preflight must reject every missing variant before extraction
    # writes the first output byte.
    foreach ($missing in $required) {
        $fixture = [System.IO.Path]::Combine($root, ('missing-' + $missing + '.zip'))
        $entries = @($required | Where-Object { $_ -cne $missing } | ForEach-Object { 'pkg/bin/' + $_ })
        New-Fixture $fixture $entries
        Assert-Throws { Test-Q4SafeZipEntries -ArchivePath $fixture } ('missing dependency ' + $missing)
        $emptyTarget = [System.IO.Path]::Combine($root, ('target-' + $missing))
        [void][System.IO.Directory]::CreateDirectory($emptyTarget)
        Assert-Throws { Expand-Q4PinnedRunner -ArchivePath $fixture -BinDirectory $emptyTarget } ('pre-extraction closure ' + $missing)
        if (@(Get-ChildItem -LiteralPath $emptyTarget -Force).Count -ne 0) {
            throw ('Missing dependency fixture wrote files before rejecting ' + $missing)
        }
    }
    $sha = (Get-FileHash -LiteralPath ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -Algorithm SHA256).Hash
    [void](Assert-Q4FileIntegrity -Path ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -ExpectedBytes 3 -ExpectedSha256 $sha)
    Assert-Throws { Assert-Q4FileIntegrity -Path ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -ExpectedBytes 4 -ExpectedSha256 $sha } 'size mismatch'
    Assert-Throws { Assert-Q4FileIntegrity -Path ([System.IO.Path]::Combine($bin, 'llama-cli.exe')) -ExpectedBytes 3 -ExpectedSha256 ('0' * 64) } 'hash mismatch'

    # Fresh-sibling repair candidate: copy pinned bytes without writing to
    # the original source or accepting a corrupted/renamed target.
    $original = [System.IO.Path]::Combine($root, 'original.gguf')
    [System.IO.File]::WriteAllText($original, 'strictly-pinned-fixture')
    $originalHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
    $originalLength = (Get-Item -LiteralPath $original).Length
    $sibling = [System.IO.Path]::Combine($root, 'new-sibling.gguf')
    [void](Copy-Q4PinnedArtifact -Source $original -Target $sibling -ExpectedBytes $originalLength -ExpectedSha256 $originalHash)
    [void](Assert-Q4FileIntegrity -Path $sibling -ExpectedBytes $originalLength -ExpectedSha256 $originalHash)
    if ([System.IO.File]::ReadAllText($original) -cne 'strictly-pinned-fixture') {
        throw 'Verified sibling copy modified source evidence.'
    }
    Assert-Throws { Copy-Q4PinnedArtifact -Source $original -Target $sibling -ExpectedBytes $originalLength -ExpectedSha256 $originalHash } 'copy existing target'
    [System.IO.File]::WriteAllText($original, 'corrupted-fixture')
    $rejected = [System.IO.Path]::Combine($root, 'corruption-rejected.gguf')
    Assert-Throws { Copy-Q4PinnedArtifact -Source $original -Target $rejected -ExpectedBytes $originalLength -ExpectedSha256 $originalHash } 'corrupted source copy'
    if ([System.IO.File]::Exists($rejected)) { throw 'Corrupt verified-copy source produced a final target.' }

    # A copy repair is permitted only to a *sibling* stage leaf. This checks
    # the canonical parent relationship, not a fragile string prefix.
    $stageA = [System.IO.Path]::Combine($root, 'original-leaf')
    $stageB = [System.IO.Path]::Combine($root, 'new-leaf')
    Assert-Q4SiblingStagePair -ExistingStage $stageA -NewStage $stageB
    Assert-Q4SiblingStagePair -ExistingStage ($stageA + '\.') -NewStage ($stageB + '\.')
    Assert-Throws { Assert-Q4SiblingStagePair -ExistingStage $stageA -NewStage $stageA } 'same stage path'
    Assert-Throws { Assert-Q4SiblingStagePair -ExistingStage $stageA -NewStage ([System.IO.Path]::Combine($stageA, 'nested-repair')) } 'destination nested inside source'
    Assert-Throws { Assert-Q4SiblingStagePair -ExistingStage ([System.IO.Path]::Combine($stageA, 'nested-source')) -NewStage $stageB } 'source nested relative to destination'
    Assert-Throws { Assert-Q4SiblingStagePair -ExistingStage $stageA -NewStage ([System.IO.Path]::Combine($root, 'different-parent', 'new-leaf')) } 'different parent'

    # An explicitly supplied blank local-copy argument must not fall through
    # into network download mode, even if PowerShell accepted its binding.
    foreach ($blank in @('', ' ', "`t")) {
        $guarded = $false
        try {
            & (Join-Path $PSScriptRoot 'Stage-Q4Trial.ps1') -Destination ([System.IO.Path]::Combine($root, 'never-stage')) -CopyFromVerifiedStage $blank | Out-Null
        } catch {
            if ($_.Exception.Message -like '*Explicit CopyFromVerifiedStage requires a nonempty source path*') {
                $guarded = $true
            } else {
                throw ('Blank copy-source guard raised an unexpected failure: ' + $_.Exception.Message)
            }
        }
        if (-not $guarded -or [System.IO.Directory]::Exists([System.IO.Path]::Combine($root, 'never-stage'))) {
            throw 'Blank CopyFromVerifiedStage did not fail before staging effects.'
        }
    }

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
    Write-Output 'PASS: offline path, complete CLI DLL closure, pre-extraction rejects, ZIP, extraction, hash and size negative tests'
} finally {
    # Only this exact GUID-scoped temp directory was created by the test itself.
    [System.IO.Directory]::Delete($root, $true)
}
