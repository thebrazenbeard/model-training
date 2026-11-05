# Windows PowerShell 5.1 source-only staging helpers.
# Security boundary: this file never executes the staged binaries.
Set-StrictMode -Version Latest

if (-not ('Q4Win32ExclusiveLeaf' -as [type])) {
    Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class Q4Win32ExclusiveLeaf {
    [DllImport("kernel32.dll", EntryPoint = "CreateDirectoryW", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern bool CreateDirectory(string path, IntPtr securityAttributes);
}
'@ -ErrorAction Stop
}

function Assert-Q4IsolatedDestination {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Destination)

    if ([string]::IsNullOrWhiteSpace($env:USERPROFILE) -or
        -not [System.IO.Path]::IsPathRooted($Destination)) {
        throw 'Destination must be absolute and inside the current user profile.'
    }
    $profileRoot = [System.IO.Path]::GetFullPath($env:USERPROFILE).TrimEnd('\')
    $full = [System.IO.Path]::GetFullPath($Destination).TrimEnd('\')
    if (-not $full.StartsWith($profileRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Destination must be inside the current user profile.'
    }
    $relative = $full.Substring($profileRoot.Length + 1)
    $components = $relative.Split([char]'\')
    if ($components.Length -lt 2 -or $components -contains '' -or $components -contains '.' -or $components -contains '..') {
        throw 'Destination requires an existing user-owned parent and a dedicated new leaf directory.'
    }
    if ([System.IO.Directory]::Exists($full) -or [System.IO.File]::Exists($full)) {
        throw 'Destination must not already exist. Never overwrite prior model evidence.'
    }
    $parent = [System.IO.Path]::GetDirectoryName($full)
    if (-not [System.IO.Directory]::Exists($parent)) {
        throw 'The isolated destination parent must exist before staging.'
    }
    $check = $parent
    while ($check.Length -ge $profileRoot.Length) {
        $item = Get-Item -LiteralPath $check -Force -ErrorAction Stop
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Destination path crosses a reparse point.'
        }
        if ($check.Equals($profileRoot, [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $check = [System.IO.Path]::GetDirectoryName($check)
        if ([string]::IsNullOrEmpty($check)) { throw 'Invalid destination ancestry.' }
    }
    return $full
}

function New-Q4ExclusiveLeaf {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)

    # Win32 CreateDirectoryW creates exactly one directory and fails atomically
    # when ANY other process has already created the leaf. Unlike
    # Directory.CreateDirectory, it cannot silently accept an existing target.
    $ok = [Q4Win32ExclusiveLeaf]::CreateDirectory($Path, [IntPtr]::Zero)
    if (-not $ok) {
        $code = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        if ($code -eq 183) { throw 'Destination leaf already exists; atomic collision refused.' }
        throw ('Exclusive destination creation failed with Win32 error ' + $code)
    }
}

function Set-Q4PrivateDirectory {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Directory)
    # ACLs restrict other ordinary accounts; they are not immutable file custody.
    # The launch supervisor MUST rehash runner and model immediately before use.
    $acl = Get-Acl -LiteralPath $Directory -ErrorAction Stop
    $acl.SetAccessRuleProtection($true, $false)
    $current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl.SetOwner($current)
    foreach ($sid in @($current, ([System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')))) {
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid,
            [System.Security.AccessControl.FileSystemRights]::FullControl,
            [System.Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit',
            [System.Security.AccessControl.PropagationFlags]::None,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Directory -AclObject $acl -ErrorAction Stop
}

function Save-Q4PinnedDownload {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][uri]$Uri,
        [Parameter(Mandatory)][string]$Target,
        [Parameter(Mandatory)][long]$ExpectedBytes,
        [Parameter(Mandatory)][string]$ExpectedSha256
    )

    if ($Uri.Scheme -ne 'https' -or $ExpectedBytes -le 0 -or
        $ExpectedSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'Invalid pinned HTTPS artifact manifest.'
    }
    if ([System.IO.File]::Exists($Target)) { throw 'Artifact target already exists.' }
    $partial = $Target + '.partial-' + ([guid]::NewGuid().ToString('N'))
    $response = $null
    $source = $null
    $sink = $null
    try {
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
        $request = [System.Net.HttpWebRequest]::Create($Uri)
        $request.Method = 'GET'
        $request.AllowAutoRedirect = $true
        $request.MaximumAutomaticRedirections = 6
        $request.Timeout = 120000
        $request.ReadWriteTimeout = 120000
        $request.UserAgent = 'BT2-Q4-Verified-Stager/1'
        $response = $request.GetResponse()
        if ($response.ResponseUri.Scheme -ne 'https') { throw 'Insecure redirect refused.' }
        if ($response.ContentLength -ge 0 -and $response.ContentLength -ne $ExpectedBytes) {
            throw 'HTTP Content-Length does not match pinned artifact size.'
        }
        $source = $response.GetResponseStream()
        $sink = [System.IO.FileStream]::new(
            $partial, [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None,
            1048576, [System.IO.FileOptions]::SequentialScan
        )
        $buffer = New-Object byte[] 1048576
        [long]$count = 0
        while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $count += [long]$read
            if ($count -gt $ExpectedBytes) { throw 'Download exceeded pinned artifact length.' }
            $sink.Write($buffer, 0, $read)
        }
        $sink.Flush()
        $sink.Dispose()
        $sink = $null
        if ($count -ne $ExpectedBytes) { throw 'Download was incomplete.' }
        $verified = Assert-Q4FileIntegrity -Path $partial -ExpectedBytes $ExpectedBytes -ExpectedSha256 $ExpectedSha256
        [System.IO.File]::Move($partial, $Target)
        return $verified
    } catch {
        # Do not leak signed redirect URLs or HTTP query credentials in diagnostics.
        throw 'Pinned artifact download or integrity verification failed.'
    } finally {
        if ($null -ne $sink) { $sink.Dispose() }
        if ($null -ne $source) { $source.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
        if ([System.IO.File]::Exists($partial)) {
            Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        }
    }
}

function Assert-Q4FileIntegrity {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path,
          [Parameter(Mandatory)][long]$ExpectedBytes,
          [Parameter(Mandatory)][string]$ExpectedSha256)
    $bytes = (Get-Item -LiteralPath $Path -ErrorAction Stop).Length
    if ($bytes -ne $ExpectedBytes) { throw 'Artifact file size mismatch.' }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
    if ($actual -cne $ExpectedSha256.ToLowerInvariant()) { throw 'Artifact SHA256 mismatch.' }
    return [pscustomobject]@{ Bytes = [long]$bytes; Sha256 = $actual }
}

function Test-Q4SafeZipEntries {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$ArchivePath)

    Add-Type -AssemblyName System.IO.Compression -ErrorAction Stop
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction Stop
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $selected = [System.Collections.Generic.List[System.IO.Compression.ZipArchiveEntry]]::new()
        $found = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        [long]$uncompressed = 0
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName
            if ([string]::IsNullOrWhiteSpace($name) -or
                $name.StartsWith('/') -or $name.StartsWith('\') -or
                $name.Contains('\') -or $name.Contains(':') -or
                $name.IndexOf([char]0) -ge 0) { throw 'Unsafe ZIP entry path.' }
            $segments = $name.TrimEnd('/').Split([char]'/')
            foreach ($seg in $segments) {
                if ($seg -eq '' -or $seg -eq '.' -or $seg -eq '..' -or
                    $seg.EndsWith('.') -or $seg.EndsWith(' ') -or
                    $seg -match '[\x00-\x1F]' -or
                    $seg -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') {
                    throw 'Unsafe ZIP entry component.'
                }
            }
            # Unix POSIX symbolic link and Windows reparse-point metadata.
            $mode = (($entry.ExternalAttributes -shr 16) -band 0xF000)
            if ($mode -eq 0xA000 -or
                (($entry.ExternalAttributes -band 0x400) -ne 0)) {
                throw 'ZIP symlinks and reparse points are forbidden.'
            }
            if ($entry.Length -lt 0 -or $entry.Length -gt 268435456) {
                throw 'ZIP entry exceeds per-file extraction limit.'
            }
            $uncompressed += [long]$entry.Length
            if ($uncompressed -gt 536870912) { throw 'ZIP exceeds extraction size budget.' }
            if ($name.EndsWith('/')) { continue }
            $base = $entry.Name
            if ($base -match '^(?i:llama-cli\.exe|llama\.dll|ggml(?:-base|-cpu(?:-[a-z0-9_]+)?)?\.dll|libomp(?:140(?:\.x86_64)?)?\.dll)$') {
                if (-not $found.Add($base)) { throw 'Duplicate selected ZIP binary filename.' }
                [void]$selected.Add($entry)
            }
        }
        if (-not $found.Contains('llama-cli.exe')) { throw 'Pinned runner ZIP lacks llama-cli.exe.' }
        # Return names, not ZipArchiveEntry instances (archive is disposed on return).
        return @($selected | ForEach-Object { $_.FullName })
    } finally {
        $zip.Dispose()
    }
}

function Expand-Q4PinnedRunner {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$ArchivePath,
          [Parameter(Mandatory)][string]$BinDirectory)

    $selected = @(Test-Q4SafeZipEntries -ArchivePath $ArchivePath)
    if (-not [System.IO.Directory]::Exists($BinDirectory)) { throw 'Bin directory must exist.' }
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $hashes = @()
        foreach ($entryName in $selected) {
            $entry = $zip.GetEntry($entryName)
            $target = [System.IO.Path]::Combine($BinDirectory, $entry.Name)
            if ([System.IO.File]::Exists($target)) { throw 'Runner extraction target already exists.' }
            $source = $entry.Open()
            $sink = $null
            try {
                $sink = [System.IO.FileStream]::new(
                    $target, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write,
                    [System.IO.FileShare]::None
                )
                $source.CopyTo($sink)
                $sink.Flush()
            } finally {
                if ($null -ne $sink) { $sink.Dispose() }
                $source.Dispose()
            }
            if ((Get-Item -LiteralPath $target).Length -ne $entry.Length) {
                throw 'Runner extraction length mismatch.'
            }
            $hashes += [pscustomobject]@{
                File = $entry.Name
                Bytes = [long]$entry.Length
                Sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
        return $hashes
    } finally {
        $zip.Dispose()
    }
}
