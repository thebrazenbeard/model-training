#Requires -Version 5.1
<#
.SYNOPSIS
Stages exactly two pinned public artifacts for an isolated, non-executing VN1 CPU trial.
.DESCRIPTION
Creates ONE new leaf inside the current user's profile. The parent must already exist.
Downloads with fixed HTTPS URLs into uniquely named .partial files, checks length and
SHA-256 BEFORE moving into place, validates all ZIP entry paths and metadata, extracts
only a fixed allowlist of CPU runner binaries, and writes a custody manifest.
Does NOT launch a model, install a service, or modify shared directories.
Requires separate host-owned supervisor approval and verification before inference.
#>
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Destination)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'Windows PowerShell 5.1 on an authorized Windows host is required.'
}
. (Join-Path $PSScriptRoot 'Q4Stager.Core.ps1')

# Immutable content pins; the Hugging Face "main" URL may move but cannot pass
# verification with different bytes. Derivation from official Qwen shards is NOT proven.
$runner = [pscustomobject]@{
    Name = 'llama-b12125-bin-win-cpu-x64.zip'
    Url = [uri]'https://github.com/ggml-org/llama.cpp/releases/download/b12125/llama-b12125-bin-win-cpu-x64.zip'
    Bytes = [long]24730921
    Sha256 = '968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8'
    Revision = '7d516e25ead4653169236b3f90d13f7bd8629724'
}
$model = [pscustomobject]@{
    Name = 'Qwen3.5-4B-Q4_K_M.gguf'
    Url = [uri]'https://huggingface.co/twoody/Qwen3.5-4B-GGUF/resolve/d741a25c4cf8eea74fe876af7631b659b366f126/Qwen3.5-4B-Q4_K_M.gguf'
    Bytes = [long]2783446880
    Sha256 = 'b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca'
    Provenance = 'publisher digest only; official Qwen derivation UNVERIFIED'
}

$full = Assert-Q4IsolatedDestination -Destination $Destination
$volume = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($full))
$minFree = $runner.Bytes + $model.Bytes + [long]1073741824
if (-not $volume.IsReady -or $volume.AvailableFreeSpace -lt $minFree) {
    throw 'Insufficient available disk space for the pinned downloads and safety reserve.'
}

# The leaf is new; no pre-existing user files can be overwritten.
# Atomic fail-if-exists: if another process wins this exact leaf, do NOT
# touch its ACL, content, or descendants. This is the only caller-created leaf.
New-Q4ExclusiveLeaf -Path $full
Set-Q4PrivateDirectory -Directory $full
$bin = [System.IO.Path]::Combine($full, 'bin')
$models = [System.IO.Path]::Combine($full, 'models')
$downloads = [System.IO.Path]::Combine($full, 'downloads')
$state = [System.IO.Path]::Combine($full, 'state')
foreach ($dir in @($bin, $models, $downloads, $state)) {
    [void][System.IO.Directory]::CreateDirectory($dir)
    Set-Q4PrivateDirectory -Directory $dir
}

$runnerFile = [System.IO.Path]::Combine($downloads, $runner.Name)
$modelFile = [System.IO.Path]::Combine($models, $model.Name)
try {
    $runnerReceipt = Save-Q4PinnedDownload -Uri $runner.Url -Target $runnerFile -ExpectedBytes $runner.Bytes -ExpectedSha256 $runner.Sha256
    $runnerHashes = @(Expand-Q4PinnedRunner -ArchivePath $runnerFile -BinDirectory $bin)
    $modelReceipt = Save-Q4PinnedDownload -Uri $model.Url -Target $modelFile -ExpectedBytes $model.Bytes -ExpectedSha256 $model.Sha256

    $receipt = [ordered]@{
        Schema = 'bt2.q4.stage.receipt.v1'
        Utc = [DateTime]::UtcNow.ToString('o')
        Host = [Environment]::MachineName
        User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        Destination = $full
        Runner = [ordered]@{ Name = $runner.Name; Bytes = $runnerReceipt.Bytes; Sha256 = $runnerReceipt.Sha256; UpstreamCommit = $runner.Revision }
        Binaries = @($runnerHashes)
        Model = [ordered]@{ Name = $model.Name; Bytes = $modelReceipt.Bytes; Sha256 = $modelReceipt.Sha256; Provenance = $model.Provenance }
        Executed = $false
        LeaseQualified = $false
    }
    $receiptPath = [System.IO.Path]::Combine($full, 'stage-receipt.json')
    $json = $receipt | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($receiptPath, $json, [System.Text.UTF8Encoding]::new($false))
    Write-Output ('STAGED_AND_HASH_VERIFIED: ' + $receiptPath)
    Write-Output 'NO_EXECUTION: supervisor lease remains a separate gate.'
} catch {
    # Retain only this newly created isolated leaf for forensic inspection.
    # Never recursively delete or overwrite it automatically.
    throw 'Staging did not complete; inspect the isolated destination. Nothing was executed.'
}
