function Test-TrainingDayNoUnexpectedWriters {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$SecurityDescriptor)

    # Exclude GenericRead/ReadAndExecute/Synchronize. FullControl contains
    # read bits and cannot itself be ORed into a mutating-rights mask.
    $mutatingRights = [Security.AccessControl.FileSystemRights]::Write -bor
        [Security.AccessControl.FileSystemRights]::Delete -bor
        [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
        [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
        [Security.AccessControl.FileSystemRights]::TakeOwnership
    $trustedSids = @('S-1-5-18', 'S-1-5-32-544')
    try {
        $trustedInstallerSid = ([Security.Principal.NTAccount]::new('NT SERVICE\TrustedInstaller')).Translate(
            [Security.Principal.SecurityIdentifier]).Value
        $trustedSids += $trustedInstallerSid
    } catch { return $false }

    try {
        foreach ($ace in $SecurityDescriptor.Access) {
            if ($ace.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) { continue }
            if (($ace.FileSystemRights -band $mutatingRights) -eq 0) { continue }
            $sid = $ace.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
            if ($sid -notin $trustedSids) { return $false }
        }
        return $true
    } catch { return $false }
}
