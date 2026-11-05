"""Read-only Windows private-file ACL diagnostic; NOT Q5 runtime admission.

This opt-in inspection never reads file contents, opens a model connection, or
creates a transcript.  The Q5 runner's Windows refusal remains authoritative.
An ACL snapshot cannot establish a persistent process lease or rule out an
already-privileged host administrator / same-owner adversary.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import PureWindowsPath
import re
import sys


SYSTEM_SID = "S-1-5-18"
ADMINISTRATORS_SID = "S-1-5-32-544"
CREATOR_OWNER_SID = "S-1-3-0"

# An untrusted principal must not be able to alter a directory entry, change
# its ACL, or replace a path component while a key handle is being inspected.
_DIRECTORY_MUTATION = (
    0x00000002  # FILE_ADD_FILE
    | 0x00000004  # FILE_ADD_SUBDIRECTORY
    | 0x00000010  # FILE_WRITE_EA
    | 0x00000040  # FILE_DELETE_CHILD
    | 0x00000100  # FILE_WRITE_ATTRIBUTES
    | 0x00010000  # DELETE
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x10000000  # GENERIC_ALL
    | 0x40000000  # GENERIC_WRITE
)
_ALLOW_ACE = 0
_DENY_ACE = 1
_INHERIT_ONLY_ACE = 0x08


class CustodyBlocked(ValueError):
    """A conservative policy refusal; error messages never include paths."""


def _segments(path: str) -> tuple[str, tuple[str, ...]]:
    """Reject UNC, device paths, ADS, relative paths, and normalization tricks."""
    if type(path) is not str or not path or path.startswith(("\\\\", "//")):
        raise CustodyBlocked("local drive path required")
    pure = PureWindowsPath(path)
    if (not pure.is_absolute() or not re.fullmatch(r"[A-Za-z]:", pure.drive)
            or path.count(":") != 1 or path.endswith(("\\", "/"))):
        raise CustodyBlocked("absolute local file path required")
    components = path[2:].replace("/", "\\").split("\\")
    if (not components or components[0] != ""
            or any(not part or part in (".", "..") or part.strip(" .") != part
                   for part in components[1:])):
        raise CustodyBlocked("ambiguous path component")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{i}" for i in range(1, 10))
    reserved.update(f"LPT{i}" for i in range(1, 10))
    if any(part.split(".", 1)[0].upper() in reserved
           for part in components[1:]):
        raise CustodyBlocked("reserved Windows device path component")
    return pure.drive.upper() + "\\", tuple(components[1:])


def _acl_allows_only_trusted(
    *, owner: str, process_user: str, aces: tuple[tuple[int, int, int, str], ...],
    is_directory: bool,
) -> None:
    """Conservative ACL classifier; no access-mask/name heuristics.

    ACE input is (type, flags, mask, canonical SID).  On path ancestors,
    foreign READ/TRAVERSE is tolerated but *any mutation grant*, including an
    INHERIT_ONLY child grant, blocks. Admins are an explicit privileged trust
    boundary; they can take ownership even without a discretionary allow ACE.
    For the actual file, all foreign ALLOW ACEs block, including read-only.
    """
    privileged = {process_user, SYSTEM_SID, ADMINISTRATORS_SID}
    if (not owner or not aces
            or (owner != process_user and
                (not is_directory or owner not in privileged))):
        raise CustodyBlocked("private owner/DACL not established")
    for entry in aces:
        if (not isinstance(entry, tuple) or len(entry) != 4
                or type(entry[0]) is not int or type(entry[1]) is not int
                or type(entry[2]) is not int or type(entry[3]) is not str):
            raise CustodyBlocked("unrecognized ACL entry")
        ace_type, flags, mask, sid = entry
        if ace_type not in (_ALLOW_ACE, _DENY_ACE):
            raise CustodyBlocked("unrecognized ACL entry type")
        if ace_type == _DENY_ACE:
            continue  # A deny entry does not grant confidentiality or mutation.
        if sid in privileged:
            continue
        if (is_directory and sid == CREATOR_OWNER_SID
                and flags & _INHERIT_ONLY_ACE):
            # It grants to a future child's creator, not foreign directory
            # mutation. Foreign creation was separately blocked above.
            continue
        if not is_directory or mask & _DIRECTORY_MUTATION:
            raise CustodyBlocked("untrusted ACL access grant")


def _windows_dependencies():
    if os.name != "nt":
        raise CustodyBlocked("native Windows required")
    try:
        import win32api  # type: ignore[import-not-found]
        import win32con  # type: ignore[import-not-found]
        import win32file  # type: ignore[import-not-found]
        import win32security  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CustodyBlocked("native Windows security API unavailable") from exc
    return win32api, win32con, win32file, win32security


def _verify_descriptor(handle, *, process_sid: str, directory: bool, security) -> None:
    descriptor = security.GetSecurityInfo(
        handle, security.SE_FILE_OBJECT,
        security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION,
    )
    owner = descriptor.GetSecurityDescriptorOwner()
    dacl = descriptor.GetSecurityDescriptorDacl()
    if owner is None or dacl is None:
        raise CustodyBlocked("owner or DACL unavailable")
    owner_text = security.ConvertSidToStringSid(owner)
    entries: list[tuple[int, int, int, str]] = []
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        if (type(ace) is not tuple or len(ace) != 3
                or type(ace[0]) is not tuple or len(ace[0]) != 2):
            raise CustodyBlocked("complex ACL entry unsupported")
        header, mask, sid = ace
        entries.append((int(header[0]), int(header[1]), int(mask),
                        security.ConvertSidToStringSid(sid)))
    _acl_allows_only_trusted(
        owner=owner_text, process_user=process_sid,
        aces=tuple(entries), is_directory=directory,
    )


def check_private_file(path: str, *, max_bytes: int = 514) -> dict:
    """Inspect same-handle file identity, owner and DACL without reading bytes.

    This is a *single-point diagnostic*, never a credential-loading primitive
    or a permission to bypass the Q5 Windows source gate.
    """
    if type(max_bytes) is not int or not 1 <= max_bytes <= 12000:
        raise CustodyBlocked("invalid diagnostic size bound")
    root, components = _segments(path)
    api, con, fileapi, security = _windows_dependencies()
    if (fileapi.GetDriveType(root) != fileapi.DRIVE_FIXED
            or api.GetVolumeInformation(root)[4].upper() != "NTFS"):
        raise CustodyBlocked("fixed local NTFS volume required")
    target = fileapi.QueryDosDevice(root[:2]).split("\0", 1)[0]
    if not target.startswith("\\Device\\HarddiskVolume"):
        raise CustodyBlocked("drive substitution or unsupported volume")
    token = security.OpenProcessToken(api.GetCurrentProcess(), con.TOKEN_QUERY)
    try:
        user = security.GetTokenInformation(token, security.TokenUser)
        process_sid = security.ConvertSidToStringSid(user[0] if type(user) is tuple else user)
    finally:
        token.Close()

    flags = fileapi.FILE_FLAG_OPEN_REPARSE_POINT
    read_control = con.READ_CONTROL | con.FILE_READ_ATTRIBUTES
    # Keep every ancestor handle open until the target has been inspected;
    # omitting FILE_SHARE_DELETE prevents ordinary rename/delete races.
    with ExitStack() as held:
        root_handle = fileapi.CreateFile(
            root, read_control, fileapi.FILE_SHARE_READ | fileapi.FILE_SHARE_WRITE,
            None, fileapi.OPEN_EXISTING,
            flags | fileapi.FILE_FLAG_BACKUP_SEMANTICS, None,
        )
        held.callback(root_handle.Close)
        root_info = fileapi.GetFileInformationByHandle(root_handle)
        if (root_info[0] & fileapi.FILE_ATTRIBUTE_REPARSE_POINT
                or not root_info[0] & fileapi.FILE_ATTRIBUTE_DIRECTORY):
            raise CustodyBlocked("reparse or non-directory drive root")
        _verify_descriptor(root_handle, process_sid=process_sid,
                           directory=True, security=security)
        current = root
        for component in components[:-1]:
            current += component
            handle = fileapi.CreateFile(
                current, read_control, fileapi.FILE_SHARE_READ | fileapi.FILE_SHARE_WRITE,
                None, fileapi.OPEN_EXISTING,
                flags | fileapi.FILE_FLAG_BACKUP_SEMANTICS, None,
            )
            held.callback(handle.Close)
            info = fileapi.GetFileInformationByHandle(handle)
            if (info[0] & fileapi.FILE_ATTRIBUTE_REPARSE_POINT
                    or not info[0] & fileapi.FILE_ATTRIBUTE_DIRECTORY):
                raise CustodyBlocked("non-directory or reparse ancestor")
            _verify_descriptor(handle, process_sid=process_sid,
                               directory=True, security=security)
            current += "\\"
        target_handle = fileapi.CreateFile(
            path, read_control, 0, None, fileapi.OPEN_EXISTING, flags, None,
        )
        held.callback(target_handle.Close)
        info = fileapi.GetFileInformationByHandle(target_handle)
        if (info[0] & fileapi.FILE_ATTRIBUTE_REPARSE_POINT
                or info[0] & fileapi.FILE_ATTRIBUTE_DIRECTORY
                or info[7] != 1):
            raise CustodyBlocked("reparse, directory or multiply-linked file")
        length = (info[5] << 32) | info[6]
        if not 1 <= length <= max_bytes:
            raise CustodyBlocked("file size outside diagnostic bound")
        _verify_descriptor(target_handle, process_sid=process_sid,
                           directory=False, security=security)
    return {
        "status": "SINGLE_POINT_ACL_SNAPSHOT_PASS",
        "contents_read": False,
        "model_inference_performed": False,
        "windows_q5_run_qualified": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check-file", choices=["check-file"])
    parser.add_argument("--path", required=True)
    parser.add_argument("--kind", choices=["key", "prompt"], default="key")
    args = parser.parse_args(argv)
    try:
        result = check_private_file(
            args.path, max_bytes=514 if args.kind == "key" else 12000,
        )
    except Exception:
        # Error codes, paths, DACL data and owner metadata do not enter logs.
        print("custody blocked", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
