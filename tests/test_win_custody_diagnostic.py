"""Synthetic Windows file-custody diagnostic tests. No owner key is inspected."""

import contextlib
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday import win_custody_diagnostic as subject


OWNER = "S-1-5-21-1234-5678-9012-1001"
USERS = "S-1-5-32-545"
EVERYONE = "S-1-1-0"


def allowed(sid: str, mask: int = 0x001F01FF, flags: int = 0):
    return (0, flags, mask, sid)


class PortablePolicyTests(unittest.TestCase):
    def test_local_absolute_path_rejects_ambiguities(self):
        self.assertEqual(subject._segments(r"C:\owner\private\key.txt"),
                         ("C:\\", ("owner", "private", "key.txt")))
        for bad in (
            r"relative\key.txt", r"\\server\share\key.txt",
            r"\\?\C:\owner\key.txt", r"C:relative\key.txt",
            r"C:\owner\..\key.txt", r"C:\owner\.\key.txt",
            r"C:\owner\key.txt:stream", r"C:\owner\key. ",
            r"C:\owner\\key.txt", "C:\\owner\\key.txt\\",
            r"C:\owner\CON.txt", r"C:\owner\LPT1\key.txt",
        ):
            with self.subTest(bad=bad), self.assertRaises(subject.CustodyBlocked):
                subject._segments(bad)

    def test_owner_and_system_acl_pass_without_foreign_grants(self):
        entries = (allowed(OWNER), allowed(subject.SYSTEM_SID))
        subject._acl_allows_only_trusted(owner=OWNER, process_user=OWNER,
                                         aces=entries, is_directory=False)
        subject._acl_allows_only_trusted(owner=OWNER, process_user=OWNER,
                                         aces=entries, is_directory=True)

    def test_admin_is_explicit_privileged_ancestor_only(self):
        entries = (allowed(OWNER), allowed(subject.ADMINISTRATORS_SID))
        subject._acl_allows_only_trusted(
            owner=subject.ADMINISTRATORS_SID, process_user=OWNER,
            aces=entries, is_directory=True,
        )
        with self.assertRaises(subject.CustodyBlocked):
            subject._acl_allows_only_trusted(
                owner=subject.ADMINISTRATORS_SID, process_user=OWNER,
                aces=entries, is_directory=False,
            )

    def test_trustedinstaller_is_known_os_ancestor_not_private_file_owner(self):
        entries = (allowed(OWNER), allowed(subject.TRUSTEDINSTALLER_SID))
        subject._acl_allows_only_trusted(
            owner=subject.TRUSTEDINSTALLER_SID, process_user=OWNER,
            aces=entries, is_directory=True,
        )
        with self.assertRaises(subject.CustodyBlocked):
            subject._acl_allows_only_trusted(
                owner=subject.TRUSTEDINSTALLER_SID, process_user=OWNER,
                aces=entries, is_directory=False,
            )
        near_match = subject.TRUSTEDINSTALLER_SID[:-1] + "5"
        with self.assertRaises(subject.CustodyBlocked):
            subject._acl_allows_only_trusted(
                owner=near_match, process_user=OWNER,
                aces=(allowed(OWNER), allowed(near_match)), is_directory=True,
            )

    def test_foreign_read_is_file_blocker_but_not_ancestor_mutation(self):
        entries = (allowed(OWNER), allowed(USERS, mask=0x00000001))
        subject._acl_allows_only_trusted(owner=OWNER, process_user=OWNER,
                                         aces=entries, is_directory=True)
        with self.assertRaises(subject.CustodyBlocked):
            subject._acl_allows_only_trusted(owner=OWNER, process_user=OWNER,
                                             aces=entries, is_directory=False)

    def test_ancestor_sibling_create_and_inherit_only_are_not_effective_mutation(self):
        for mask, flags in ((0x2, 0), (0x4, 0), (0x6, 0),
                            (0xE0010000, 0x0B)):
            entries = (allowed(OWNER), allowed(USERS, mask=mask, flags=flags))
            with self.subTest(mask=hex(mask), flags=hex(flags)):
                subject._acl_allows_only_trusted(
                    owner=OWNER, process_user=OWNER,
                    aces=entries, is_directory=True,
                )
                if flags:
                    with self.assertRaises(subject.CustodyBlocked):
                        subject._acl_allows_only_trusted(
                            owner=OWNER, process_user=OWNER,
                            aces=entries, is_directory=False,
                        )

    def test_effective_foreign_path_mutation_still_denied_on_ancestor(self):
        for mask in (0x10, 0x40, 0x100, 0x10000, 0x40000, 0x80000,
                     0x10000000, 0x40000000, 0x4 | 0x40):
            entries = (allowed(OWNER), allowed(USERS, mask=mask))
            with self.subTest(mask=hex(mask)), self.assertRaises(subject.CustodyBlocked):
                subject._acl_allows_only_trusted(
                    owner=OWNER, process_user=OWNER,
                    aces=entries, is_directory=True,
                )

    def test_null_empty_untrusted_owner_unknown_aces_denied(self):
        variants = (
            (OWNER, ()),
            (USERS, (allowed(OWNER),)),
            (OWNER, (allowed(OWNER), allowed(EVERYONE, mask=0x80000000))),
            (OWNER, (allowed(OWNER), (5, 0, 0x1, OWNER))),
            (OWNER, (allowed(OWNER), (0, 0, 0x1, 123))),
        )
        for owner, entries in variants:
            with self.subTest(owner=owner, entries=entries), self.assertRaises(subject.CustodyBlocked):
                subject._acl_allows_only_trusted(
                    owner=owner, process_user=OWNER,
                    aces=entries, is_directory=False,
                )

    def test_nonwindows_cli_refuses_and_does_not_log_path_or_contents(self):
        with patch.object(subject.os, "name", "posix"):
            err = StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(subject.main([
                    "check-file", "--kind", "key",
                    "--path", r"C:\owner\synthetic-secret-file.txt",
                ]), 2)
            self.assertNotIn("synthetic-secret-file", err.getvalue())
            self.assertNotIn("owner", err.getvalue())


@unittest.skipUnless(os.name == "nt", "native Win32 ACL tests require Windows")
class NativeWindowsSyntheticFileTests(unittest.TestCase):
    def _fixture(self, root: Path, *, foreign_allow: bool):
        import win32api
        import win32con
        import win32file
        import win32security

        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY,
        )
        try:
            sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        finally:
            token.Close()
        acl = win32security.ACL()
        acl.AddAccessAllowedAce(win32security.ACL_REVISION,
                                win32con.GENERIC_ALL, sid)
        system = win32security.ConvertStringSidToSid(subject.SYSTEM_SID)
        acl.AddAccessAllowedAce(win32security.ACL_REVISION,
                                win32con.GENERIC_ALL, system)
        if foreign_allow:
            world = win32security.ConvertStringSidToSid(EVERYONE)
            acl.AddAccessAllowedAce(win32security.ACL_REVISION,
                                    win32con.GENERIC_READ, world)
        sd = win32security.SECURITY_DESCRIPTOR()
        sd.SetSecurityDescriptorOwner(sid, False)
        sd.SetSecurityDescriptorDacl(True, acl, False)
        sd.SetSecurityDescriptorControl(
            win32security.SE_DACL_PROTECTED, win32security.SE_DACL_PROTECTED,
        )
        sa = win32security.SECURITY_ATTRIBUTES()
        sa.SECURITY_DESCRIPTOR = sd
        sa.bInheritHandle = False
        path = root / ("synthetic-open.txt" if foreign_allow else "synthetic-private.txt")
        handle = win32file.CreateFile(
            str(path), win32con.GENERIC_WRITE, 0, sa,
            win32con.CREATE_NEW, win32con.FILE_ATTRIBUTE_NORMAL, None,
        )
        try:
            status, count = win32file.WriteFile(handle, b"SYNTHETIC_NOT_OWNER_SECRET")
            self.assertEqual(status, 0)
        finally:
            handle.Close()
        return path, win32security.ConvertSidToStringSid(sid)

    def test_handle_bound_private_dacl_and_world_read_rejection(self):
        import win32con
        import win32file
        import win32security
        import ntsecuritycon

        self.assertEqual(subject._FILE_READ_ATTRIBUTES,
                         ntsecuritycon.FILE_READ_ATTRIBUTES)
        self.assertEqual(subject._FILE_ATTRIBUTE_DIRECTORY,
                         win32con.FILE_ATTRIBUTE_DIRECTORY)
        self.assertEqual(subject._FILE_ATTRIBUTE_REPARSE_POINT,
                         win32con.FILE_ATTRIBUTE_REPARSE_POINT)

        with tempfile.TemporaryDirectory(prefix="w8-custody-") as temp:
            for foreign in (False, True):
                path, sid = self._fixture(Path(temp), foreign_allow=foreign)
                handle = win32file.CreateFile(
                    str(path), win32con.READ_CONTROL | subject._FILE_READ_ATTRIBUTES,
                    0, None, win32con.OPEN_EXISTING,
                    win32file.FILE_FLAG_OPEN_REPARSE_POINT, None,
                )
                try:
                    if foreign:
                        with self.assertRaises(subject.CustodyBlocked):
                            subject._verify_descriptor(
                                handle, process_sid=sid, directory=False,
                                security=win32security,
                            )
                    else:
                        subject._verify_descriptor(
                            handle, process_sid=sid, directory=False,
                            security=win32security,
                        )
                finally:
                    handle.Close()

    def test_inherited_foreign_write_on_child_is_not_hidden_by_parent(self):
        # Native, synthetic NTFS inheritance check. Parent has only a harmless
        # foreign ADD_SUBDIRECTORY effective grant, plus an INHERIT_ONLY write
        # grant that MUST be re-evaluated on the existing child directory.
        import win32api
        import win32con
        import win32file
        import win32security

        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY,
        )
        try:
            owner = win32security.GetTokenInformation(
                token, win32security.TokenUser,
            )[0]
        finally:
            token.Close()
        owner_sid = win32security.ConvertSidToStringSid(owner)
        with tempfile.TemporaryDirectory(prefix="w8-native-inheritance-") as temp:
            parent = Path(temp) / "controlled-parent"
            parent.mkdir()
            acl = win32security.ACL()
            acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, 0x03,
                                      win32con.GENERIC_ALL, owner)
            system = win32security.ConvertStringSidToSid(subject.SYSTEM_SID)
            acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, 0x03,
                                      win32con.GENERIC_ALL, system)
            everyone = win32security.ConvertStringSidToSid(EVERYONE)
            acl.AddAccessAllowedAce(win32security.ACL_REVISION, 0x0004, everyone)
            acl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION, 0x0B, win32con.GENERIC_WRITE,
                everyone,
            )
            sd = win32security.SECURITY_DESCRIPTOR()
            sd.SetSecurityDescriptorOwner(owner, False)
            sd.SetSecurityDescriptorDacl(True, acl, False)
            sd.SetSecurityDescriptorControl(
                win32security.SE_DACL_PROTECTED,
                win32security.SE_DACL_PROTECTED,
            )
            win32security.SetFileSecurity(
                str(parent), win32security.DACL_SECURITY_INFORMATION, sd,
            )
            child = parent / "inherited-child"
            child.mkdir()
            for path, should_block in ((parent, False), (child, True)):
                handle = win32file.CreateFile(
                    str(path), win32con.READ_CONTROL | subject._FILE_READ_ATTRIBUTES,
                    win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                    None, win32con.OPEN_EXISTING,
                    win32file.FILE_FLAG_BACKUP_SEMANTICS |
                    win32file.FILE_FLAG_OPEN_REPARSE_POINT, None,
                )
                try:
                    if should_block:
                        descriptor = win32security.GetSecurityInfo(
                            handle, win32security.SE_FILE_OBJECT,
                            win32security.DACL_SECURITY_INFORMATION,
                        )
                        child_acl = descriptor.GetSecurityDescriptorDacl()
                        self.assertIsNotNone(child_acl)
                        self.assertTrue(any(
                            header[0] == subject._ALLOW_ACE
                            and not header[1] & subject._INHERIT_ONLY_ACE
                            and win32security.ConvertSidToStringSid(ace_sid) == EVERYONE
                            and mask & subject._DIRECTORY_MUTATION
                            for header, mask, ace_sid in (
                                child_acl.GetAce(i)
                                for i in range(child_acl.GetAceCount())
                            )
                        ), "child must actually inherit an effective foreign write grant")
                        with self.assertRaises(subject.CustodyBlocked):
                            subject._verify_descriptor(
                                handle, process_sid=owner_sid, directory=True,
                                security=win32security,
                            )
                    else:
                        subject._verify_descriptor(
                            handle, process_sid=owner_sid, directory=True,
                            security=win32security,
                        )
                finally:
                    handle.Close()

    def test_full_read_only_diagnostic_on_synthetic_private_file(self):
        # This exercises the actual drive, ancestor, handle and DACL APIs on
        # Windows CI. A blocked result is a test failure, not a synthetic PASS.
        with tempfile.TemporaryDirectory(prefix="w8-custody-path-") as temp:
            path, _ = self._fixture(Path(temp), foreign_allow=False)
            try:
                result = subject.check_private_file(str(path), max_bytes=514)
            except subject.CustodyBlocked:
                # Ephemeral public Windows CI synthetic-path diagnosis only.
                # Print *only* the ordinal of the first refused ancestor and
                # ACE type/flags/mask/well-known category. Never print names,
                # paths, token/user SIDs, private file bytes or prompt text.
                import sys
                import win32api
                import win32con
                import win32file
                import win32security

                token = win32security.OpenProcessToken(
                    win32api.GetCurrentProcess(), win32con.TOKEN_QUERY,
                )
                try:
                    user = win32security.GetTokenInformation(
                        token, win32security.TokenUser,
                    )[0]
                    process_sid = win32security.ConvertSidToStringSid(user)
                finally:
                    token.Close()
                root, segments = subject._segments(str(path))
                candidate = root
                known = {
                    "S-1-1-0": "Everyone", "S-1-5-11": "AuthenticatedUsers",
                    "S-1-5-32-545": "Users", "S-1-3-0": "CreatorOwner",
                    subject.SYSTEM_SID: "LOCAL_SYSTEM",
                    subject.ADMINISTRATORS_SID: "BUILTIN_ADMINISTRATORS",
                    subject.TRUSTEDINSTALLER_SID: "TRUSTEDINSTALLER",
                    process_sid: "CURRENT_TOKEN_USER",
                }

                def synthetic_sid_category(sid):
                    sid_text = win32security.ConvertSidToStringSid(sid)
                    if sid_text in known:
                        return known[sid_text]
                    try:
                        _name, domain, usage = win32security.LookupAccountSid(None, sid)
                        public_domain = domain if domain in (
                            "NT AUTHORITY", "BUILTIN", "NT SERVICE",
                        ) else "OTHER_DOMAIN"
                        return f"OTHER_SID_TYPE_{int(usage)}_{public_domain}"
                    except Exception:
                        return "OTHER_UNRESOLVED"

                for ordinal in range(len(segments)):
                    if ordinal:
                        candidate = os.path.join(candidate, segments[ordinal - 1])
                    handle = win32file.CreateFile(
                        candidate,
                        win32con.READ_CONTROL | subject._FILE_READ_ATTRIBUTES,
                        win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                        None, win32file.OPEN_EXISTING,
                        win32file.FILE_FLAG_BACKUP_SEMANTICS |
                        win32file.FILE_FLAG_OPEN_REPARSE_POINT, None,
                    )
                    try:
                        try:
                            subject._verify_descriptor(
                                handle, process_sid=process_sid, directory=True,
                                security=win32security,
                            )
                        except subject.CustodyBlocked:
                            desc = win32security.GetSecurityInfo(
                                handle, win32security.SE_FILE_OBJECT,
                                win32security.DACL_SECURITY_INFORMATION |
                                win32security.OWNER_SECURITY_INFORMATION,
                            )
                            acl = desc.GetSecurityDescriptorDacl()
                            print("synthetic_ci_ancestor_owner_class:", ordinal,
                                  synthetic_sid_category(
                                      desc.GetSecurityDescriptorOwner(),
                                  ), file=sys.stderr)
                            for i in range(min(acl.GetAceCount(), 20)):
                                header, mask, sid = acl.GetAce(i)
                                label = synthetic_sid_category(sid)
                                print("synthetic_ci_ancestor_ace:", ordinal,
                                      int(header[0]), int(header[1]),
                                      hex(int(mask)), label, file=sys.stderr)
                            break
                    finally:
                        handle.Close()
                raise
            self.assertEqual(result["status"], "SINGLE_POINT_ACL_SNAPSHOT_PASS")
            self.assertFalse(result["contents_read"])
            self.assertFalse(result["windows_q5_run_qualified"])


if __name__ == "__main__":
    unittest.main()
