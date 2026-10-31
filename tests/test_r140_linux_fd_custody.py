"""CPU-only Linux tests; run with ``python -m unittest ...`` (no pytest)."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday import r140_linux_fd_custody as custody


def _stage(output: Path, *, step: int = 3, suffix: str = "abc", state_override=None) -> str:
    name = f".checkpoint-{step}-staging-{suffix}"
    directory = output / name
    directory.mkdir()
    weights = b"synthetic-adapter-bytes\x01"
    (directory / "lora-trainables.safetensors").write_bytes(weights)
    state = {
        "schema": "R18_MEZO_STATE_V1",
        "status": "CHECKPOINT_BYTES_PINNED_NOT_RELOADED",
        "completed_steps": step,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "run_contract_sha256": "a" * 64,
        "optimizer_window_seconds": 3.0,
        "counters": {"forward_passes": 2},
    }
    if state_override:
        state.update(state_override)
    (directory / "state.json").write_text(json.dumps(state))
    return name


def _publish(output: Path, stage: str, *, step: int = 3):
    fd = custody.open_linux_directory_no_symlinks(output)
    try:
        return custody.publish_prepared_linux(fd, stage_name=stage, step=step)
    finally:
        os.close(fd)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux descriptor prototype")
class LinuxFdSourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_source_receipt_never_authorizes_training_or_resume(self):
        stage = _stage(self.path)
        receipt = _publish(self.path, stage)
        self.assertEqual(receipt["status"], "SOURCE_ONLY_FD_PINNED_NOT_RELOADED")
        for key in ("authorizes_training", "authorizes_resume", "physical_durability_verified",
                    "physical_reload_verified", "host_authority_verified"):
            self.assertIs(receipt[key], False, key)
        checkpoint = self.path / "checkpoint-3"
        self.assertTrue(checkpoint.is_dir())
        self.assertFalse((self.path / stage).exists())
        marker = (checkpoint / "commit.json").read_bytes()
        self.assertEqual(hashlib.sha256(marker).hexdigest(), receipt["checkpoint_commit_sha256"])

    def test_existing_destination_never_replaced(self):
        (self.path / "checkpoint-3").mkdir()
        sentinel = self.path / "checkpoint-3" / "important"
        sentinel.write_text("must remain")
        stage = _stage(self.path)
        with self.assertRaisesRegex(custody.LinuxCustodyError, "DESTINATION_ALREADY_EXISTS"):
            _publish(self.path, stage)
        self.assertEqual(sentinel.read_text(), "must remain")
        self.assertTrue((self.path / stage).exists())

    def test_rejects_symlink_stage_member(self):
        stage = _stage(self.path)
        member = self.path / stage / "state.json"
        copy = self.path / "external.json"
        member.rename(copy)
        member.symlink_to(copy)
        with self.assertRaisesRegex(custody.LinuxCustodyError, "MEMBER_OPEN_UNSAFE"):
            _publish(self.path, stage)
        self.assertFalse((self.path / "checkpoint-3").exists())

    def test_rejects_hardlink_alias(self):
        stage = _stage(self.path)
        os.link(self.path / stage / "state.json", self.path / "extra-link")
        with self.assertRaisesRegex(custody.LinuxCustodyError, "MEMBER_TYPE_OR_SIZE_INVALID"):
            _publish(self.path, stage)

    def test_rejects_symlink_ancestor(self):
        root = self.path / "real"
        root.mkdir()
        (self.path / "alias").symlink_to(root, target_is_directory=True)
        with self.assertRaisesRegex(custody.LinuxCustodyError, "OUTPUT_COMPONENT_UNSAFE"):
            custody.open_linux_directory_no_symlinks(self.path / "alias")

    def test_rejects_mismatched_source_state(self):
        stage = _stage(self.path, state_override={"completed_steps": 4})
        with self.assertRaisesRegex(custody.LinuxCustodyError, "STATE_CONTRACT_MISMATCH"):
            _publish(self.path, stage)

    def test_rejects_nested_nonfinite_json_number(self):
        stage = _stage(self.path)
        (self.path / stage / "state.json").write_text(
            '{"schema":"R18_MEZO_STATE_V1","status":"CHECKPOINT_BYTES_PINNED_NOT_RELOADED",'
            '"completed_steps":3,"weights_sha256":"' + hashlib.sha256(b"synthetic-adapter-bytes\x01").hexdigest()
            + '","run_contract_sha256":"' + 'a' * 64
            + '","optimizer_window_seconds":3,"counters":{"overflow":1e309}}'
        )
        with self.assertRaisesRegex(custody.LinuxCustodyError, "STATE_JSON_INVALID"):
            _publish(self.path, stage)

    def test_rejects_huge_json_integer_without_overflow_escape(self):
        stage = _stage(self.path, state_override={
            "optimizer_window_seconds": 10**1000,
        })
        with self.assertRaisesRegex(custody.LinuxCustodyError, "STATE_CONTRACT_MISMATCH"):
            _publish(self.path, stage)

    def test_rejects_duplicate_json_key(self):
        stage = _stage(self.path)
        member = self.path / stage / "state.json"
        member.write_bytes(member.read_bytes().replace(b'"completed_steps": 3', b'"completed_steps": 3, "completed_steps": 3'))
        with self.assertRaisesRegex(custody.LinuxCustodyError, "STATE_JSON_INVALID"):
            _publish(self.path, stage)

    def test_output_fd_remains_bound_if_path_rebound(self):
        original = self.path / "owned"
        decoy = self.path / "decoy"
        original.mkdir()
        decoy.mkdir()
        stage = _stage(original)
        fd = custody.open_linux_directory_no_symlinks(original)
        try:
            original.rename(self.path / "moved")
            original.symlink_to(decoy, target_is_directory=True)
            receipt = custody.publish_prepared_linux(fd, stage_name=stage, step=3)
        finally:
            os.close(fd)
        self.assertEqual(receipt["checkpoint_name"], "checkpoint-3")
        self.assertTrue((self.path / "moved" / "checkpoint-3").is_dir())
        self.assertFalse((decoy / "checkpoint-3").exists())

    def test_swapped_stage_name_after_validation_fails_ambiguous(self):
        stage = _stage(self.path)
        malicious = self.path / "attacker"
        malicious.mkdir()
        real_rename = custody._rename_noreplace

        def swap_before_rename(dirfd, source, destination):
            os.rename(source, "sequestered-stage", src_dir_fd=dirfd, dst_dir_fd=dirfd)
            os.symlink(malicious, source, dir_fd=dirfd)
            real_rename(dirfd, source, destination)

        with patch.object(custody, "_rename_noreplace", swap_before_rename):
            with self.assertRaisesRegex(custody.LinuxCustodyError, "POST_RENAME_CUSTODY_UNKNOWN"):
                _publish(self.path, stage)

    def test_rename_reports_failure_after_kernel_commit_holds_unknown(self):
        stage = _stage(self.path)
        real_rename = custody._rename_noreplace

        def commit_then_fail(dirfd, source, destination):
            real_rename(dirfd, source, destination)
            raise OSError("injected dispatch interruption after mutation")

        with patch.object(custody, "_rename_noreplace", commit_then_fail):
            with self.assertRaisesRegex(custody.LinuxCustodyError, "POST_RENAME_CUSTODY_UNKNOWN"):
                _publish(self.path, stage)
        self.assertTrue((self.path / "checkpoint-3").is_dir())

    def test_same_bytes_different_inode_after_rename_is_not_custody(self):
        stage = _stage(self.path)
        real_readback = custody._readback
        calls = 0

        def replace_same_bytes_before_target_readback(dirfd, expected, member_ids):
            nonlocal calls
            calls += 1
            if calls == 2:
                state_fd = os.open("state.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dirfd)
                try:
                    raw = os.read(state_fd, 1024 * 1024)
                finally:
                    os.close(state_fd)
                os.unlink("state.json", dir_fd=dirfd)
                new_fd = os.open("state.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=dirfd)
                try:
                    os.write(new_fd, raw)
                finally:
                    os.close(new_fd)
            return real_readback(dirfd, expected, member_ids)

        with patch.object(custody, "_readback", replace_same_bytes_before_target_readback):
            with self.assertRaisesRegex(custody.LinuxCustodyError, "POST_RENAME_CUSTODY_UNKNOWN"):
                _publish(self.path, stage)
        self.assertTrue((self.path / "checkpoint-3").is_dir())

    def test_fails_closed_for_relative_and_wrong_output_descriptor(self):
        with self.assertRaisesRegex(custody.LinuxCustodyError, "OUTPUT_PATH_INVALID"):
            custody.open_linux_directory_no_symlinks(Path("relative"))
        fd = os.open(self.path / "not-dir", os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            with self.assertRaisesRegex(custody.LinuxCustodyError, "OUTPUT_DESCRIPTOR_NOT_DIRECTORY"):
                custody.publish_prepared_linux(fd, stage_name=".checkpoint-3-staging-abc", step=3)
        finally:
            os.close(fd)

    def test_rejects_5000_digit_stage_without_integer_conversion_error(self):
        fd = custody.open_linux_directory_no_symlinks(self.path)
        try:
            stage = ".checkpoint-" + "1" * 5000 + "-staging-abc"
            with self.assertRaisesRegex(custody.LinuxCustodyError, "PUBLICATION_REQUEST_INVALID"):
                custody.publish_prepared_linux(fd, stage_name=stage, step=3)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
