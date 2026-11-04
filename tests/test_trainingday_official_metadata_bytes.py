"""Strict offline tests for source-only Qwen3.5 metadata identity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import trainingday.official_metadata_bytes as metadata_module
from trainingday.official_metadata_bytes import (
    OFFICIAL_METADATA_BYTE_PINS, OfficialMetadataBytesError,
    _verify_bytes_against_pins, inspect_official_metadata_bytes_source_only,
    main,
)
from trainingday.official_public_base import (
    METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS, OFFICIAL_REVISION,
)
from trainingday.pinned_base import TRUSTED_OFFICIAL_METADATA_PIN_ITEMS


class OfficialMetadataBytesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {
            "config.json": b"TEST ONLY: config version A!",
            "tokenizer.json": b"TEST ONLY: tokenizer version A!",
        }
        for name, payload in self.files.items():
            (self.root / name).write_bytes(payload)
        self.pins = tuple((name, len(value), hashlib.sha256(value).hexdigest())
                          for name, value in self.files.items())

    def test_exact_synthetic_bytes_pass_test_seam_only(self):
        self.assertEqual(_verify_bytes_against_pins(self.root, self.pins),
                         tuple(self.files))
        with self.assertRaises(OfficialMetadataBytesError):
            inspect_official_metadata_bytes_source_only(self.root)

    def test_synthetic_success_receipt_never_claims_atomicity_or_admission(self):
        # Temporarily replace official pins in the isolated test process only;
        # this is not a legitimate official upstream metadata validation.
        with patch.object(metadata_module, "OFFICIAL_METADATA_BYTE_PINS", self.pins):
            with patch.object(metadata_module, "METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS",
                              tuple(self.files)):
                receipt = inspect_official_metadata_bytes_source_only(self.root)
        self.assertEqual(receipt["metadata_byte_matches"], list(self.files))
        for field in ("atomic_immutable_snapshot_verified",
                      "host_path_custody_verified", "shard_bytes_verified",
                      "owner_signature_verified", "training_admission_granted",
                      "physical_launch_allowed"):
            self.assertIs(receipt[field], False, field)

    def test_single_byte_tamper_with_same_size_fails(self):
        path = self.root / "tokenizer.json"
        path.write_bytes(path.read_bytes().replace(b"version A", b"version B"))
        self.assertEqual(path.stat().st_size, self.pins[1][1])
        with self.assertRaisesRegex(OfficialMetadataBytesError, "incorrect metadata"):
            _verify_bytes_against_pins(self.root, self.pins)

    def test_missing_and_truncated_metadata_fail(self):
        (self.root / "config.json").unlink()
        with self.assertRaises(OfficialMetadataBytesError):
            _verify_bytes_against_pins(self.root, self.pins)
        (self.root / "config.json").write_bytes(b"short")
        with self.assertRaisesRegex(OfficialMetadataBytesError, "wrong size"):
            _verify_bytes_against_pins(self.root, self.pins)

    def test_symlink_file_and_symlinked_snapshot_fail(self):
        original = self.root / "tokenizer.json"
        alias_target = self.root / "actual-tokenizer"
        original.rename(alias_target)
        original.symlink_to(alias_target)
        with self.assertRaises(OfficialMetadataBytesError):
            _verify_bytes_against_pins(self.root, self.pins)
        original.unlink()
        original.write_bytes(self.files["tokenizer.json"])
        alias = self.root / "alias-directory"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OfficialMetadataBytesError):
            _verify_bytes_against_pins(alias, self.pins)

    def test_symlink_swap_of_ancestor_during_secure_open_fails(self):
        checked = self.root / "parent" / "snapshot"
        checked.mkdir(parents=True)
        for name, content in self.files.items():
            (checked / name).write_bytes(content)
        target = self.root / "elsewhere"
        target.mkdir()
        original_open = os.open
        swapped = False

        def swap_just_before_ancestor_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path == "parent" and "dir_fd" in kwargs and not swapped:
                swapped = True
                (self.root / "parent").rename(self.root / "parent-real")
                (self.root / "parent").symlink_to(target, target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)

        try:
            with patch("trainingday.official_metadata_bytes.os.open",
                       side_effect=swap_just_before_ancestor_open):
                with self.assertRaises(OfficialMetadataBytesError):
                    _verify_bytes_against_pins(checked, self.pins)
            self.assertTrue(swapped)
        finally:
            if swapped:
                (self.root / "parent").unlink()
                (self.root / "parent-real").rename(self.root / "parent")

    def test_root_swapped_to_symlink_after_anchor_does_not_false_pass(self):
        checked = self.root / "snapshot"
        checked.mkdir()
        for name, content in self.files.items():
            (checked / name).write_bytes(content)
        original_open = os.open
        swapped = False

        def swap_before_first_file_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path == "config.json" and "dir_fd" in kwargs and not swapped:
                swapped = True
                checked.rename(self.root / "snapshot-moved")
                checked.symlink_to(self.root / "snapshot-moved", target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)

        try:
            with patch("trainingday.official_metadata_bytes.os.open",
                       side_effect=swap_before_first_file_open):
                with self.assertRaises(OfficialMetadataBytesError):
                    _verify_bytes_against_pins(checked, self.pins)
            self.assertTrue(swapped)
        finally:
            if swapped:
                checked.unlink()
                (self.root / "snapshot-moved").rename(checked)

    def test_disappearing_earlier_file_fails_final_revalidation(self):
        original_open = os.open
        removed = False

        def unlink_first_before_opening_second(path, flags, *args, **kwargs):
            nonlocal removed
            if path == "tokenizer.json" and "dir_fd" in kwargs and not removed:
                removed = True
                (self.root / "config.json").unlink()
            return original_open(path, flags, *args, **kwargs)

        with patch("trainingday.official_metadata_bytes.os.open",
                   side_effect=unlink_first_before_opening_second):
            with self.assertRaisesRegex(OfficialMetadataBytesError, "disappeared"):
                _verify_bytes_against_pins(self.root, self.pins)
        self.assertTrue(removed)

    def test_same_byte_replacement_of_earlier_file_detected(self):
        original_open = os.open
        replaced = False

        def replace_first_before_opening_second(path, flags, *args, **kwargs):
            nonlocal replaced
            if path == "tokenizer.json" and "dir_fd" in kwargs and not replaced:
                replaced = True
                (self.root / "config.json").rename(self.root / "old-config")
                (self.root / "config.json").write_bytes(self.files["config.json"])
            return original_open(path, flags, *args, **kwargs)

        with patch("trainingday.official_metadata_bytes.os.open",
                   side_effect=replace_first_before_opening_second):
            with self.assertRaisesRegex(OfficialMetadataBytesError, "disappeared"):
                _verify_bytes_against_pins(self.root, self.pins)
        self.assertTrue(replaced)

    def test_unavailable_secure_dirfd_fails_closed(self):
        with patch("trainingday.official_metadata_bytes._SECURE_DIRFD_SUPPORTED", False):
            with self.assertRaisesRegex(OfficialMetadataBytesError, "unavailable"):
                _verify_bytes_against_pins(self.root, self.pins)

    def test_nonregular_file_is_rejected_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO unavailable")
        candidate = self.root / "tokenizer.json"
        candidate.unlink()
        os.mkfifo(candidate)
        with self.assertRaises(OfficialMetadataBytesError):
            _verify_bytes_against_pins(self.root, self.pins)

    def test_invalid_or_duplicated_pins_fail(self):
        for pins in (
            (),
            self.pins + (self.pins[0],),
            (("../escape", 3, "a" * 64),),
            (("tokenizer.json", True, "a" * 64),),
            (("tokenizer.json", -1, "a" * 64),),
            (("tokenizer.json", 2, "A" * 64),),
            (("tokenizer.json", 999999999, "a" * 64),),
        ):
            with self.subTest(pins=pins):
                with self.assertRaises(OfficialMetadataBytesError):
                    _verify_bytes_against_pins(self.root, pins)

    def test_relative_paths_fail(self):
        with self.assertRaisesRegex(OfficialMetadataBytesError, "real absolute"):
            _verify_bytes_against_pins("relative/to/workspace", self.pins)

    def test_official_pin_inventory_is_stable_but_not_admitted(self):
        self.assertEqual(tuple(name for name, _, _ in OFFICIAL_METADATA_BYTE_PINS),
                         METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS)
        self.assertEqual(len(OFFICIAL_METADATA_BYTE_PINS), 7)
        self.assertEqual(OFFICIAL_REVISION,
                         "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(OFFICIAL_METADATA_BYTE_PINS[2], (
            "tokenizer.json", 12807982,
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"))
        self.assertEqual(OFFICIAL_METADATA_BYTE_PINS[3], (
            "model.safetensors.index.json", 76196,
            "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"))
        self.assertEqual(TRUSTED_OFFICIAL_METADATA_PIN_ITEMS, ())

    def test_lfs_pointer_is_not_tokenizer_object(self):
        # The /raw tokenizer path returns this 133-byte Git LFS pointer
        # shape, not the 12.8 MB tokenizer required for a source-byte match.
        pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                   b"oid sha256:" + b"5f9e" + b"0" * 59 + b"\nsize 12807982\n")
        (self.root / "tokenizer.json").write_bytes(pointer)
        with self.assertRaises(OfficialMetadataBytesError):
            inspect_official_metadata_bytes_source_only(self.root)

    def test_cli_rejects_without_launch_flag(self):
        with patch("builtins.print") as output:
            status = main(["--snapshot", str(self.root)])
        self.assertEqual(status, 2)
        receipt = json.loads(output.call_args.args[0])
        self.assertFalse(receipt["physical_launch_allowed"])
        self.assertEqual(receipt["status"], "HOLD")

    def test_cli_directory_fstat_failures_emit_structured_hold(self):
        # First observation anchors the opened snapshot; second revalidates
        # it after all file descriptors have been checked. Both are fallible.
        original_fstat = os.fstat
        for failing_directory_observation in (1, 2):
            directory_observations = 0

            def injected_fstat(fd):
                nonlocal directory_observations
                info = original_fstat(fd)
                if stat.S_ISDIR(info.st_mode):
                    directory_observations += 1
                    if directory_observations == failing_directory_observation:
                        raise OSError("injected directory fstat failure")
                return info

            with self.subTest(failing_directory_observation=failing_directory_observation):
                with patch.object(metadata_module, "OFFICIAL_METADATA_BYTE_PINS", self.pins):
                    with patch.object(metadata_module,
                                      "METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS",
                                      tuple(self.files)):
                        with patch("trainingday.official_metadata_bytes.os.fstat",
                                   side_effect=injected_fstat), patch("builtins.print") as output:
                            status = main(["--snapshot", str(self.root)])
                self.assertEqual(directory_observations,
                                 failing_directory_observation)
                self.assertEqual(status, 2)
                receipt = json.loads(output.call_args.args[0])
                self.assertEqual(receipt["schema"], metadata_module.SCHEMA)
                self.assertEqual(receipt["status"], "HOLD")
                self.assertFalse(receipt["physical_launch_allowed"])
                self.assertIn("directory", receipt["error"].lower())

    def test_cli_exitstack_close_failure_emits_structured_hold(self):
        original_fstat = os.fstat
        original_close = os.close
        directory_observations = 0
        failed_cleanup = False

        def observe_directory_revalidation(fd):
            nonlocal directory_observations
            info = original_fstat(fd)
            if stat.S_ISDIR(info.st_mode):
                directory_observations += 1
            return info

        def close_and_fail_once(fd):
            nonlocal failed_cleanup
            original_close(fd)
            # This only fires in ExitStack after the *second* directory fstat.
            # Close first to avoid leaking the injected-failure descriptor.
            if directory_observations == 2 and not failed_cleanup:
                failed_cleanup = True
                raise OSError("injected descriptor cleanup failure")

        with patch.object(metadata_module, "OFFICIAL_METADATA_BYTE_PINS", self.pins):
            with patch.object(metadata_module, "METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS",
                              tuple(self.files)):
                with patch("trainingday.official_metadata_bytes.os.fstat",
                           side_effect=observe_directory_revalidation):
                    with patch("trainingday.official_metadata_bytes.os.close",
                               side_effect=close_and_fail_once), patch("builtins.print") as output:
                        status = main(["--snapshot", str(self.root)])
        self.assertTrue(failed_cleanup)
        self.assertEqual(directory_observations, 2)
        self.assertEqual(status, 2)
        receipt = json.loads(output.call_args.args[0])
        self.assertEqual(receipt["status"], "HOLD")
        self.assertFalse(receipt["physical_launch_allowed"])


if __name__ == "__main__":
    unittest.main()
