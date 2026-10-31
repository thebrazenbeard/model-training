"""Source-only checkpoint publication boundary: not a training/resume test."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from trainingday import r18_checkpoint_commit as checkpoint
except ImportError:
    checkpoint = None


class CheckpointPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def _write_valid(self, stage: Path, *, payload: bytes = b"synthetic-adapter-weights") -> None:
        (stage / "lora-trainables.safetensors").write_bytes(payload)
        state = {
            "schema": "R18_MEZO_STATE_V1",
            "status": "CHECKPOINT_BYTES_PINNED_NOT_RELOADED",
            "completed_steps": 3,
            "run_contract_sha256": "a" * 64,
            "weights_sha256": hashlib.sha256(payload).hexdigest(),
            "optimizer_window_seconds": 10.5,
            "counters": {"forward_evaluations": 6},
        }
        (stage / "state.json").write_text(json.dumps(state), encoding="utf-8")

    def test_readback_rejects_path_subclass_with_forged_receipt_string(self) -> None:
        """One path object must not read bytes from A while attesting path B."""
        real = self.output / "checkpoint-3"
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )

        class DeceptivePath(type(Path())):
            def __str__(self) -> str:
                return "/unowned/forged/checkpoint-3"

            def __fspath__(self) -> str:
                return type(Path()).__str__(self)

        deceptive = DeceptivePath(real)
        self.assertEqual(os.fspath(deceptive), str(real))
        self.assertNotEqual(str(deceptive), str(real))
        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.verify_published_checkpoint(
                deceptive, expected_step=3,
                expected_commit_sha256=receipt["checkpoint_commit_sha256"],
            )
        self.assertEqual(caught.exception.reason, "READBACK_REQUEST_INVALID")

    def test_publisher_rejects_path_subclass_with_forged_receipt_string(self) -> None:
        """No successful publication receipt may refer to a phantom output."""
        real = self.output / "output"
        real.mkdir()

        class DeceptivePath(type(Path())):
            def __str__(self) -> str:
                return "/unowned/forged/output"

            def __fspath__(self) -> str:
                return type(Path()).__str__(self)

        deceptive = DeceptivePath(real)
        self.assertEqual(os.fspath(deceptive), str(real))
        self.assertNotEqual(str(deceptive), str(real))
        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.publish_checkpoint_atomic(
                deceptive, step=3, write_stage=self._write_valid
            )
        self.assertEqual(caught.exception.reason, "PUBLICATION_REQUEST_INVALID")
        self.assertFalse((real / "checkpoint-3").exists())

    def test_publishes_complete_source_only_directory_with_last_commit_marker(self) -> None:
        self.assertIsNotNone(checkpoint, "checkpoint atomic publisher missing")
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = self.output / "checkpoint-3"
        self.assertEqual(receipt["checkpoint_path"], str(target))
        self.assertEqual(
            sorted(path.name for path in target.iterdir()),
            ["commit.json", "lora-trainables.safetensors", "state.json"],
        )
        commit = json.loads((target / "commit.json").read_text(encoding="utf-8"))
        self.assertEqual(commit["schema"], "R18_CHECKPOINT_COMMIT_V1")
        self.assertEqual(commit["completed_steps"], 3)
        self.assertEqual(
            commit["members"]["state.json"],
            hashlib.sha256((target / "state.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            commit["members"]["lora-trainables.safetensors"],
            hashlib.sha256((target / "lora-trainables.safetensors").read_bytes()).hexdigest(),
        )
        self.assertIs(receipt["authorizes_resume"], False)
        self.assertIs(receipt["authorizes_training"], False)
        self.assertIs(receipt["physical_durability_verified"], False)
        self.assertEqual(
            sorted(path.name for path in self.output.iterdir()), ["checkpoint-3"]
        )

    def test_verified_readback_must_match_external_commit_hash(self) -> None:
        self.assertIsNotNone(checkpoint)
        self.assertTrue(hasattr(checkpoint, "verify_published_checkpoint"))
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        reviewed = checkpoint.verify_published_checkpoint(
            Path(receipt["checkpoint_path"]),
            expected_step=3,
            expected_commit_sha256=receipt["checkpoint_commit_sha256"],
        )
        self.assertIs(reviewed["source_bytes_verified"], True)
        self.assertIs(reviewed["authorizes_resume"], False)
        self.assertIs(reviewed["authorizes_training"], False)
        self.assertIs(reviewed["physical_reload_verified"], False)

    def test_failed_writer_preserves_forensic_staging_without_adopting_target(self) -> None:
        self.assertIsNotNone(checkpoint)

        def fail_after_weights(stage: Path) -> None:
            (stage / "lora-trainables.safetensors").write_bytes(b"partial")
            raise RuntimeError("injected crash")

        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=fail_after_weights
            )
        self.assertEqual(caught.exception.reason, "STAGING_WRITER_FAILED")
        self.assertIsNotNone(caught.exception.stage_path)
        self.assertTrue(caught.exception.stage_path.is_dir())
        self.assertFalse((self.output / "checkpoint-3").exists())
        self.assertFalse((caught.exception.stage_path / "commit.json").exists())

    def test_existing_destination_never_overwritten(self) -> None:
        self.assertIsNotNone(checkpoint)
        target = self.output / "checkpoint-3"
        target.mkdir()
        marker = target / "important.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=self._write_valid
            )
        self.assertEqual(caught.exception.reason, "DESTINATION_ALREADY_EXISTS")
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(list(self.output.iterdir()), [target])

    def test_concurrent_destination_created_before_exclusive_rename_is_preserved(self) -> None:
        self.assertIsNotNone(checkpoint)
        original = checkpoint._exclusive_rename

        def simulate_race(stage: Path, target: Path) -> None:
            target.mkdir()
            (target / "attacker-marker.txt").write_text("preserve", encoding="utf-8")
            original(stage, target)

        with mock.patch.object(checkpoint, "_exclusive_rename", side_effect=simulate_race):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "DESTINATION_ALREADY_EXISTS")
        self.assertEqual(
            (self.output / "checkpoint-3" / "attacker-marker.txt").read_text(),
            "preserve",
        )
        self.assertIsNotNone(caught.exception.stage_path)
        self.assertTrue((caught.exception.stage_path / "commit.json").is_file())

    def test_stage_with_extra_member_is_rejected_without_commit(self) -> None:
        self.assertIsNotNone(checkpoint)

        def extra_member(stage: Path) -> None:
            self._write_valid(stage)
            (stage / "unsafe.txt").write_text("no", encoding="utf-8")

        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=extra_member
            )
        self.assertEqual(caught.exception.reason, "STAGING_MEMBERS_NOT_EXACT")
        self.assertFalse((caught.exception.stage_path / "commit.json").exists())

    def test_stage_with_symlink_member_is_rejected(self) -> None:
        self.assertIsNotNone(checkpoint)
        external = self.output / "external-weights"
        external.write_bytes(b"should remain untouched")

        def substituted_member(stage: Path) -> None:
            self._write_valid(stage)
            (stage / "lora-trainables.safetensors").unlink()
            os.symlink(external, stage / "lora-trainables.safetensors")

        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=substituted_member
            )
        self.assertEqual(caught.exception.reason, "MEMBER_TYPE_OR_SIZE_INVALID")
        self.assertEqual(external.read_bytes(), b"should remain untouched")

    def test_duplicate_json_and_weight_hash_mismatch_are_rejected(self) -> None:
        self.assertIsNotNone(checkpoint)

        def duplicate_state(stage: Path) -> None:
            self._write_valid(stage)
            state_file = stage / "state.json"
            state_file.write_text(
                state_file.read_text()[:-1] + ', "schema":"R18_MEZO_STATE_V1"}',
                encoding="utf-8",
            )

        with self.assertRaises(checkpoint.CheckpointPublicationError) as duplicate:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=duplicate_state
            )
        self.assertEqual(duplicate.exception.reason, "STATE_NOT_STRICT_JSON")

        def forged_weights(stage: Path) -> None:
            self._write_valid(stage)
            (stage / "lora-trainables.safetensors").write_bytes(b"altered")

        with self.assertRaises(checkpoint.CheckpointPublicationError) as altered:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=4, write_stage=forged_weights
            )
        self.assertEqual(altered.exception.reason, "STATE_CONTRACT_MISMATCH")

    def test_post_rename_sync_failure_is_unknown_outcome_without_automatic_replay(self) -> None:
        self.assertIsNotNone(checkpoint)
        original = checkpoint._sync_directory

        def fail_on_parent(directory: Path) -> None:
            if directory == self.output:
                raise checkpoint.CheckpointPublicationError("DIRECTORY_FSYNC_FAILED")
            original(directory)

        with mock.patch.object(checkpoint, "_sync_directory", side_effect=fail_on_parent):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, self.output / "checkpoint-3")
        self.assertTrue((self.output / "checkpoint-3" / "commit.json").is_file())

    def test_unexpected_post_rename_exception_preserves_target_unknown_status(self) -> None:
        """A non-custody exception after publication cannot be a writer failure."""
        original = checkpoint._sync_directory

        def fail_after_rename(directory: Path) -> None:
            if directory == self.output:
                raise RuntimeError("unknown parent sync")
            original(directory)

        with mock.patch.object(checkpoint, "_sync_directory", side_effect=fail_after_rename):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, self.output / "checkpoint-3")
        self.assertTrue((self.output / "checkpoint-3" / "commit.json").is_file())

    @unittest.skipIf(os.name == "nt", "requires POSIX symlink swap")
    def test_stage_directory_symlink_swap_cannot_claim_commit_success(self) -> None:
        """A renamed symlink must not inherit validation of the old stage inode."""
        original = checkpoint._exclusive_rename

        def swap_stage_for_symlink(stage: Path, target: Path) -> None:
            stage.rename(stage.with_name(stage.name + "-moved"))
            os.symlink(self.output, stage, target_is_directory=True)
            original(stage, target)

        with mock.patch.object(checkpoint, "_exclusive_rename", side_effect=swap_stage_for_symlink):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, self.output / "checkpoint-3")
        self.assertTrue((self.output / "checkpoint-3").is_symlink())

    def test_stage_directory_replacement_with_other_directory_cannot_claim_success(self) -> None:
        original = checkpoint._exclusive_rename

        def swap_stage_for_other_directory(stage: Path, target: Path) -> None:
            stage.rename(stage.with_name(stage.name + "-moved"))
            stage.mkdir()
            original(stage, target)

        with mock.patch.object(checkpoint, "_exclusive_rename", side_effect=swap_stage_for_other_directory):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, self.output / "checkpoint-3")
        self.assertTrue((self.output / "checkpoint-3").is_dir())
        self.assertFalse((self.output / "checkpoint-3" / "commit.json").exists())

    def test_rename_that_succeeds_then_raises_has_unknown_custody(self) -> None:
        """An exception before the caller sets renamed cannot imply no target."""
        original = checkpoint._exclusive_rename

        def renamed_then_unexpected_exception(stage: Path, target: Path) -> None:
            original(stage, target)
            raise RuntimeError("lost rename acknowledgment")

        with mock.patch.object(
            checkpoint, "_exclusive_rename", side_effect=renamed_then_unexpected_exception
        ):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, self.output / "checkpoint-3")
        self.assertTrue((self.output / "checkpoint-3" / "commit.json").is_file())

    def test_successful_rename_with_misleading_collision_error_is_still_unknown(self) -> None:
        original = checkpoint._exclusive_rename

        def renamed_then_bad_collision_receipt(stage: Path, target: Path) -> None:
            original(stage, target)
            raise checkpoint.CheckpointPublicationError("DESTINATION_ALREADY_EXISTS")

        with mock.patch.object(
            checkpoint, "_exclusive_rename", side_effect=renamed_then_bad_collision_receipt
        ):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, self.output / "checkpoint-3")

    @unittest.skipIf(os.name == "nt", "requires POSIX output-root symlink swap")
    def test_output_directory_swap_before_rename_cannot_claim_source_commit(self) -> None:
        """The output root is part of custody, not only the staged inode."""
        output = self.output / "output"
        output.mkdir()
        displaced = self.output / "output-displaced"
        original = checkpoint._exclusive_rename

        def swap_output_root(stage: Path, target: Path) -> None:
            output.rename(displaced)
            os.symlink(displaced, output, target_is_directory=True)
            original(stage, target)

        with mock.patch.object(checkpoint, "_exclusive_rename", side_effect=swap_output_root):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, output / "checkpoint-3")
        self.assertTrue(output.is_symlink())
        self.assertTrue((displaced / "checkpoint-3" / "commit.json").is_file())

    @unittest.skipIf(os.name == "nt", "requires POSIX ancestor symlink swap")
    def test_ancestor_swap_during_parent_sync_is_postrename_unknown(self) -> None:
        """An unchanged output inode reached through a new symlink is not custody."""
        outer = self.output / "outer"
        outer.mkdir()
        output = outer / "output"
        output.mkdir()
        displaced = self.output / "outer-displaced"
        original = checkpoint._sync_directory
        swapped = False

        def rebind_outer_after_sync(directory: Path) -> None:
            nonlocal swapped
            original(directory)
            if directory == output and not swapped:
                swapped = True
                outer.rename(displaced)
                os.symlink(displaced, outer, target_is_directory=True)

        with mock.patch.object(checkpoint, "_sync_directory", rebind_outer_after_sync):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    output, step=3, write_stage=self._write_valid
                )
        self.assertTrue(swapped)
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertEqual(caught.exception.target_path, output / "checkpoint-3")
        self.assertTrue(outer.is_symlink())
        self.assertTrue((displaced / "output" / "checkpoint-3" / "commit.json").is_file())

    @unittest.skipIf(os.name == "nt", "requires POSIX ancestor symlink swap")
    def test_direct_readback_rejects_symlinked_grandparent_directory(self) -> None:
        """Rebound ancestors must not be silently followed during readback."""
        outer = self.output / "outer"
        outer.mkdir()
        output = outer / "output"
        output.mkdir()
        receipt = checkpoint.publish_checkpoint_atomic(
            output, step=3, write_stage=self._write_valid
        )
        displaced = self.output / "outer-displaced"
        outer.rename(displaced)
        os.symlink(displaced, outer, target_is_directory=True)

        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.verify_published_checkpoint(
                output / "checkpoint-3", expected_step=3,
                expected_commit_sha256=receipt["checkpoint_commit_sha256"],
            )
        self.assertEqual(caught.exception.reason, "READBACK_DIRECTORY_CHANGED")
        self.assertTrue(outer.is_symlink())

    @unittest.skipIf(os.name == "nt", "requires POSIX ancestor symlink swap")
    def test_publisher_rejects_already_symlinked_grandparent(self) -> None:
        outer = self.output / "outer"
        outer.mkdir()
        output = outer / "output"
        output.mkdir()
        displaced = self.output / "outer-displaced"
        outer.rename(displaced)
        os.symlink(displaced, outer, target_is_directory=True)
        with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
            checkpoint.publish_checkpoint_atomic(
                output, step=3, write_stage=self._write_valid
            )
        self.assertEqual(caught.exception.reason, "PUBLICATION_REQUEST_INVALID")
        self.assertFalse((displaced / "output" / "checkpoint-3").exists())

    @unittest.skipIf(os.name == "nt", "requires POSIX ancestor symlink swap")
    def test_ancestor_swap_during_exclusive_rename_cannot_claim_success(self) -> None:
        outer = self.output / "outer"
        outer.mkdir()
        output = outer / "output"
        output.mkdir()
        displaced = self.output / "outer-displaced"
        original = checkpoint._exclusive_rename

        def rebind_outer_before_rename(stage: Path, target: Path) -> None:
            outer.rename(displaced)
            os.symlink(displaced, outer, target_is_directory=True)
            original(stage, target)

        with mock.patch.object(checkpoint, "_exclusive_rename", rebind_outer_before_rename):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertTrue(outer.is_symlink())

    @unittest.skipIf(os.name == "nt", "requires POSIX target symlink swap")
    def test_target_swap_during_parent_sync_cannot_claim_source_commit(self) -> None:
        """A post-sync check must not bless a replaced checkpoint pathname."""
        target = self.output / "checkpoint-3"
        displaced = self.output / "checkpoint-3-displaced"
        original = checkpoint._sync_directory

        def swap_after_readback(directory: Path) -> None:
            if directory == self.output:
                target.rename(displaced)
                os.symlink(self.output, target, target_is_directory=True)
            original(directory)

        with mock.patch.object(checkpoint, "_sync_directory", side_effect=swap_after_readback):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertIsNone(caught.exception.stage_path)
        self.assertEqual(caught.exception.target_path, target)
        self.assertTrue(target.is_symlink())
        self.assertTrue((displaced / "commit.json").is_file())

    @unittest.skipIf(os.name == "nt", "requires POSIX target symlink swap")
    def test_readback_target_rebound_during_member_listing_is_not_verified(self) -> None:
        """The final descriptor hashes cannot endorse a replaced directory path."""
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = self.output / "checkpoint-3"
        displaced = self.output / "checkpoint-3-displaced"
        original_iterdir = Path.iterdir
        swapped = False

        def rebind_during_listing(path: Path):
            nonlocal swapped
            if path == target and not swapped:
                swapped = True
                target.rename(displaced)
                os.symlink(displaced, target, target_is_directory=True)
            return original_iterdir(path)

        with mock.patch.object(Path, "iterdir", rebind_during_listing):
            with self.assertRaises(checkpoint.CheckpointPublicationError):
                checkpoint.verify_published_checkpoint(
                    target, expected_step=3,
                    expected_commit_sha256=receipt["checkpoint_commit_sha256"],
                )
        self.assertTrue(swapped)
        self.assertTrue(target.is_symlink())

    @unittest.skipIf(os.name == "nt", "requires POSIX target symlink swap")
    def test_publisher_final_readback_rebinding_is_unknown_not_success(self) -> None:
        """A target exchanged during final readback cannot issue success."""
        target = self.output / "checkpoint-3"
        displaced = self.output / "checkpoint-3-displaced"
        original_iterdir = Path.iterdir
        target_listings = 0

        def rebind_on_final_listing(path: Path):
            nonlocal target_listings
            if path == target:
                target_listings += 1
                if target_listings == 2:
                    target.rename(displaced)
                    os.symlink(displaced, target, target_is_directory=True)
            return original_iterdir(path)

        with mock.patch.object(Path, "iterdir", rebind_on_final_listing):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(target_listings, 2)
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertEqual(caught.exception.target_path, target)
        self.assertTrue(target.is_symlink())

    def test_readback_rejects_wrong_external_digest_and_tampered_payload(self) -> None:
        self.assertIsNotNone(checkpoint)
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = Path(receipt["checkpoint_path"])
        original_hash = receipt["checkpoint_commit_sha256"]
        with self.assertRaises(checkpoint.CheckpointPublicationError) as wrong:
            checkpoint.verify_published_checkpoint(
                target, expected_step=3, expected_commit_sha256="0" * 64
            )
        self.assertEqual(wrong.exception.reason, "COMMIT_HASH_MISMATCH")
        (target / "lora-trainables.safetensors").write_bytes(b"tampered-weights")
        with self.assertRaises(checkpoint.CheckpointPublicationError) as tampered:
            checkpoint.verify_published_checkpoint(
                target, expected_step=3, expected_commit_sha256=original_hash
            )
        self.assertEqual(tampered.exception.reason, "STATE_CONTRACT_MISMATCH")

    def test_readback_rejects_truncated_commit_and_unexpected_member(self) -> None:
        self.assertIsNotNone(checkpoint)
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = Path(receipt["checkpoint_path"])
        (target / "foreign.bin").write_bytes(b"forbidden")
        with self.assertRaises(checkpoint.CheckpointPublicationError) as extra:
            checkpoint.verify_published_checkpoint(
                target, expected_step=3,
                expected_commit_sha256=receipt["checkpoint_commit_sha256"],
            )
        self.assertEqual(extra.exception.reason, "READBACK_MEMBERS_NOT_EXACT")
        (target / "foreign.bin").unlink()
        (target / "commit.json").write_bytes(b'{"schema":')
        with self.assertRaises(checkpoint.CheckpointPublicationError) as truncated:
            checkpoint.verify_published_checkpoint(
                target, expected_step=3,
                expected_commit_sha256=receipt["checkpoint_commit_sha256"],
            )
        self.assertEqual(truncated.exception.reason, "COMMIT_HASH_MISMATCH")

    def test_readback_rejects_self_rehashed_forged_commit_authority(self) -> None:
        self.assertIsNotNone(checkpoint)
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = Path(receipt["checkpoint_path"])
        marker = json.loads((target / "commit.json").read_text())
        marker["authorizes_training"] = True
        forged = json.dumps(marker, sort_keys=True).encode()
        (target / "commit.json").write_bytes(forged)
        with self.assertRaises(checkpoint.CheckpointPublicationError) as forged_receipt:
            checkpoint.verify_published_checkpoint(
                target, expected_step=3,
                expected_commit_sha256=hashlib.sha256(forged).hexdigest(),
            )
        self.assertEqual(forged_receipt.exception.reason, "COMMIT_DECLARATION_INVALID")

    def test_readback_rejects_state_replacement_after_payload_validation(self) -> None:
        """Parsed hashes cannot bless a member replaced before verification ends."""
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = self.output / "checkpoint-3"
        original = checkpoint._validate_payload
        replaced = False

        def replace_after_parse(directory: Path, *, step: int):
            nonlocal replaced
            result = original(directory, step=step)
            if directory == target and not replaced:
                replaced = True
                other = target / "replacement-state.json"
                other.write_bytes(b"forged payload")
                os.replace(other, target / "state.json")
            return result

        with mock.patch.object(checkpoint, "_validate_payload", replace_after_parse):
            with self.assertRaises(checkpoint.CheckpointPublicationError):
                checkpoint.verify_published_checkpoint(
                    target, expected_step=3,
                    expected_commit_sha256=receipt["checkpoint_commit_sha256"],
                )
        self.assertTrue(replaced)
        self.assertEqual((target / "state.json").read_bytes(), b"forged payload")

    def test_publisher_final_readback_member_replacement_is_unknown(self) -> None:
        """A final-readback member swap must not yield a committed receipt."""
        target = self.output / "checkpoint-3"
        original = checkpoint._validate_payload
        calls = 0

        def replace_during_final_readback(directory: Path, *, step: int):
            nonlocal calls
            result = original(directory, step=step)
            calls += 1
            if directory == target and calls == 3:
                other = target / "replacement-state.json"
                other.write_bytes(b"forged payload")
                os.replace(other, target / "state.json")
            return result

        with mock.patch.object(checkpoint, "_validate_payload", replace_during_final_readback):
            with self.assertRaises(checkpoint.CheckpointPublicationError) as caught:
                checkpoint.publish_checkpoint_atomic(
                    self.output, step=3, write_stage=self._write_valid
                )
        self.assertEqual(calls, 3)
        self.assertEqual(caught.exception.reason, "POST_RENAME_DURABILITY_UNKNOWN")
        self.assertEqual(caught.exception.target_path, target)
        self.assertEqual((target / "state.json").read_bytes(), b"forged payload")

    def test_readback_rechecks_all_members_after_payload_validation(self) -> None:
        """Weights, state and marker must remain pinned after payload parsing."""
        for member in (
            "lora-trainables.safetensors", "state.json", "commit.json",
        ):
            with self.subTest(member=member):
                output = self.output / member.replace(".", "-")
                output.mkdir()
                receipt = checkpoint.publish_checkpoint_atomic(
                    output, step=3, write_stage=self._write_valid
                )
                target = output / "checkpoint-3"
                original = checkpoint._validate_payload

                def replace_after_parse(directory: Path, *, step: int):
                    hashes = original(directory, step=step)
                    if directory == target:
                        other = target / "replacement.bin"
                        other.write_bytes(b"unrelated member content")
                        os.replace(other, target / member)
                    return hashes

                with mock.patch.object(
                    checkpoint, "_validate_payload", replace_after_parse,
                ):
                    with self.assertRaises(checkpoint.CheckpointPublicationError):
                        checkpoint.verify_published_checkpoint(
                            target, expected_step=3,
                            expected_commit_sha256=receipt["checkpoint_commit_sha256"],
                        )

    def test_readback_detects_inplace_equal_size_member_mutation(self) -> None:
        """An unchanged inode/size is insufficient after a byte mutation."""
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        target = self.output / "checkpoint-3"
        original = checkpoint._validate_payload

        def mutate_after_parse(directory: Path, *, step: int):
            hashes = original(directory, step=step)
            if directory == target:
                state_path = target / "state.json"
                with state_path.open("r+b") as state_file:
                    state_file.seek(0)
                    state_file.write(b"!" * state_path.stat().st_size)
            return hashes

        with mock.patch.object(checkpoint, "_validate_payload", mutate_after_parse):
            with self.assertRaises(checkpoint.CheckpointPublicationError):
                checkpoint.verify_published_checkpoint(
                    target, expected_step=3,
                    expected_commit_sha256=receipt["checkpoint_commit_sha256"],
                )

    def test_readback_parses_exact_bytes_it_hashes_without_reopening_members(self) -> None:
        """A second path open permits hash/parse races over mutable evidence."""
        self.assertIsNotNone(checkpoint)
        receipt = checkpoint.publish_checkpoint_atomic(
            self.output, step=3, write_stage=self._write_valid
        )
        observed_reopens = []
        original_open = Path.open

        def record_path_open(path: Path, *args, **kwargs):
            if path.name in {"state.json", "commit.json"}:
                observed_reopens.append(path.name)
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", record_path_open):
            reviewed = checkpoint.verify_published_checkpoint(
                Path(receipt["checkpoint_path"]), expected_step=3,
                expected_commit_sha256=receipt["checkpoint_commit_sha256"],
            )
        self.assertIs(reviewed["source_bytes_verified"], True)
        self.assertEqual(
            observed_reopens, [], "hash and JSON parse must use one descriptor"
        )

    def test_deep_json_state_and_nonfinite_timing_are_rejected_before_marker(self) -> None:
        self.assertIsNotNone(checkpoint)

        def deeply_nested(stage: Path) -> None:
            self._write_valid(stage)
            (stage / "state.json").write_text(
                '{"evidence":' + '[' * 1100 + '0' + ']' * 1100 + '}',
                encoding="utf-8",
            )

        with self.assertRaises(checkpoint.CheckpointPublicationError) as deep:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=deeply_nested
            )
        self.assertEqual(deep.exception.reason, "STATE_NOT_STRICT_JSON")

        def nonfinite(stage: Path) -> None:
            self._write_valid(stage)
            state = json.loads((stage / "state.json").read_text())
            state["optimizer_window_seconds"] = float("nan")
            (stage / "state.json").write_text(json.dumps(state), encoding="utf-8")

        with self.assertRaises(checkpoint.CheckpointPublicationError) as nan:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=3, write_stage=nonfinite
            )
        self.assertEqual(nan.exception.reason, "STATE_NOT_STRICT_JSON")

    def test_request_rejects_bool_step_and_symlinked_output(self) -> None:
        self.assertIsNotNone(checkpoint)
        with self.assertRaises(checkpoint.CheckpointPublicationError) as wrong_step:
            checkpoint.publish_checkpoint_atomic(
                self.output, step=True, write_stage=self._write_valid
            )
        self.assertEqual(wrong_step.exception.reason, "PUBLICATION_REQUEST_INVALID")
        linked = self.output / "linked-output"
        os.symlink(self.output, linked, target_is_directory=True)
        with self.assertRaises(checkpoint.CheckpointPublicationError) as bad_root:
            checkpoint.publish_checkpoint_atomic(
                linked, step=3, write_stage=self._write_valid
            )
        self.assertEqual(bad_root.exception.reason, "PUBLICATION_REQUEST_INVALID")


if __name__ == "__main__":
    unittest.main()
