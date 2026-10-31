"""Dependency-free source/custody gates; NOT evidence of GPU training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
import trainingday.checkpoint_custody as custody_module
from trainingday.checkpoint_custody import create_custody, validate_custody, write_external_custody
from trainingday.train_neural import _r16_prepare_output


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R16CheckpointCustodyTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output_root = self.root / "outputs"
        self.output_root.mkdir()
        self.binding = self.root / "base-binding.json"
        self.binding.write_text("pinned-base-bytes", encoding="utf-8")
        self.args = argparse.Namespace(
            r16_checkpointable=True, output_root=self.output_root,
            run_id="r16-correct-a", base_binding=self.binding,
            producer_git_head="1d38e4f25b5663bf8a84ff5e139875d4152e3680",
            mode="correct", steps=16, cases=128, seed=41,
            learning_rate=2e-5, expected_initial_digest="d" * 64,
            resume_checkpoint=None, resume_custody=None, resume_custody_sha256=None,
        )
        self.preflight = {
            "corpus": {"manifest_sha256": "a" * 64},
            "base_model": "Qwen/Qwen3.5-4B", "base_revision": "b" * 40,
            "training_input_sha256": "c" * 64,
        }

    def checkpoint(self, output: Path, step: int = 8):
        root = output / "trainer-work" / f"checkpoint-{step}"
        root.mkdir(parents=True)
        files = {
            "trainer_state.json": json.dumps({"global_step": step}).encode(),
            "optimizer.pt": b"optimizer-state",
            "scheduler.pt": b"scheduler-state",
            "adapter_model.safetensors": b"adapter-weights",
            "adapter_config.json": b'{"peft_type":"LORA"}',
            "rng_state.pth": b"random-state",
        }
        for name, content in files.items():
            (root / name).write_bytes(content)
        return root

    def prepare_resume(self):
        output, _, _, contract_sha = _r16_prepare_output(self.args, self.preflight)
        checkpoint = self.checkpoint(output)
        custody = create_custody(checkpoint, contract_sha256=contract_sha)
        external = self.root / "externally-pinned-custody.json"
        external.write_text(json.dumps(custody, sort_keys=True), encoding="utf-8")
        self.args.resume_checkpoint = checkpoint
        self.args.resume_custody = external
        self.args.resume_custody_sha256 = _hash(external)
        return output, checkpoint, contract_sha, custody

    def test_fresh_then_exact_pinned_resume_preflight(self):
        output, checkpoint, contract_sha, custody = self.prepare_resume()
        resumed_output, resumed_checkpoint, step, new_sha = _r16_prepare_output(
            self.args, self.preflight
        )
        self.assertEqual((resumed_output, resumed_checkpoint, step, new_sha),
                         (output, checkpoint, 8, contract_sha))
        self.assertEqual(validate_custody(checkpoint, custody, contract_sha256=contract_sha), 8)

    def test_optimizer_tamper_fails_closed(self):
        _, checkpoint, _, _ = self.prepare_resume()
        (checkpoint / "optimizer.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(AdmissionError, "checkpoint bytes"):
            _r16_prepare_output(self.args, self.preflight)

    def test_missing_rng_fails_closed(self):
        _, checkpoint, _, _ = self.prepare_resume()
        (checkpoint / "rng_state.pth").unlink()
        with self.assertRaisesRegex(AdmissionError, "RNG"):
            _r16_prepare_output(self.args, self.preflight)

    def test_stale_hyperparameter_fails_closed(self):
        self.prepare_resume()
        self.args.learning_rate = 1e-5
        with self.assertRaisesRegex(AdmissionError, "run contract"):
            _r16_prepare_output(self.args, self.preflight)

    def test_stale_training_input_fails_closed(self):
        self.prepare_resume()
        self.preflight["training_input_sha256"] = "9" * 64
        with self.assertRaisesRegex(AdmissionError, "run contract"):
            _r16_prepare_output(self.args, self.preflight)

    def test_external_receipt_must_be_pinned(self):
        self.prepare_resume()
        self.args.resume_custody_sha256 = "0" * 64
        with self.assertRaisesRegex(AdmissionError, "externally pinned"):
            _r16_prepare_output(self.args, self.preflight)

    def test_custody_cannot_live_inside_output(self):
        output, _, _, _ = self.prepare_resume()
        duplicate = output / "bad-custody.json"
        duplicate.write_bytes(Path(self.args.resume_custody).read_bytes())
        self.args.resume_custody = duplicate
        self.args.resume_custody_sha256 = _hash(duplicate)
        with self.assertRaisesRegex(AdmissionError, "externally pinned"):
            _r16_prepare_output(self.args, self.preflight)

    def test_step_beyond_target_fails_closed(self):
        output, _, contract_sha, _ = self.prepare_resume()
        checkpoint = self.checkpoint(output, step=16)
        custody = create_custody(checkpoint, contract_sha256=contract_sha)
        external = self.root / "wrong-step-custody.json"
        external.write_text(json.dumps(custody), encoding="utf-8")
        self.args.resume_checkpoint = checkpoint
        self.args.resume_custody = external
        self.args.resume_custody_sha256 = _hash(external)
        with self.assertRaisesRegex(AdmissionError, "step outside"):
            _r16_prepare_output(self.args, self.preflight)

    def test_checkpoint_directory_step_must_match_state(self):
        output, _, contract_sha, _ = self.prepare_resume()
        checkpoint = self.checkpoint(output, step=16)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 8}))
        with self.assertRaisesRegex(AdmissionError, "step mismatch"):
            create_custody(checkpoint, contract_sha256=contract_sha)

    def test_checkpoint_missing_optimizer_is_not_resume(self):
        output, _, contract_sha, _ = self.prepare_resume()
        checkpoint = self.checkpoint(output, step=16)
        (checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(AdmissionError, "missing optimizer"):
            create_custody(checkpoint, contract_sha256=contract_sha)

    def test_external_checkpoint_receipt_creation(self):
        _, checkpoint, _, _ = self.prepare_resume()
        new_receipt = self.root / "new-custody.json"
        result = write_external_custody(checkpoint, new_receipt)
        self.assertEqual(result["step"], 8)
        self.assertEqual(result["receipt_sha256"], _hash(new_receipt))
        with self.assertRaisesRegex(AdmissionError, "unsafe"):
            write_external_custody(checkpoint, new_receipt)

    def test_external_receipt_rejects_run_internal_destination(self):
        output, checkpoint, _, _ = self.prepare_resume()
        with self.assertRaisesRegex(AdmissionError, "unsafe"):
            write_external_custody(checkpoint, output / "internal-custody.json")

    def test_external_receipt_rejects_dotdot_alias_into_run_output(self):
        output, checkpoint, _, _ = self.prepare_resume()
        (self.root / "outside").mkdir()
        alias = self.root / "outside" / ".." / "outputs" / output.name / "alias-custody.json"
        self.assertEqual(alias.resolve().parent, output.resolve())
        with self.assertRaisesRegex(AdmissionError, "unsafe"):
            write_external_custody(checkpoint, alias)
        self.assertFalse((output / "alias-custody.json").exists())

    def test_external_receipt_staging_sync_failure_never_publishes_target(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "fsync-fault-custody.json"
        with patch("trainingday.checkpoint_custody.os.fsync", side_effect=OSError("disk fault")):
            with self.assertRaisesRegex(AdmissionError, "staging failed"):
                write_external_custody(checkpoint, destination)
        self.assertFalse(destination.exists())

    def test_external_receipt_parent_sync_failure_is_unknown_after_publication(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "parent-sync-fault-custody.json"
        original_sync = os.fsync
        count = 0

        def fail_second_sync(fd):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("injected post-publish parent sync failure")
            return original_sync(fd)

        with patch("trainingday.checkpoint_custody.os.fsync", side_effect=fail_second_sync):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertEqual(count, 2)
        self.assertTrue(destination.is_file())
        self.assertEqual(json.loads(destination.read_text())["step"], 8)

    def test_external_receipt_link_uncertain_after_publication_is_unknown(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "link-uncertain-custody.json"
        original_link = os.link

        def link_then_lose_ack(source, target, **kwargs):
            original_link(source, target, **kwargs)
            raise RuntimeError("publication acknowledged outcome unknown")

        with patch("trainingday.checkpoint_custody.os.link", side_effect=link_then_lose_ack):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(destination.is_file())

    def test_external_receipt_destination_race_never_overwrites_winner(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "raced-custody.json"
        original_link = os.link

        def concurrent_create(source, target, **kwargs):
            Path(target).write_bytes(b"concurrent-owner")
            return original_link(source, target, **kwargs)

        with patch("trainingday.checkpoint_custody.os.link", side_effect=concurrent_create):
            with self.assertRaisesRegex(AdmissionError, "unsafe"):
                write_external_custody(checkpoint, destination)
        self.assertEqual(destination.read_bytes(), b"concurrent-owner")

    def test_external_receipt_staging_symlink_swap_cannot_forge_success(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "stage-swap-custody.json"
        impostor = self.root / "impostor.json"
        impostor.write_bytes(b"another payload")
        original_link = os.link

        def replace_stage(source, target, **kwargs):
            source = Path(source)
            source.rename(source.with_name(source.name + "-displaced"))
            source.symlink_to(impostor)
            return original_link(source, target, **kwargs)

        with patch("trainingday.checkpoint_custody.os.link", side_effect=replace_stage):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(destination.exists() or destination.is_symlink())

    def test_external_receipt_preserves_replaced_staging_member_after_link(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "post-link-replaced-stage.json"
        original_link = os.link
        replaced = []

        def replace_stage_after_link(source, target, **kwargs):
            original_link(source, target, **kwargs)
            source = Path(source)
            source.rename(source.with_name(source.name + "-displaced"))
            source.write_bytes(b"not-our-staging-file")
            replaced.append(source)

        with patch("trainingday.checkpoint_custody.os.link", side_effect=replace_stage_after_link):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(destination.is_file())
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0].read_bytes(), b"not-our-staging-file")

    def test_external_receipt_parent_rebound_to_symlink_after_sync_is_unknown(self):
        _, checkpoint, _, _ = self.prepare_resume()
        parent = self.root / "external-parent"
        parent.mkdir()
        destination = parent / "receipt.json"
        relocated = self.root / "external-parent-relocated"
        original_sync = custody_module._sync_directory

        def sync_then_rebind(directory):
            original_sync(directory)
            if Path(directory) == parent:
                parent.rename(relocated)
                parent.symlink_to(relocated, target_is_directory=True)

        with patch("trainingday.checkpoint_custody._sync_directory", side_effect=sync_then_rebind):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(parent.is_symlink())
        self.assertEqual((relocated / "receipt.json").read_bytes(), destination.read_bytes())

    def test_external_receipt_target_swapped_to_symlink_after_sync_is_unknown(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "post-sync-target-custody.json"
        relocated = self.root / "post-sync-target-custody-displaced.json"
        impostor = self.root / "impostor-post-sync.json"
        impostor.write_bytes(b"not the committed custody receipt")
        original_sync = custody_module._sync_directory

        def sync_then_replace_target(directory):
            original_sync(directory)
            if Path(directory) == destination.parent:
                destination.rename(relocated)
                destination.symlink_to(impostor)

        with patch("trainingday.checkpoint_custody._sync_directory", side_effect=sync_then_replace_target):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(destination.is_symlink())
        self.assertTrue(relocated.is_file())

    def test_external_receipt_fileexists_ack_after_actual_link_is_unknown(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "post-link-exists-ack.json"
        original_link = os.link

        def link_then_raise_exists(source, target, **kwargs):
            original_link(source, target, **kwargs)
            raise FileExistsError("ack ambiguous after actual link")

        with patch("trainingday.checkpoint_custody.os.link", side_effect=link_then_raise_exists):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(destination.is_file())

    def test_external_receipt_replaced_while_final_fd_open_is_unknown(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "final-fd-swap-custody.json"
        displaced = self.root / "final-fd-swap-displaced.json"
        impostor = self.root / "final-fd-swap-impostor.json"
        impostor.write_bytes(b"malicious target")
        original_open = os.open
        target_opens = 0

        def open_then_swap_target(path, flags, *args, **kwargs):
            nonlocal target_opens
            descriptor = original_open(path, flags, *args, **kwargs)
            if Path(path) == destination:
                target_opens += 1
                if target_opens == 2:
                    destination.rename(displaced)
                    destination.symlink_to(impostor)
            return descriptor

        with patch("trainingday.checkpoint_custody.os.open", side_effect=open_then_swap_target):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertEqual(target_opens, 2)
        self.assertTrue(destination.is_symlink())
        self.assertTrue(displaced.is_file())

    def test_external_receipt_success_never_unlinks_a_substituted_stage(self):
        _, checkpoint, _, _ = self.prepare_resume()
        destination = self.root / "stage-cleanup-custody.json"
        original_unlink = Path.unlink
        attempted_stage_unlinks = []

        def replace_stage_before_unlink(path, *args, **kwargs):
            path = Path(path)
            if ".r16-stage-" in path.name:
                displaced = path.with_name(path.name + "-displaced")
                path.rename(displaced)
                path.write_bytes(b"unrelated substituted receipt")
                attempted_stage_unlinks.append(path)
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", new=replace_stage_before_unlink):
            result = write_external_custody(checkpoint, destination)
        self.assertEqual(result["receipt_sha256"], _hash(destination))
        self.assertEqual(attempted_stage_unlinks, [], "stage cleanup cannot atomically verify its target")
        retained = list(self.root.glob(".stage-cleanup-custody.json.r16-stage-*"))
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].read_bytes(), destination.read_bytes())

    def test_external_receipt_ancestor_rebound_after_sync_is_unknown(self):
        _, checkpoint, _, _ = self.prepare_resume()
        outer = self.root / "outer-custody"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        relocated = self.root / "outer-custody-relocated"
        destination = inner / "receipt.json"
        original_sync = custody_module._sync_directory

        def sync_then_rebind_ancestor(directory):
            original_sync(directory)
            if Path(directory) == inner:
                outer.rename(relocated)
                outer.symlink_to(relocated, target_is_directory=True)

        with patch("trainingday.checkpoint_custody._sync_directory", side_effect=sync_then_rebind_ancestor):
            with self.assertRaisesRegex(AdmissionError, "POST_PUBLISH_DURABILITY_UNKNOWN"):
                write_external_custody(checkpoint, destination)
        self.assertTrue(outer.is_symlink())
        self.assertTrue((relocated / "inner" / "receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
