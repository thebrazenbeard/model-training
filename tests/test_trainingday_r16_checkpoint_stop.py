"""CPU-only tests for an intentional R16 stop *after* a complete save."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from trainingday.admission import AdmissionError
from trainingday.checkpoint_stop import (
    build_checkpoint_stop_callback,
    make_paused_phase_receipt,
    validate_checkpoint_stop_request,
)
from trainingday.train_neural import _r16_contract


class FakeTrainerCallback:
    """Only the inheritance seam is faked; custody reads real temporary files."""


class CheckpointStopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "trainer-work"
        self.work.mkdir()
        self.args = SimpleNamespace(
            output_dir=str(self.work), max_steps=16,
            save_strategy="steps", save_steps=8, save_only_model=False,
        )
        self.state = SimpleNamespace(global_step=8, is_world_process_zero=True)
        self.control = SimpleNamespace(should_training_stop=False)
        self.contract_sha = "a" * 64

    def checkpoint(self, *, step=8, missing=None):
        target = self.work / f"checkpoint-{step}"
        target.mkdir()
        files = {
            "trainer_state.json": json.dumps({"global_step": step}).encode(),
            "optimizer.pt": b"optimizer-state",
            "scheduler.pt": b"scheduler-state",
            "adapter_model.safetensors": b"adapter-state",
            "adapter_config.json": b"{}",
            "rng_state.pth": b"rng-state",
        }
        for name, content in files.items():
            if name != missing:
                (target / name).write_bytes(content)
        return target

    def make_callback(self):
        callback = build_checkpoint_stop_callback(
            FakeTrainerCallback, trainer_work=self.work,
            contract_sha256=self.contract_sha, stop_step=8,
        )
        self.assertIsInstance(callback, FakeTrainerCallback)
        return callback

    def test_after_save_complete_step8_requests_graceful_stop(self):
        self.checkpoint()
        callback = self.make_callback()
        result = callback.on_save(self.args, self.state, self.control)
        self.assertIs(result, self.control)
        self.assertTrue(result.should_training_stop)
        self.assertTrue(callback.triggered)
        self.assertEqual(callback.custody["step"], 8)
        self.assertIn("optimizer.pt", callback.custody["files"])
        self.assertEqual(callback.custody["state"],
                         "CHECKPOINT_BYTES_PINNED_NOT_RESUME_ATTESTED")

    def test_callback_never_stops_at_unrelated_step(self):
        self.checkpoint(step=8)
        self.state.global_step = 7
        callback = self.make_callback()
        callback.on_save(self.args, self.state, self.control)
        self.assertFalse(self.control.should_training_stop)
        self.assertFalse(callback.triggered)

    def test_incomplete_checkpoint_fails_without_stop_receipt(self):
        for missing in ("optimizer.pt", "scheduler.pt", "rng_state.pth",
                        "adapter_model.safetensors", "trainer_state.json"):
            with self.subTest(missing=missing):
                with tempfile.TemporaryDirectory() as scratch:
                    self.work = Path(scratch)
                    self.args.output_dir = str(self.work)
                    self.checkpoint(missing=missing)
                    callback = self.make_callback()
                    self.control.should_training_stop = False
                    with self.assertRaises(AdmissionError):
                        callback.on_save(self.args, self.state, self.control)
                    self.assertFalse(callback.triggered)
                    self.assertFalse(self.control.should_training_stop)

    def test_no_checkpoint_and_wrong_trainer_settings_fail_closed(self):
        callback = self.make_callback()
        with self.assertRaises(AdmissionError):
            callback.on_save(self.args, self.state, self.control)
        self.checkpoint()
        self.args.save_only_model = True
        with self.assertRaises(AdmissionError):
            callback.on_save(self.args, self.state, self.control)
        self.assertFalse(callback.triggered)

    def test_spoofed_or_overshot_step_cannot_claim_graceful_stop(self):
        self.checkpoint()
        callback = self.make_callback()
        for step in (True, "8", 8.0, 9):
            with self.subTest(step=step):
                self.state.global_step = step
                with self.assertRaises(AdmissionError):
                    callback.on_save(self.args, self.state, self.control)
                self.assertFalse(callback.triggered)

    def test_request_requires_fresh_r16_sixteen_step_trained_arm(self):
        base = dict(r16_checkpointable=True, mode="correct", steps=16,
                    stop_after_checkpoint_step=8, resume_checkpoint=None)
        self.assertTrue(validate_checkpoint_stop_request(argparse.Namespace(**base)))
        for change in ({"r16_checkpointable": False}, {"mode": "zero"},
                       {"steps": 8}, {"stop_after_checkpoint_step": 16},
                       {"resume_checkpoint": Path("checkpoint-8")},
                       {"stop_after_checkpoint_step": True}):
            with self.subTest(change=change):
                with self.assertRaises(AdmissionError):
                    validate_checkpoint_stop_request(argparse.Namespace(**(base | change)))

    def test_stop_flag_is_not_part_of_immutable_training_contract(self):
        binding = self.root / "binding"
        binding.write_bytes(b"pinned")
        args = argparse.Namespace(
            r16_checkpointable=True, base_binding=binding,
            producer_git_head="1d38e4f25b5663bf8a84ff5e139875d4152e3680",
            mode="correct", steps=16, cases=128, seed=41,
            learning_rate=0.00002, run_id="r16-interrupted",
            expected_initial_digest="d" * 64, stop_after_checkpoint_step=8,
        )
        preflight = {"corpus": {"manifest_sha256": "m" * 64},
                     "base_model": "Qwen/Qwen3.5-4B", "base_revision": "b" * 40,
                     "training_input_sha256": "c" * 64}
        contract = _r16_contract(args, preflight)
        args.stop_after_checkpoint_step = None
        self.assertEqual(contract, _r16_contract(args, preflight))
        self.assertEqual(contract["steps"], 16)
        self.assertNotIn("stop_after_checkpoint_step", contract)

    def test_partial_receipt_cannot_masquerade_as_completed_training(self):
        self.checkpoint()
        callback = self.make_callback()
        callback.on_save(self.args, self.state, self.control)
        receipt = make_paused_phase_receipt(
            {"mode": "correct", "steps": 16}, custody=callback.custody,
            contract_sha256=self.contract_sha, custody_file_sha256="e" * 64,
        )
        self.assertEqual(receipt["status"], "R16_STEP8_CHECKPOINT_PAUSED_NOT_COMPLETE")
        self.assertEqual(receipt["planned_optimizer_steps"], 16)
        self.assertEqual(receipt["trainer_global_step_observed"], 8)
        self.assertIsNone(receipt["new_optimizer_steps"])
        self.assertFalse(receipt["final_adapter_saved"])
        self.assertFalse(receipt["model_promotion_authorized"])
        self.assertFalse(receipt["optimizer_step_count_independently_verified"])
        self.assertEqual(receipt["optimizer_resume_equivalence"],
                         "UNVERIFIED_REQUIRES_UNINTERRUPTED_MATCH")

    def test_partial_receipt_rejects_forged_custody(self):
        self.checkpoint()
        callback = self.make_callback()
        callback.on_save(self.args, self.state, self.control)
        for change in ({"step": 16}, {"state": "RESUME_VERIFIED"},
                       {"run_contract_sha256": "9" * 64}):
            with self.subTest(change=change):
                with self.assertRaises(AdmissionError):
                    make_paused_phase_receipt(
                        {}, custody=callback.custody | change,
                        contract_sha256=self.contract_sha,
                        custody_file_sha256="e" * 64,
                    )


if __name__ == "__main__":
    unittest.main()
