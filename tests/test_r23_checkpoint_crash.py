"""R23 synthetic crash/restart custody tests; no model, CUDA, or real adapter data.

These exercise the low-level checkpoint helpers only. Strict R18 equal-time
training rejects resume and is NOT made resumable by these tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.r18_mezo import canonical_hash, prepare_run, restore_checkpoint


class CrashCustodyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = argparse.Namespace(
            output_root=self.root, run_id="synthetic-crash", steps=16,
            checkpoint_every=0, expected_initial_digest="a" * 64,
            resume_checkpoint=None, resume_receipt_sha256=None,
            budget_seconds=531,
        )
        self.preflight = {"recipe": "mezo", "selected_id_sha256": "b" * 64}

    def checkpoint(self):
        output, contract, *_ = prepare_run(self.args, self.preflight)
        directory = output / "checkpoint-8"
        directory.mkdir()
        weights = directory / "lora-trainables.safetensors"
        weights.write_bytes(b"synthetic-safetensors-placeholder")
        state = {
            "schema": "R18_MEZO_STATE_V1",
            "status": "CHECKPOINT_BYTES_PINNED_NOT_RELOADED",
            "run_contract_sha256": canonical_hash(contract),
            "completed_steps": 8,
            "optimizer_window_seconds": 120.0,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "counters": {"example_presentations": 64},
        }
        state_path = directory / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        self.args.resume_checkpoint = directory
        self.pin_receipt(state_path)
        return directory, state

    def pin_receipt(self, state_path):
        self.args.resume_receipt_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()

    def test_crash_before_contract_is_fail_closed_on_restart(self):
        occupied = self.root / self.args.run_id
        occupied.mkdir()
        (occupied / "run-contract.json").write_bytes(b'{"incomplete":')
        with self.assertRaisesRegex(AdmissionError, "already occupied"):
            prepare_run(self.args, self.preflight)
        self.args.resume_checkpoint = occupied / "checkpoint-8"
        self.args.resume_receipt_sha256 = "f" * 64
        with self.assertRaises(AdmissionError):
            prepare_run(self.args, self.preflight)

    def test_crash_after_weights_before_state_is_rejected(self):
        directory, _ = self.checkpoint()
        (directory / "state.json").unlink()
        with self.assertRaisesRegex(AdmissionError, "state missing"):
            prepare_run(self.args, self.preflight)

    def test_truncated_state_with_pinned_truncated_hash_is_rejected(self):
        directory, _ = self.checkpoint()
        state_path = directory / "state.json"
        state_path.write_bytes(b'{"schema":"R18_MEZO_STATE_V1",')
        self.pin_receipt(state_path)
        with self.assertRaisesRegex(AdmissionError, "invalid checkpoint JSON"):
            prepare_run(self.args, self.preflight)

    def test_truncated_weights_with_original_state_is_rejected(self):
        directory, _ = self.checkpoint()
        (directory / "lora-trainables.safetensors").write_bytes(b"cut")
        with self.assertRaisesRegex(AdmissionError, "custody mismatch"):
            prepare_run(self.args, self.preflight)

    def test_walltime_exceeds_cap_even_when_receipt_rehashed(self):
        directory, state = self.checkpoint()
        state["optimizer_window_seconds"] = 532
        state_path = directory / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        self.pin_receipt(state_path)
        with self.assertRaisesRegex(AdmissionError, "wall-time budget malformed"):
            prepare_run(self.args, self.preflight)

    def test_nonfinite_walltime_rejected_even_when_receipt_rehashed(self):
        directory, state = self.checkpoint()
        state["optimizer_window_seconds"] = float("nan")
        state_path = directory / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        self.pin_receipt(state_path)
        with self.assertRaisesRegex(AdmissionError, "wall-time budget malformed"):
            prepare_run(self.args, self.preflight)


class FakeTensor:
    def __init__(self, values, *, shape=(2,), dtype="fp32"):
        self.values = list(values)
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"

    def to(self, *, device):
        if device != "cpu":
            raise ValueError("synthetic tensors only")
        return self

    def copy_(self, other):
        self.values = list(other.values)


class RestorePreflightTests(unittest.TestCase):
    def test_destination_dtype_mismatch_never_changes_any_tensor(self):
        first = FakeTensor([1, 2])
        second = FakeTensor([3, 4], dtype="fp16")
        loaded = {
            "first": FakeTensor([10, 20]),
            "second": FakeTensor([30, 40]),
        }
        fake_torch = types.SimpleNamespace(float32="fp32", no_grad=nullcontext)
        fake_safetensors_torch = types.ModuleType("safetensors.torch")
        fake_safetensors_torch.load_file = lambda _path, device: loaded
        fake_safetensors = types.ModuleType("safetensors")
        fake_safetensors.torch = fake_safetensors_torch
        with patch.dict(sys.modules, {
            "safetensors": fake_safetensors,
            "safetensors.torch": fake_safetensors_torch,
        }):
            with self.assertRaisesRegex(AdmissionError, "shape/dtype changed"):
                restore_checkpoint(
                    [("first", first), ("second", second)], fake_torch,
                    Path("synthetic-checkpoint"), {},
                )
        self.assertEqual(first.values, [1, 2])
        self.assertEqual(second.values, [3, 4])

    def test_late_tensor_mismatch_never_partially_changes_previous_tensor(self):
        first = FakeTensor([1, 2])
        second = FakeTensor([3, 4])
        loaded = {
            "first": FakeTensor([10, 20]),
            "second": FakeTensor([30, 40], shape=(3,)),
        }
        fake_torch = types.SimpleNamespace(float32="fp32", no_grad=nullcontext)
        fake_safetensors_torch = types.ModuleType("safetensors.torch")
        fake_safetensors_torch.load_file = lambda _path, device: loaded
        fake_safetensors = types.ModuleType("safetensors")
        fake_safetensors.torch = fake_safetensors_torch
        with patch.dict(sys.modules, {
            "safetensors": fake_safetensors,
            "safetensors.torch": fake_safetensors_torch,
        }):
            with self.assertRaisesRegex(AdmissionError, "shape/dtype changed"):
                restore_checkpoint(
                    [("first", first), ("second", second)], fake_torch,
                    Path("synthetic-checkpoint"), {},
                )
        self.assertEqual(first.values, [1, 2], "earlier tensor mutated before validation ended")
        self.assertEqual(second.values, [3, 4])


if __name__ == "__main__":
    unittest.main()
