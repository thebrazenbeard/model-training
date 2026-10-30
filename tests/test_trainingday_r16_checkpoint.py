"""Dependency-free source/custody gates; NOT evidence of GPU training."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trainingday.admission import AdmissionError
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


if __name__ == "__main__":
    unittest.main()
