"""Negative checkpoint-shape tests; full 2-step CPU witness runs in CI."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trainingday import qwen35_full32_trl_optimizer_resume_cpu as witness
from trainingday import qwen35_full32_trl_integrated_cpu as integrated


class OptimizerResumeCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="trainingday-resume-manifest-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / witness.VERSIONED_ADAPTER).mkdir()
        self.files = [f"{witness.VERSIONED_ADAPTER}/adapter_model.safetensors",
                      f"{witness.VERSIONED_ADAPTER}/adapter_config.json", witness.OPTIMIZER]
        for f in self.files:
            (self.root / f).write_bytes(f.encode())
        self.manifest = {
            "schema": witness.SCHEMA, "tokenizer_sha256": integrated.TOKENIZER_SHA,
            "model_seed": integrated.BASE_SEED, "step": 1, "cursor": 1,
            "lr": witness.LR, "before_second_batch_sha256": {
                "input_ids": "a" * 64, "labels": "b" * 64, "attention_mask": "c" * 64},
            "initial_frozen_sha256": "d" * 64, "config_sha256": "e" * 64,
            "rng_state_sha256": "f" * 64,
            "runtime": {"torch": "2.14+cpu", "transformers": "5.x", "trl": "0.x",
                        "peft": "0.x", "tokenizers": "0.x", "cpu_capability": "AVX2",
                        "threads": 2, "mkldnn": True, "deterministic_algorithms": False},
            "trainable_names": [f"layer.{i}.lora_A.weight" for i in range(152)] +
                               [f"layer.{i}.lora_B.weight" for i in range(152)],
            "files": {f: hashlib.sha256(f.encode()).hexdigest() for f in self.files},
        }
        self._write()

    def _write(self):
        (self.root / witness.MANIFEST).write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_integrity_positive(self):
        self.assertEqual(witness._validate_checkpoint(self.root), self.manifest)

    def test_corrupted_optimizer_file_rejected(self):
        (self.root / witness.OPTIMIZER).write_bytes(b"malicious")
        with self.assertRaisesRegex(ValueError, "checkpoint file corrupted"):
            witness._validate_checkpoint(self.root)

    def test_stale_cursor_and_step_rejected(self):
        for field in ("cursor", "step"):
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = 0
                self._write()
                with self.assertRaisesRegex(ValueError, "version/step"):
                    witness._validate_checkpoint(self.root)
                self.manifest[field] = original

    def test_reject_ambiguous_training_tensor_list(self):
        self.manifest["trainable_names"][0] = self.manifest["trainable_names"][1]
        self._write()
        with self.assertRaisesRegex(ValueError, "identities not admitted"):
            witness._validate_checkpoint(self.root)

    def test_reject_removed_runtime_and_extra_file_binding(self):
        original = copy.deepcopy(self.manifest)
        del self.manifest["runtime"]["torch"]
        self._write()
        with self.assertRaisesRegex(ValueError, "runtime fingerprint missing"):
            witness._validate_checkpoint(self.root)
        self.manifest = original
        self.manifest["files"]["untrusted.bin"] = "0" * 64
        self._write()
        with self.assertRaisesRegex(ValueError, "required checkpoint files missing"):
            witness._validate_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
