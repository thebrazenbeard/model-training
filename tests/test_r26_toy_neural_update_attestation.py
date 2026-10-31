"""R26: source-only adversarial tensor/step witness, never 4B qualification."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trainingday.r26_toy_update_attestation import (
    AttestationError,
    encode_toy_adapter,
    replay_public_toy,
    verify_public_toy_control_triplet,
    verify_public_toy_run,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class R26ToyUpdateWitness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = b"PUBLIC_SYNTHETIC_BASE_V1"
        self.tokenizer = b"PUBLIC_SYNTHETIC_TOKENIZER_V1"
        self.initial = encode_toy_adapter((0.0, 0.0))
        self.pins = {
            "expected_base_sha256": sha(self.base),
            "expected_tokenizer_sha256": sha(self.tokenizer),
            "expected_initial_adapter_sha256": sha(self.initial),
        }

    def arm(self, mode, steps=None, *, changed=None, learning_rate=0.25):
        steps = 0 if mode == "zero" else (2 if steps is None else steps)
        directory = self.root / mode
        directory.mkdir()
        states = replay_public_toy((0.0, 0.0), mode, steps, learning_rate)
        final = encode_toy_adapter(states[-1] if states else (0.0, 0.0))
        files = {
            "base.bin": self.base,
            "tokenizer.txt": self.tokenizer,
            "adapter_before.bin": self.initial,
            "adapter_after.bin": final,
        }
        for name, data in files.items():
            (directory / name).write_bytes(data)
        manifest = {
            "schema": "TRAININGDAY_R26_PUBLIC_TOY_UPDATE_V1",
            "mode": mode,
            "claimed_optimizer_steps": steps,
            "learning_rate": learning_rate,
            "fixture_id": "R26_PUBLIC_ORTHOGONAL_TWO_PARAMETER_V1",
            "sha256": {name: sha(data) for name, data in files.items()},
            "claimed_step_sha256": [sha(encode_toy_adapter(state)) for state in states],
        }
        if changed is not None:
            changed(manifest, directory)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def check(self, directory):
        return verify_public_toy_run(directory, **self.pins)

    def test_actual_public_toy_tensor_change_and_fresh_process_reload(self):
        result = self.check(self.arm("correct"))
        self.assertTrue(result["physical_toy_adapter_bytes_changed"])
        self.assertTrue(result["declared_step_replay_matches"])
        self.assertTrue(result["fresh_process_toy_reload_matches"])
        self.assertLess(result["correct_label_toy_loss_after"], result["correct_label_toy_loss_before"])
        for key in ("qwen35_neural_update_verified", "behavioral_generalization_verified",
                    "host_lease_authenticated", "authorizes_gpu_start", "authorizes_promotion"):
            self.assertFalse(result[key])

    def test_controls_are_distinct_and_reject_noop_as_learning(self):
        paths = {mode: self.arm(mode) for mode in ("zero", "correct", "corrupt")}
        result = verify_public_toy_control_triplet(paths, **self.pins)
        self.assertEqual(result["status"], "PUBLIC_TOY_CONTROL_REPLAY_ONLY")
        self.assertFalse(result["model_behavioral_improvement_verified"])
        self.assertFalse(result["qualifies_neural_training"])
        self.assertFalse(self.check(paths["zero"])["physical_toy_adapter_bytes_changed"])
        self.assertTrue(self.check(paths["corrupt"])["physical_toy_adapter_bytes_changed"])
        self.assertGreater(self.check(paths["corrupt"])["correct_label_toy_loss_after"],
                           self.check(paths["corrupt"])["correct_label_toy_loss_before"])

    def test_forged_positive_optimizer_steps_with_identical_weights_rejected(self):
        def forge(manifest, directory):
            (directory / "adapter_after.bin").write_bytes(self.initial)
            manifest["sha256"]["adapter_after.bin"] = sha(self.initial)

        with self.assertRaisesRegex(AttestationError, "optimizer replay"):
            self.check(self.arm("correct", changed=forge))

    def test_relabelled_step_count_with_new_manifest_hashes_rejected(self):
        def forge(manifest, directory):
            manifest["claimed_optimizer_steps"] = 3
            manifest["claimed_step_sha256"].append("a" * 64)

        with self.assertRaisesRegex(AttestationError, "optimizer replay"):
            self.check(self.arm("correct", changed=forge))

    def test_rehashed_fake_corrupt_labels_cannot_be_called_correct_training(self):
        def forge(manifest, directory):
            manifest["mode"] = "correct"

        with self.assertRaisesRegex(AttestationError, "optimizer replay"):
            self.check(self.arm("corrupt", changed=forge))

    def test_tamper_base_tokenizer_and_initial_adapter_external_pins(self):
        for target in ("base.bin", "tokenizer.txt", "adapter_before.bin"):
            with self.subTest(target=target):
                name = target.split(".")[0]
                directory = self.root / name
                directory.mkdir()
                original = self.arm("correct") if not (self.root / "correct").exists() else self.root / "correct"
                # Copy an exact fixture under a fresh path for each adversarial attempt.
                for file in original.iterdir():
                    (directory / file.name).write_bytes(file.read_bytes())
                mutated = (directory / target).read_bytes() + b"X"
                (directory / target).write_bytes(mutated)
                manifest = json.loads((directory / "manifest.json").read_text())
                manifest["sha256"][target] = sha(mutated)
                (directory / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(AttestationError, "external pin"):
                    self.check(directory)

    def test_manifests_malformed_boolean_steps_nan_and_duplicate_keys_fail(self):
        for value in (True, -1, 17, 9999999999999999999):
            with self.subTest(steps=value):
                directory = self.arm("correct") if not (self.root / "correct").exists() else self.root / "correct"
                manifest = json.loads((directory / "manifest.json").read_text())
                manifest["claimed_optimizer_steps"] = value
                (directory / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(AttestationError):
                    self.check(directory)
                (directory / "manifest.json").write_text(json.dumps({**manifest, "claimed_optimizer_steps": 2}))
        directory = self.root / "correct"
        good = json.loads((directory / "manifest.json").read_text())
        good["learning_rate"] = float("nan")
        (directory / "manifest.json").write_text(json.dumps(good))
        with self.assertRaises(AttestationError):
            self.check(directory)
        (directory / "manifest.json").write_text('{"schema":"one","schema":"two"}')
        with self.assertRaises(AttestationError):
            self.check(directory)

    def test_forged_zero_control_with_nonzero_steps_is_rejected(self):
        def forge(manifest, directory):
            manifest["claimed_optimizer_steps"] = 1
            manifest["claimed_step_sha256"] = ["a" * 64]

        with self.assertRaises(AttestationError):
            self.check(self.arm("zero", changed=forge))

    def test_symlinked_provenance_rejected(self):
        directory = self.arm("correct")
        path = directory / "base.bin"
        path.unlink()
        path.symlink_to(self.root / "escape")
        with self.assertRaises(AttestationError):
            self.check(directory)

    def test_enormous_integer_learning_rate_rejected_without_overflow(self):
        for rate in (10**1000, -(10**1000), True, float("nan"), float("inf")):
            with self.subTest(rate=type(rate).__name__):
                with self.assertRaises(AttestationError):
                    replay_public_toy((0.0, 0.0), "correct", 1, rate)

    def test_enormous_integer_tensor_values_rejected_without_overflow(self):
        for value in (10**1000, -(10**1000), True, float("nan"), float("inf")):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(AttestationError):
                    encode_toy_adapter((value, 0.0))

    def test_unmatched_correct_corrupt_step_budgets_cannot_compare(self):
        paths = {
            "zero": self.arm("zero"),
            "correct": self.arm("correct", steps=1),
            "corrupt": self.arm("corrupt", steps=16),
        }
        with self.assertRaisesRegex(AttestationError, "unmatched public toy control schedule"):
            verify_public_toy_control_triplet(paths, **self.pins)

    def test_unmatched_learning_rates_cannot_compare(self):
        paths = {
            "zero": self.arm("zero"),
            "correct": self.arm("correct", learning_rate=0.25),
            "corrupt": self.arm("corrupt", learning_rate=0.125),
        }
        with self.assertRaisesRegex(AttestationError, "unmatched public toy control schedule"):
            verify_public_toy_control_triplet(paths, **self.pins)


if __name__ == "__main__":
    unittest.main()
