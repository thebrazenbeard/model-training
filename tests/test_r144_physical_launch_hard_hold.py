"""Source-only hard-HOLD tests: never import CUDA or touch real hosts."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday import r18_mezo, r18_qlora, train_neural
from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable,
    require_physical_launch_authority,
)


class PhysicalLaunchHardHoldTests(unittest.TestCase):
    def test_no_self_declared_input_grants_authority(self):
        with patch.dict(os.environ, {"TRAININGDAY_AUTHORIZED": "true",
                                      "VERA_GPU_LEASE": "self-signed"}):
            for operation in ("neural", "qlora", "mezo", "fit-probe"):
                with self.subTest(operation=operation):
                    with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                        require_physical_launch_authority(operation=operation)

    def test_neural_preflight_allowed_but_training_held_before_heavy_import(self):
        args = SimpleNamespace(preflight_only=True, mode="correct", steps=8,
                               r16_checkpointable=False)
        fake = ({"status": "ADMITTED_SOURCE_ONLY_NOT_TRAINED"},
                [{"prompt": "2+2", "response": "4"}],
                {"model_id": "Qwen/Qwen3.5-4B", "revision": "a" * 40},
                Path("/nonexistent-model-snapshot"))
        with patch.object(train_neural, "validate_run", return_value=fake):
            self.assertEqual(train_neural.train(args)["status"], "SOURCE_ONLY_PREFLIGHT")
            args.preflight_only = False
            with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                train_neural.train(args)

    def test_qlora_preflight_allowed_but_run_probe_and_load_held(self):
        args = SimpleNamespace(preflight_only=True, output_root=Path("/unreadable"),
                               run_id="no-output")
        fake = ({"status": "SOURCE_ONLY_PREFLIGHT"}, [], Path("/not-read"), {})
        with patch.object(r18_qlora, "preflight", return_value=fake):
            self.assertEqual(r18_qlora.run(args)["status"], "SOURCE_ONLY_PREFLIGHT")
            args.preflight_only = False
            with patch.object(r18_qlora, "probe_sweep") as probe:
                with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                    r18_qlora.run(args)
                probe.assert_not_called()
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            r18_qlora.probe_worker(SimpleNamespace())
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            r18_qlora._load_model(Path("/not-read"), 41, 0.00002)

    def test_mezo_preflight_allowed_but_fit_probe_load_held(self):
        args = SimpleNamespace(preflight_only=True, fit_sweep_only=True,
                               resume_checkpoint=None)
        fake = ({"status": "SOURCE_ONLY_PREFLIGHT"}, [], Path("/not-read"))
        with patch.object(r18_mezo, "validate_source", return_value=fake):
            self.assertEqual(r18_mezo.run(args)["status"], "SOURCE_ONLY_PREFLIGHT")
            args.preflight_only = False
            with patch.object(r18_mezo, "run_fit_sweep") as sweep:
                with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                    r18_mezo.run(args)
                sweep.assert_not_called()
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            r18_mezo._probe_child(SimpleNamespace())
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            r18_mezo._gpu_stack(SimpleNamespace(), Path("/not-read"))


if __name__ == "__main__":
    unittest.main()
