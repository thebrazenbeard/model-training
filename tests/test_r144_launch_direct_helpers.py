"""Source-only adversarial checks of direct R18 helper entrypoints.

No subprocess, CUDA stack, model, or host is ever invoked by these tests.
"""

import builtins
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday import r18_mezo, r18_qlora
from trainingday.physical_launch_authority import PhysicalLaunchAuthorityUnavailable


class DirectHelperLaunchHoldTests(unittest.TestCase):
    def test_mezo_nvml_rejects_before_host_probe(self):
        with patch.object(r18_mezo.subprocess, "run",
                          side_effect=AssertionError("nvidia-smi spawned")) as runner:
            with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                r18_mezo._nvml_used_bytes()
            runner.assert_not_called()

    def test_qlora_probe_sweep_rejects_before_subprocess(self):
        args = SimpleNamespace(base_binding=Path("unused"), seed=1,
                               learning_rate=0.001, initial_adapter=None)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(r18_qlora.subprocess, "run",
                              side_effect=AssertionError("subprocess called")) as runner:
                with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                    r18_qlora.probe_sweep(args, Path(folder), {"probe_budget_seconds": 1})
                runner.assert_not_called()

    def test_qlora_training_loop_rejects_before_torch_import(self):
        ordinary_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("torch imported before host authority")
            return ordinary_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_torch):
            with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                r18_qlora._training_loop(SimpleNamespace(), Path("unused"), [], {}, {})

    def test_mezo_fit_sweep_rejects_before_output_and_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_root=Path(folder), run_id="adversarial",
                                   plan=Path("unused"), manifest=Path("unused"),
                                   base_binding=Path("unused"), work_root=Path("unused"),
                                   selection_v2_binding=Path("unused"), mode="zero",
                                   seed=1, budget_seconds=1, learning_rate=0.001,
                                   epsilon=0.001, initial_adapter=None)
            with patch.object(r18_mezo, "_read_json",
                              return_value={"probe_budget_seconds": 1}):
                with patch.object(r18_mezo.subprocess, "run",
                                  side_effect=AssertionError("subprocess called")) as runner:
                    with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                        r18_mezo.run_fit_sweep(args, {"plan_sha256": "0" * 64})
                    runner.assert_not_called()
            self.assertFalse((Path(folder) / "adversarial").exists())

    def test_mezo_prepare_run_rejects_before_directory_write(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_root=Path(folder), run_id="adversarial",
                                   resume_checkpoint=None, checkpoint_every=1,
                                   expected_initial_digest="0" * 64)
            with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                r18_mezo.prepare_run(args, {"source_admission_status": "SOURCE_ONLY"})
            self.assertFalse((Path(folder) / "adversarial").exists())


if __name__ == "__main__":
    unittest.main()
