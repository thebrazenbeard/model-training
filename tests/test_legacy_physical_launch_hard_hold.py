"""Legacy physical entrypoints must fail before local-spec or GPU effects.

These tests deliberately avoid a CUDA dependency, private training files,
network access, or a mock ``READY`` receipt. A caller-controlled local JSON
spec is never a substitute for a host-owned launch authorization.
"""

from __future__ import annotations

import unittest
import json
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from successor.experiments import train_v10r2_dev, train_v10r2_lane_b
from successor import train_pilot_sft
from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable,
)


_NONEXISTENT = Path("/never/read/this/private-training-spec.json")


class LegacyPhysicalLaunchHardHoldTests(unittest.TestCase):
    def _assert_dev_runner_held_before_spec_or_stack(self, module: object) -> None:
        stack_loader = Mock(side_effect=AssertionError("GPU stack was reached"))
        with patch.object(
            module,
            "load_and_validate_dev_spec",
            side_effect=AssertionError("local spec was read"),
        ) as load_spec:
            with self.assertRaisesRegex(
                PhysicalLaunchAuthorityUnavailable,
                "HARD HOLD",
            ):
                module.execute_dev_training(
                    _NONEXISTENT.parent,
                    train_jsonl=_NONEXISTENT,
                    spec_path=_NONEXISTENT,
                    training_stack_loader=stack_loader,
                )
            load_spec.assert_not_called()
            stack_loader.assert_not_called()

    def test_v10r2_dev_cannot_get_to_local_spec_or_gpu_stack(self) -> None:
        self._assert_dev_runner_held_before_spec_or_stack(train_v10r2_dev)

    def test_v10r2_lane_b_cannot_get_to_local_spec_or_gpu_stack(self) -> None:
        self._assert_dev_runner_held_before_spec_or_stack(train_v10r2_lane_b)

    def test_pilot_sft_cannot_get_to_private_data_or_model_import(self) -> None:
        with patch.object(
            train_pilot_sft,
            "prepare",
            side_effect=AssertionError("private corpus was opened"),
        ) as prepare:
            with self.assertRaisesRegex(
                PhysicalLaunchAuthorityUnavailable,
                "HARD HOLD",
            ):
                train_pilot_sft.train(_NONEXISTENT)
            prepare.assert_not_called()

    def test_v10r2_clis_preserve_json_hold_and_exit_two(self) -> None:
        for module in (train_v10r2_dev, train_v10r2_lane_b):
            with self.subTest(module=module.__name__):
                output = io.StringIO()
                with redirect_stdout(output):
                    rc = module._main(
                        [
                            "--repo-root", str(_NONEXISTENT.parent),
                            "--train-jsonl", str(_NONEXISTENT),
                            "--spec", str(_NONEXISTENT),
                        ]
                    )
                self.assertEqual(rc, 2)
                receipt = json.loads(output.getvalue())
                self.assertEqual(receipt["status"], "HOLD")
                self.assertIn("HARD HOLD", receipt["reason"])

    def test_pilot_cli_exits_two_without_training_complete(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["train_pilot_sft.py", "--spec", str(_NONEXISTENT)]):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as cm:
                    train_pilot_sft.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertTrue(output.getvalue().startswith("TRAINING_HOLD "))
        self.assertNotIn("TRAINING_COMPLETE", output.getvalue())
        self.assertEqual(json.loads(output.getvalue().split(" ", 1)[1])["status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
