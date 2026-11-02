"""Read-only regression checks for the Lappy WSL2 CUDA visibility route."""

import subprocess
import unittest
from unittest.mock import patch

from trainingday.lappy_wsl2_preflight import (
    inspect_wsl_nvidia_gpu,
    is_wsl2,
    parse_gpu_row,
)


class Wsl2PreflightTests(unittest.TestCase):
    def test_detects_wsl2_and_rejects_native_linux(self):
        self.assertTrue(is_wsl2("Linux version 6.6.87.2-microsoft-standard-WSL2"))
        self.assertFalse(is_wsl2("Linux version 6.8.0-generic"))
        self.assertFalse(is_wsl2("Linux version microsoft WSL1"))

    def test_gpu_row_parses_wsl_nvidia_output(self):
        self.assertEqual(
            parse_gpu_row("NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 617.42\n"),
            {"gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
             "vram_mib": 4096, "driver_version": "617.42"},
        )

    def test_malformed_output_fails_closed(self):
        for raw in ("", "NVIDIA, four GB, unknown", "GPU, 4096 MiB", "a, 0 MiB, 1"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_gpu_row(raw)

    @patch("trainingday.lappy_wsl2_preflight.subprocess.run")
    @patch("trainingday.lappy_wsl2_preflight.Path.exists", return_value=True)
    def test_visibility_is_not_training_authority(self, _exists, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 617.42\n",
            stderr="",
        )
        result = inspect_wsl_nvidia_gpu("Linux version 6.6-microsoft-standard-WSL2")
        self.assertEqual(result["status"], "NVIDIA_GPU_VISIBLE_TRAINING_HOLD")
        self.assertTrue(result["wsl2_detected"])
        self.assertTrue(result["nvidia_gpu_visible"])
        self.assertFalse(result["data_rights_granted"])
        self.assertFalse(result["gpu_lease_granted"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["vram_mib"], 4096)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    @patch("trainingday.lappy_wsl2_preflight.subprocess.run")
    @patch("trainingday.lappy_wsl2_preflight.Path.exists", return_value=True)
    def test_gpu_probe_failure_never_claims_readiness(self, _exists, run):
        run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
        result = inspect_wsl_nvidia_gpu("Linux version microsoft-standard-WSL2")
        self.assertEqual(result["status"], "NVIDIA_GPU_NOT_VERIFIED_TRAINING_HOLD")
        self.assertFalse(result["nvidia_gpu_visible"])
        self.assertFalse(result["training_authorized"])

    @patch("trainingday.lappy_wsl2_preflight.subprocess.run")
    def test_non_wsl_does_not_invoke_gpu_probe(self, run):
        result = inspect_wsl_nvidia_gpu("Linux version 6.8.0-generic")
        self.assertEqual(result["status"], "NOT_WSL2_TRAINING_HOLD")
        run.assert_not_called()

    @patch("trainingday.lappy_wsl2_preflight.subprocess.run")
    @patch("trainingday.lappy_wsl2_preflight.Path.exists", return_value=False)
    def test_missing_known_wsl_utility_does_not_search_path(self, _exists, run):
        result = inspect_wsl_nvidia_gpu("Linux version microsoft-standard-WSL2")
        self.assertFalse(result["nvidia_gpu_visible"])
        self.assertEqual(result["status"], "NVIDIA_GPU_NOT_VERIFIED_TRAINING_HOLD")
        run.assert_not_called()

    @patch("trainingday.lappy_wsl2_preflight.subprocess.run")
    @patch("trainingday.lappy_wsl2_preflight.Path.exists", return_value=True)
    def test_failed_nvidia_utility_does_not_claim_visibility(self, _exists, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="GPU, 4096 MiB, 617.42", stderr="error",
        )
        result = inspect_wsl_nvidia_gpu("Linux version microsoft-standard-WSL2")
        self.assertFalse(result["nvidia_gpu_visible"])
        self.assertFalse(result["training_authorized"])


if __name__ == "__main__":
    unittest.main()
