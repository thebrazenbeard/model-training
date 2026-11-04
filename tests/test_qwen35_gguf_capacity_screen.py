"""Offline negative controls for the official GGUF planning screen."""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from trainingday.qwen35_gguf_capacity_screen import (
    GIB, SHARD_BYTES, command_templates, main, screen,
)


class CapacityScreenTests(unittest.TestCase):
    def test_pins_exact_two_source_shards(self):
        self.assertEqual(SHARD_BYTES, 5_329_398_688 + 3_990_429_408)

    def test_vn1_reported_approx_six_point_nine_gib_fails_ram(self):
        report = screen(round(6.90 * GIB), round(62.56 * GIB))
        self.assertEqual(report["capacity_screen"], "SHORTFALL")
        self.assertGreater(report["ram_shortfall_against_proxy_bytes"], 0)
        self.assertEqual(report["scratch_shortfall_against_proxy_bytes"], 0)
        self.assertFalse(report["capacity_qualified"])

    def test_large_unverified_capacity_not_release(self):
        report = screen(200 * GIB, 200 * GIB)
        self.assertEqual(report["capacity_screen"], "NO_SHORTFALL_PROVEN")
        self.assertFalse(report["capacity_qualified"])
        self.assertFalse(report["physical_launch_authorized"])
        self.assertFalse(report["official_gguf_created"])

    def test_negative_bool_float_string_capacity_refused(self):
        for bad in (-1, True, 1.5, "1234", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                screen(bad, 20 * GIB)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                screen(20 * GIB, bad)

    def test_command_templates_are_explicit_offline_and_local(self):
        base, quant, adapter = command_templates()
        self.assertEqual(base[-3:], ("--outtype", "bf16", "--no-mtp"))
        self.assertEqual(quant[-1], "Q4_K_M")
        self.assertIn("--base", adapter)
        self.assertIn("<PINNED_OFFICIAL_HF_DIR>", adapter)
        self.assertFalse(any("--remote" in cmd or "--base-model-id" in cmd
                             for args in command_templates() for cmd in args))

    def test_cli_always_hold_no_subprocess_even_with_huge_numbers(self):
        with patch("subprocess.Popen") as popen, patch("os.system") as system:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--available-ram-bytes", str(200 * GIB),
                             "--free-scratch-bytes", str(200 * GIB)])
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(out.getvalue())["host_process_attempted"])
            popen.assert_not_called()
            system.assert_not_called()


if __name__ == "__main__":
    unittest.main()
