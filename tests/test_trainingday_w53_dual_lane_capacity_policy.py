"""W53: preserve distinct no-launch Q4, plan, and conversion gates.

Pure source-policy assertions. These tests do not inspect weights, open a
model, reserve host capacity, or establish process ownership.
"""

import unittest

from trainingday.offline_conversion_capacity import (
    RAM_MIN_MIB, SCHEMA, evaluate_sample,
)
from trainingday.official_inference_admission import (
    INFERENCE_MIN_AVAILABLE_RAM, STAGE_DISK_RESERVE,
)
from trainingday.real_qwen35_inference import (
    InferenceHold, probe, require_probe_resources,
)
from unittest.mock import patch


GIB = 1024 ** 3


class W53ThreeLaneBoundaryTests(unittest.TestCase):
    def test_policies_remain_separate(self):
        self.assertEqual(INFERENCE_MIN_AVAILABLE_RAM, 6 * GIB)
        self.assertEqual(STAGE_DISK_RESERVE, 24 * GIB)
        self.assertEqual(RAM_MIN_MIB, 12 * 1024)

    def test_q4_2783446880_bytes_has_five_gib_disabled_probe_floor(self):
        # This is an example candidate size from an external publication,
        # not an authenticated locally present official-derived GGUF.
        q4_candidate_size = 2_783_446_880
        self.assertEqual(max(5 * GIB, int(q4_candidate_size * 1.3) + GIB),
                         5 * GIB)
        preflight = {"base_gguf_bytes": q4_candidate_size}
        with patch("trainingday.real_qwen35_inference.available_ram_bytes",
                   return_value=5 * GIB - 1):
            with self.assertRaises(InferenceHold):
                require_probe_resources(preflight, host_idle_acknowledged=True)
        with patch("trainingday.real_qwen35_inference.available_ram_bytes",
                   return_value=5 * GIB):
            self.assertEqual(require_probe_resources(
                preflight, host_idle_acknowledged=True), 5 * GIB)

    def test_q4_five_gib_numeric_pass_does_not_make_probe_callable(self):
        with self.assertRaisesRegex(InferenceHold, "runtime probe disabled"):
            probe()
        self.assertLess(5 * GIB, INFERENCE_MIN_AVAILABLE_RAM)

    def test_vn1_plausible_q4_headroom_never_licenses_bf16_conversion(self):
        # A fictional, caller-declared 6.8 GiB host sample. This is not a
        # live telemetry receipt or proof a real Q4 model is installed.
        sample = {
            "schema": SCHEMA,
            "host_id": "SYNTHETIC_VN1",
            "environment": "windows",
            "observed_utc": "2026-11-05T05:00:00Z",
            "host_ram_available_mib": 6963,
            "swap_available_mib": 8192,
            "staging_free_mib": 62 * 1024,
        }
        report = evaluate_sample(sample)
        self.assertGreater(sample["host_ram_available_mib"] * 1024**2,
                           INFERENCE_MIN_AVAILABLE_RAM)
        self.assertEqual(report["numeric_capacity"], "HOLD")
        self.assertIn("HOST_RAM_BELOW_12_GIB", report["reasons"])
        self.assertFalse(report["launch_authorized"])
        self.assertFalse(report["model_bytes_verified"])
        self.assertFalse(report["owner_lease_verified"])

    def test_even_numeric_conversion_surplus_cannot_grant_inference(self):
        report = evaluate_sample({
            "schema": SCHEMA,
            "host_id": "SYNTHETIC_OWNER_HOST",
            "environment": "linux",
            "observed_utc": "2026-11-05T05:00:00Z",
            "host_ram_available_mib": 64 * 1024,
            "swap_available_mib": 8 * 1024,
            "staging_free_mib": 100 * 1024,
        })
        self.assertEqual(report["numeric_capacity"], "SCREEN_PASS_ONLY")
        self.assertFalse(report["launch_authorized"])
        self.assertFalse(report["physical_inference_or_conversion_performed"])


if __name__ == "__main__":
    unittest.main()
