"""No-host, no-model regressions for the negative-only conversion screen."""

import json
import subprocess
import sys
import unittest

from trainingday.offline_conversion_capacity import (
    CapacityInputError, SCHEMA, evaluate_sample, parse_sample,
)


def sample(**changes):
    data = {
        "schema": SCHEMA,
        "host_id": "LAPPY",
        "environment": "windows_wsl2",
        "observed_utc": "2026-11-05T05:31:53Z",
        "host_ram_available_mib": 5878,
        "guest_ram_available_mib": 15105,
        "swap_available_mib": 7441,
        "staging_free_mib": 426 * 1024,
    }
    data.update(changes)
    return data


class CapacityGateTests(unittest.TestCase):
    def test_wsl_guest_surplus_does_not_override_windows_host(self):
        outcome = evaluate_sample(sample())
        self.assertEqual(outcome["numeric_capacity"], "HOLD")
        self.assertIn("HOST_RAM_BELOW_12_GIB", outcome["reasons"])
        self.assertFalse(outcome["launch_authorized"])

    def test_vn2_swap_and_ram_both_fail(self):
        outcome = evaluate_sample(sample(host_id="VN2", environment="linux",
            host_ram_available_mib=6247, swap_available_mib=46,
            guest_ram_available_mib=None, staging_free_mib=62 * 1024))
        self.assertIn("SWAP_BELOW_1_GIB", outcome["reasons"])
        self.assertIn("HOST_RAM_BELOW_12_GIB", outcome["reasons"])

    def test_numeric_pass_never_authorizes(self):
        outcome = evaluate_sample(sample(host_ram_available_mib=20000,
            guest_ram_available_mib=20000))
        self.assertEqual(outcome["numeric_capacity"], "SCREEN_PASS_ONLY")
        self.assertFalse(outcome["launch_authorized"])
        self.assertFalse(outcome["owner_lease_verified"])

    def test_forced_lease_claim_rejected(self):
        with self.assertRaises(CapacityInputError):
            evaluate_sample(sample(host_owned_lease=True))

    def test_missing_host_memory_not_substituted_by_guest(self):
        data = sample()
        del data["host_ram_available_mib"]
        with self.assertRaises(CapacityInputError):
            evaluate_sample(data)

    def test_fake_utc_timestamp_rejected(self):
        with self.assertRaises(CapacityInputError):
            evaluate_sample(sample(observed_utc="zzzz-zz-zzTzz:zz:zzZ"))

    def test_boolean_numeric_forgery_rejected(self):
        with self.assertRaises(CapacityInputError):
            evaluate_sample(sample(host_ram_available_mib=True))

    def test_duplicate_json_key_rejected(self):
        raw = b'{"schema":"TRAININGDAY_OFFLINE_CONVERSION_CAPACITY_V1",' \
              b'"host_id":"LAPPY","host_id":"VN1"}'
        with self.assertRaises(CapacityInputError):
            parse_sample(raw)

    def test_unexpected_fields_rejected(self):
        with self.assertRaises(CapacityInputError):
            parse_sample(json.dumps(sample(secret="x")).encode())

    def test_disk_shortfall_holds(self):
        result = evaluate_sample(sample(staging_free_mib=5 * 1024))
        self.assertIn("STAGING_BELOW_25_GIB", result["reasons"])

    def test_cli_is_never_a_positive_launch_authority(self):
        ready = sample(host_ram_available_mib=20000,
                       guest_ram_available_mib=20000)
        process = subprocess.run(
            [sys.executable, "-B", "-m", "trainingday.offline_conversion_capacity"],
            input=json.dumps(ready).encode(), capture_output=True, check=False,
            timeout=5,
        )
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result["numeric_capacity"], "SCREEN_PASS_ONLY")
        self.assertFalse(result["launch_authorized"])


if __name__ == "__main__":
    unittest.main()
