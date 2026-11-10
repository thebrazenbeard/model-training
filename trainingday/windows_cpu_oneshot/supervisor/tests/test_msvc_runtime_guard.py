"""Offline fixture tests; no native DLL, runner or host process is executed."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import msvc_runtime_guard as g


def fixture(version=(14, 50, 0, 0)):
    return {
        "host": "VERA-NODE-1", "architecture": "AMD64", "process_bits": 64,
        "runner_sha256": g.RUNNER_SHA256,
        "app_local_runtime_files": [],
        "registry": {"view": "Registry64", "installed": 1,
                     "version": "v" + ".".join(map(str, version))},
        "runtime_files": {
            name: {"system32": True, "pe_machine": "AMD64",
                   "version_parts": list(version), "signature_status": "Valid",
                   "signer_subject": "CN=Microsoft Corporation, O=Microsoft Corporation, C=US"}
            for name in g.RUNTIME_FILES
        },
    }


# This is ONLY a test fixture. There is no reviewed build-run attestation for
# the real runner; shipped CURRENT_CONTRACT must continue to HOLD.
TEST_PROVENANCE = g.BuildContract(
    g.RUNNER_SHA256, "14.40.0.0", "a" * 64, "BUILD_RUN_VERIFIED",
)


class ParserTests(unittest.TestCase):
    def test_numeric_comparison_not_string_comparison(self):
        self.assertLess(g.parse_version("14.9.0.0"), g.parse_version("14.50.0.0"))

    def test_windows_registry_zero_padding_is_explicit_exception(self):
        self.assertEqual(g.parse_registry_version("v14.32.31332.00"), (14, 32, 31332, 0))
        with self.assertRaises(g.Hold):
            g.parse_version("v14.32.31332.00")

    def test_strict_parser_rejects_malformed(self):
        for value in (None, True, 14.4, "14.40", "14.40.0", "14.40.0.0.0",
                      " 14.40.0.0", "14.40.0.0 ", "V14.40.0.0", "-14.40.0.0",
                      "14.040.0.0", "14.40.0.00", "14.40.x.0", "14.40.0.-1",
                      "14.40.0.4294967296", "14.40.0.０", "14.40.0." + "9" * 5000):
            with self.subTest(value=value), self.assertRaises(g.Hold):
                g.parse_version(value)


class GateTests(unittest.TestCase):
    def check_hold(self, evidence, reason, contract=TEST_PROVENANCE):
        result = g.evaluate(evidence, contract)
        self.assertEqual(result["state"], "HOLD")
        self.assertTrue(any(text.startswith(reason) for text in result["reasons"]), result)

    def test_current_unverified_build_always_holds(self):
        self.check_hold(fixture(), "exact_build_run_provenance_unknown", g.CURRENT_CONTRACT)
        self.check_hold(fixture(), "unknown_or_invalid_required_runtime_minimum", g.CURRENT_CONTRACT)

    def test_source_only_claimed_minimum_still_holds(self):
        claimed = g.BuildContract(g.RUNNER_SHA256, "14.1.0.0", "a" * 64, "SOURCE_ONLY")
        self.check_hold(fixture(), "exact_build_run_provenance_unknown", claimed)

    def test_fixture_can_only_pass_prerequisite(self):
        result = g.evaluate(fixture(), TEST_PROVENANCE)
        self.assertEqual(result["state"], "ALLOW_VERSION_PREREQUISITE_ONLY")
        self.assertEqual(result["scope"], "version_prerequisite_only")

    def test_old_vn1_version_holds_against_a_hypothetical_proven_minimum(self):
        self.check_hold(fixture((14, 32, 31332, 0)), "runtime_file_too_old:")
        self.check_hold(fixture((14, 32, 31332, 0)), "redist_registry_too_old")

    def test_missing_or_wrong_registry(self):
        for key, value in (("installed", 0), ("installed", True),
                           ("view", "Registry32"), ("version", None),
                           ("version", "14.40")):
            with self.subTest(key=key):
                evidence = fixture()
                evidence["registry"][key] = value
                self.assertEqual(g.evaluate(evidence, TEST_PROVENANCE)["state"], "HOLD")

    def test_file_version_mismatch(self):
        evidence = fixture()
        evidence["runtime_files"]["MSVCP140.dll"]["version_parts"] = [14, 41, 0, 0]
        self.check_hold(evidence, "registry_runtime_file_version_mismatch:")

    def test_no_file_missing_allowed(self):
        evidence = fixture()
        del evidence["runtime_files"]["VCRUNTIME140_1.dll"]
        self.check_hold(evidence, "runtime_file_set_unknown")
        self.check_hold(evidence, "runtime_file_missing:")

    def test_wrong_file_arch_or_location(self):
        for field, value in (("system32", False), ("pe_machine", "I386")):
            evidence = fixture()
            evidence["runtime_files"]["MSVCP140.dll"][field] = value
            self.check_hold(evidence, "runtime_file_location_or_arch_invalid:")

    def test_signer_must_be_verified_microsoft(self):
        for field, value in (("signature_status", "UnknownError"),
                             ("signer_subject", "CN=Not Microsoft Corporation LLC"),
                             ("signer_subject", None)):
            evidence = fixture()
            evidence["runtime_files"]["MSVCP140.dll"][field] = value
            self.check_hold(evidence, "runtime_file_signature_unverified:")

    def test_app_local_runtime_shadow_holds(self):
        evidence = fixture()
        evidence["app_local_runtime_files"] = ["MSVCP140.dll"]
        self.check_hold(evidence, "app_local_runtime_shadow_or_unknown")

    def test_host_and_process_arch_holds(self):
        for field, value in (("host", "VN2"), ("architecture", "I386"),
                             ("process_bits", 32)):
            evidence = fixture()
            evidence[field] = value
            self.assertEqual(g.evaluate(evidence, TEST_PROVENANCE)["state"], "HOLD")

    def test_runner_sha_mismatch_holds(self):
        evidence = fixture()
        evidence["runner_sha256"] = "0" * 64
        self.check_hold(evidence, "runner_digest_mismatch")

    def test_pinned_runner_digest_changed_holds(self):
        claimed = g.BuildContract("0" * 64, "14.40.0.0", "a" * 64, "BUILD_RUN_VERIFIED")
        self.check_hold(fixture(), "artifact_digest_not_approved", claimed)

    def test_unknown_evidence_fails_closed(self):
        self.check_hold(None, "missing_host_evidence")
        self.check_hold({}, "wrong_host_or_architecture")

    def test_bad_file_version_data(self):
        for bad in ([14, 50, 0], [14, 50, True, 0], [14, 50, -1, 0], "14.50.0.0"):
            with self.subTest(value=bad):
                evidence = fixture()
                evidence["runtime_files"]["MSVCP140.dll"]["version_parts"] = bad
                self.check_hold(evidence, "runtime_file_version_invalid:")

    def test_collector_failure_no_launcher_or_state_creation(self):
        with patch.object(g, "collect_windows_evidence", side_effect=g.Hold("offline")):
            self.assertEqual(g.main(TEST_PROVENANCE), 3)

    def test_unverified_real_contract_does_not_even_spawn_collector(self):
        with patch.object(g, "collect_windows_evidence") as collector:
            self.assertEqual(g.main(), 3)
            collector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
