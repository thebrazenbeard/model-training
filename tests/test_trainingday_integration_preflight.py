"""This integration report must never promote source to physical readiness."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainingday import integration_preflight


class IntegrationPreflightTests(unittest.TestCase):
    def test_gaps_are_ordered_and_provenance_typed(self):
        value = integration_preflight.report()
        self.assertFalse(value["ready"])
        self.assertFalse(value["physical_launch_attempted"])
        self.assertEqual(value["schema"], "TRAININGDAY_INTEGRATION_PREFLIGHT_V1")
        self.assertEqual(value["model_id"], "Qwen/Qwen3.5-4B")
        steps = value["steps"]
        indices = {step["name"]: i for i, step in enumerate(steps)}
        self.assertEqual(len(indices), 11)
        for step in steps:
            self.assertIn(step["status"], ("SOURCE_ONLY", "HOLD", "UNKNOWN"))
            self.assertNotEqual(step["evidence_class"], "RUNTIME_PROBE")
            for dependency in step["depends_on"]:
                self.assertIn(dependency, indices)
                self.assertLess(indices[dependency], indices[step["name"]])
        self.assertEqual(steps[0]["status"], "SOURCE_ONLY")
        self.assertEqual(steps[5]["status"], "HOLD")
        self.assertLess(indices["independent_artifact_restore"],
                        indices["trained_vera_conversation"])
        self.assertNotIn("trained_vera_conversation",
                         steps[3]["depends_on"])
        self.assertNotIn("independent_protected_evaluation",
                         steps[10]["depends_on"])

    def test_metadata_enrollment_does_not_claim_weight_or_host_admission(self):
        required = integration_preflight.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS
        with patch.object(integration_preflight, "TRUSTED_OFFICIAL_METADATA_PIN_ITEMS",
                          tuple((name, "a" * 64) for name in required)):
            value = integration_preflight.report()
        self.assertTrue(value["source_metadata_pins_enrolled"])
        self.assertEqual(value["steps"][1]["status"], "UNKNOWN")
        self.assertEqual(value["steps"][5]["status"], "HOLD")
        self.assertFalse(value["ready"])

    def test_fake_positive_launcher_fails_closed(self):
        with patch.object(integration_preflight, "require_physical_launch_authority",
                          return_value=None):
            with self.assertRaises(integration_preflight.UnexpectedAuthorityChange):
                integration_preflight.report()

    def test_mismatched_official_revision_fails_closed(self):
        with patch.object(integration_preflight, "BASE_REVISION", "other-revision"):
            with self.assertRaises(integration_preflight.SourceIdentityMismatch):
                integration_preflight.report()

    def test_cli_json_is_parseable_and_denies_release(self):
        run = subprocess.run(
            [sys.executable, "-m", "trainingday.integration_preflight", "--json"],
            capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(run.returncode, 1)
        self.assertEqual(run.stderr, "")
        value = json.loads(run.stdout)
        self.assertFalse(value["ready"])
        self.assertIn("official adapter GGUF conversion", value["warning"])


if __name__ == "__main__":
    unittest.main()
