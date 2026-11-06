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
        self.assertEqual(value["schema"], "TRAININGDAY_INTEGRATION_PREFLIGHT_V2")
        self.assertEqual(value["model_id"], "Qwen/Qwen3.5-4B")
        steps = value["steps"]
        indices = {step["name"]: i for i, step in enumerate(steps)}
        self.assertEqual(len(indices), 13)
        for step in steps:
            self.assertIn(step["status"], ("SOURCE_ONLY", "HOLD", "UNKNOWN"))
            self.assertNotEqual(step["evidence_class"], "RUNTIME_PROBE")
            for dependency in step["depends_on"]:
                self.assertIn(dependency, indices)
                self.assertLess(indices[dependency], indices[step["name"]])
        self.assertEqual(steps[0]["status"], "SOURCE_ONLY")
        self.assertEqual(steps[5]["status"], "HOLD")
        self.assertLess(indices["independent_artifact_restore"],
                        indices["unqualified_new_run_conversation_probe"])
        self.assertNotIn("trained_vera_conversation", indices)
        self.assertNotIn("unqualified_new_run_conversation_probe",
                         steps[3]["depends_on"])
        self.assertNotIn("independent_protected_evaluation",
                         steps[indices["unqualified_new_run_conversation_probe"]]["depends_on"])
        self.assertEqual(steps[indices["unqualified_new_run_conversation_probe"]]
                         ["evidence_class"], "NO_LIVE_PROBE_OR_RELEASE_QUALIFICATION")

    def test_new_run_requires_distinct_recovery_and_conversion_handoffs(self):
        """R16 recovery and old adapter conversion cannot stand in for new run."""
        value = integration_preflight.report()
        steps = {step["name"]: step for step in value["steps"]}
        self.assertIn("independent_new_checkpoint_restore", steps)
        self.assertIn("new_run_adapter_gguf_conversion", steps)
        self.assertIn("reloaded_optimizer_checkpoint",
                      steps["independent_new_checkpoint_restore"]["depends_on"])
        self.assertIn("independent_new_checkpoint_restore",
                      steps["independent_protected_evaluation"]["depends_on"])
        self.assertIn("independent_new_checkpoint_restore",
                      steps["new_run_adapter_gguf_conversion"]["depends_on"])
        self.assertIn("new_run_adapter_gguf_conversion",
                      steps["unqualified_new_run_conversation_probe"]["depends_on"])
        self.assertNotIn("recovered_R16_adapter_provenance",
                         steps["new_run_adapter_gguf_conversion"]["depends_on"])
        self.assertEqual(steps["new_run_adapter_gguf_conversion"]["status"],
                         "UNKNOWN")
        self.assertFalse(value["ready"])

    def test_metadata_enrollment_does_not_claim_weight_or_host_admission(self):
        required = integration_preflight.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS
        with patch.object(integration_preflight, "TRUSTED_OFFICIAL_METADATA_PIN_ITEMS",
                          tuple((name, "a" * 64) for name in required)):
            value = integration_preflight.report()
        self.assertTrue(value["source_metadata_pins_enrolled"])
        self.assertEqual(value["steps"][1]["status"], "UNKNOWN")
        self.assertEqual(value["steps"][5]["status"], "HOLD")
        self.assertFalse(value["ready"])

    def test_malformed_metadata_digest_enrollment_stays_hold(self):
        required = integration_preflight.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS
        valid = tuple((name, "a" * 64) for name in required)
        malformed = (
            tuple((name, "not-a-sha") for name in required),
            tuple((name, "A" * 64) for name in required),
            tuple((name, "a" * 63 + "g") for name in required),
            tuple((name, 123) for name in required),
            valid[:-1] + (valid[0],),
            valid[:-1] + (("../tokenizer.json", "a" * 64),),
            valid[:-1] + ((required[-1], "a" * 64, "extra"),),
            list(valid),
        )
        for injected in malformed:
            with self.subTest(injected=injected):
                with patch.object(integration_preflight,
                                  "TRUSTED_OFFICIAL_METADATA_PIN_ITEMS", injected):
                    value = integration_preflight.report()
                self.assertFalse(value["source_metadata_pins_enrolled"])
                self.assertEqual(value["steps"][1]["status"], "HOLD")
                self.assertFalse(value["ready"])
                self.assertFalse(value["physical_launch_attempted"])

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
