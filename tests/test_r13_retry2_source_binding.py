"""R13 retry2 preserves the failed first attempt and binds exact FP32 restoration."""
import hashlib
import json
from pathlib import Path
import unittest

from successor.experiments import preflight_r13_behavior_retry2_launch as retry

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"successor/experiments"
SPEC=HERE/"V10R3R13_FROZEN_BEHAVIOR_SPEC_20261016_RETRY2.json"


class R13Retry2Tests(unittest.TestCase):
    def test_retry2_frozen_spec_and_distinct_output(self):
        sha=hashlib.sha256(SPEC.read_bytes()).hexdigest()
        self.assertEqual(sha,"0127d7e88f3ed608117d2a1388d0c82a3d921df75fbf408374db3945b1060418")
        self.assertNotIn(b"\r\n",SPEC.read_bytes())
        x=json.loads(SPEC.read_text(encoding="utf8"))
        self.assertEqual(x["output"]["namespace"],str(retry.NEW_ADAPTER))
        self.assertTrue(str(retry.NEW_ADAPTER).endswith("-retry2"))
        self.assertEqual(x["source_subject"]["train_sha256"],
            "90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5")
        self.assertEqual(x["resume_adapter"]["adapter_model_sha256"],
            "a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c")
        self.assertEqual(x["trainer"]["max_optimizer_steps"],32)

    def test_controllers_have_distinct_safety_receipts(self):
        old=(HERE/"run_r13_with_preactive_restore_20261016.ps1").read_text(encoding="utf8")
        current=(HERE/"run_r13_exact_fp32_retry2_20261016.ps1").read_text(encoding="utf8")
        self.assertIn("VERA_R13_TRAIN_20261016_CONTROLLER_V3.json",old)
        self.assertIn("VERA_R13_TRAIN_20261016_CONTROLLER_V4.json",current)
        self.assertIn("BT2 Vera R13 Retry2 Recovery 20261016",current)
        self.assertIn("Register-ScheduledTask -TaskName $taskName",current)
        self.assertIn("RestorePreActive",current)
        self.assertIn("if(-not $PreflightOnly){",current)
        self.assertIn("preflight_r13_behavior_retry2_launch",current)
        self.assertNotIn("PREACTIVE_FORCE_KILL",current)

    def test_retry2_recovery_only_matches_retry2_spec(self):
        recover=(HERE/"restore_preactive_r13_retry2_20261016.ps1").read_text(encoding="utf8")
        self.assertIn("V10R3R13_FROZEN_BEHAVIOR_SPEC_20261016_RETRY2.json",recover)
        self.assertIn("Contains($spec)",recover)
        self.assertIn("Contains($repo)",recover)
        self.assertIn("OUTSIDE_PREACTIVE_AUTHORIZED_WINDOW_NO_START",recover)


if __name__=="__main__":
    unittest.main()
