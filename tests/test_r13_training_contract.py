"""R13 deliberate, strictly bounded continuation without widening legacy recipes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from successor.experiments.train_v10r2_dev import (
    DevTrainingHold, load_and_validate_dev_spec
)

REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "successor/experiments/V10R3R12_FP32_VERA_VOICE_EXECUTION_SPEC_20261015_TRL_RETRY2.json"


def check_spec(spec):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "candidate.json"
        p.write_text(json.dumps(spec), encoding="utf8")
        return load_and_validate_dev_spec(p)


class R13TrainingGateTests(unittest.TestCase):
    def make(self):
        spec = json.loads(PARENT.read_text(encoding="utf8"))
        spec["experiment_class"] = "R13_FROZEN_DIVERSE_IDENTITY_AND_JUDGMENT_32_STEPS"
        spec["trainer"]["max_optimizer_steps"] = 32
        spec["trainer"]["learning_rate"] = 1e-5
        spec["development_window"] = {"start_row": 416, "row_count": 256}
        spec["resume_adapter"]["next_train_row"] = 416
        return spec

    def test_r13_32_steps_allowed_only_for_exact_window_and_precision(self):
        spec = self.make()
        self.assertEqual(check_spec(spec)["trainer"]["max_optimizer_steps"], 32)
        spec["development_window"]["row_count"] = 248
        with self.assertRaisesRegex(DevTrainingHold, "row count"):
            check_spec(spec)

    def test_legacy_class_cannot_silently_train_32_steps(self):
        spec = self.make()
        spec["experiment_class"] = "TARGETED_VERA_VOICE_AND_INDEPENDENT_JUDGMENT"
        with self.assertRaisesRegex(DevTrainingHold, "1 through 8"):
            check_spec(spec)

    def test_r13_refuses_precision_learning_rate_and_optimizer_drift(self):
        base = self.make()
        for mutate in [
            lambda x:x.update({"adapter_precision_policy":"LEGACY_BF16"}),
            lambda x:x["trainer"].update({"learning_rate":2e-5}),
            lambda x:x["trainer"].update({"max_optimizer_steps":24}),
            lambda x:x["trainer"].update({"optimizer":"adamw_torch_8bit"}),
        ]:
            x = copy.deepcopy(base); mutate(x)
            with self.subTest(changed=x):
                with self.assertRaises(DevTrainingHold):
                    check_spec(x)


if __name__=="__main__":
    unittest.main()
