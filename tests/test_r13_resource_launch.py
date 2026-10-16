"""R13 launch must stop before optimizer if GPU workload is resident."""
import unittest
from pathlib import Path
from successor.experiments import preflight_r13_behavior_launch as gate


class R13ResourceTests(unittest.TestCase):
    def test_existing_serving_gpu_load_forces_hold(self):
        result=gate.classify(ram=12.4,virtual=30,gpu=770,active=0,output_exists=False)
        self.assertEqual(result["status"],"HOLD_NO_OPTIMIZER")
        self.assertIn("GPU_OCCUPIED_PRESERVE_SERVING_PROCESS",result["reasons"])
        self.assertFalse(result["services_interrupted"])
        self.assertEqual(result["optimizer_steps_executed"],0)

    def test_collision_and_other_training_are_protected(self):
        held=gate.classify(ram=12,virtual=20,gpu=3980,active=1,output_exists=True)
        self.assertIn("ANOTHER_TRAINER_ACTIVE",held["reasons"])
        self.assertIn("OUTPUT_EXISTS_PRESERVE_ADAPTER",held["reasons"])

    def test_exact_receipt_path_only(self):
        accepted=gate.RECEIPT.with_name("VERA_R13_BEHAVIOR_LAUNCH_GATE_20261016_V999999.json")
        self.assertEqual(gate.validated_receipt(accepted),accepted.resolve())
        with self.assertRaisesRegex(RuntimeError,"Unsafe"):
            gate.validated_receipt(Path("C:/TEMP/VERA_R13_BEHAVIOR_LAUNCH_GATE_20261016_V2.json"))

    def test_source_frozen_sha_and_recipe(self):
        from successor.experiments.train_v10_qwen35_authorized import sha256_file
        from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec
        self.assertEqual(sha256_file(gate.SPEC_PATH),
            "ad2a3d56c9a5ee648764e9469676b78155bb993de690edda9627189a767b0fdf")
        spec=load_and_validate_dev_spec(gate.SPEC_PATH)
        self.assertEqual(spec["experiment_class"],gate.CLASS)
        self.assertEqual(spec["trainer"]["max_optimizer_steps"],32)
        self.assertNotIn(b"\r\n",gate.SPEC_PATH.read_bytes())


if __name__=="__main__":
    unittest.main()
