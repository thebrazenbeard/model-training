"""Freeze public documentation's R14/R16 producer and launch-status distinction."""

from pathlib import Path
import unittest


README = Path(__file__).resolve().parents[1] / "trainingday" / "README.md"


class TrainingDayReadmeContractTests(unittest.TestCase):
    def test_r14_legacy_and_r16_current_pins_are_distinct(self):
        source = README.read_text(encoding="utf-8")
        self.assertIn("Legacy R14/v1", source)
        self.assertIn("188d3340e4498e69d8c54d66ebee9dabd6582d31", source)
        self.assertIn("Current R16/v2", source)
        self.assertIn("1d38e4f25b5663bf8a84ff5e139875d4152e3680", source)
        self.assertIn("90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836", source)
        self.assertNotIn("--producer-git-head 188d3340e4498e69d8c54d66ebee9dabd6582d31", source)

    def test_training_held_and_gguf_serving_remains_future(self):
        source = README.read_text(encoding="utf-8")
        self.assertIn("HARD HOLD", source)
        self.assertIn("trainingday/physical_launch_authority.py", source)
        self.assertIn("GGUF", source)
        self.assertIn("KoboldCPP", source)
        self.assertIn("v-e-r-a", source)
        self.assertIn("not runtime-qualified", source)


if __name__ == "__main__":
    unittest.main()
