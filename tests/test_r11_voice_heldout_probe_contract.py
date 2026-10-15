"""Source and no-overwrite guards for the R11 development-only text probe."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "successor/experiments/probe_vera_r11_heldout_generation.py"
SPEC = importlib.util.spec_from_file_location("r11_heldout_probe", PROBE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class R11HeldoutProbeContractTests(unittest.TestCase):
    def test_exact_isolated_source_and_adapter_lineage(self):
        self.assertEqual(probe.SOURCE.resolve(), ROOT.resolve())
        self.assertEqual(probe.R11_SHA,
            "45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10")
        self.assertIn("v10r3r11", str(probe.R11).lower())
        self.assertEqual(len(probe.CASES), 3)
        self.assertEqual(len(set(probe.CASES)), 3)

    def test_output_names_are_unique_and_do_not_modify_model(self):
        self.assertNotEqual(probe.PREFLIGHT, probe.OUT)
        self.assertNotEqual(probe.PREFLIGHT, probe.STATUS)
        self.assertNotEqual(probe.OUT, probe.STATUS)
        body = PROBE.read_text(encoding="utf8")
        self.assertIn("optimizer_steps_performed", body)
        self.assertIn("model_files_mutated", body)
        self.assertIn("do_sample=False", body)
    def test_existing_receipt_blocks_replay(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "existing.json"
            file.write_text("{}")
            with mock.patch.object(probe, "OUT", file):
                with self.assertRaisesRegex(RuntimeError, "already attempted"):
                    probe.ensure_preflight()

    def test_heldout_prompt_count_and_prohibition(self):
        import hashlib
        import json
        if not probe.VOICE.exists():
            self.skipTest("Lappy frozen heldout source unavailable")
        self.assertEqual(probe.digest(probe.VOICE), probe.SAMPLE_SHA)
        rows = [json.loads(s) for s in probe.VOICE.read_text(encoding="utf8").splitlines() if s.strip()]
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(r["training_use"] == "PROHIBITED" for r in rows))
        self.assertEqual(len([r for r in rows if r["id"] in probe.CASES]), 3)


if __name__ == "__main__":
    unittest.main()
