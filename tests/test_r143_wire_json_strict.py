"""CPU-only hostile JSON-wire checks for R18 source evidence entrypoints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday import r18_benchmark as bench, r18_qlora as q


class EvidenceWireJsonTests(unittest.TestCase):
    def test_benchmark_loader_rejects_escaped_and_nested_duplicate_members(self):
        with tempfile.TemporaryDirectory(prefix="r143-wire-bench-") as directory:
            path = Path(directory) / "receipt.json"
            for raw in (
                b'{"eval_access":true,"\\u0065val_access":false}',
                b'{"receipt":{"status":"TRAINED_UNSCORED","STATUS":"TRAINED_UNSCORED"}}',
                b'{"arms":[{"control":"correct","control":"corrupt"}]}',
            ):
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaises(bench.BenchmarkError):
                        bench._json_file(path)
            path.write_text(json.dumps({"schema": "wire-control", "eval_access": False}))
            self.assertEqual(bench._json_file(path)["eval_access"], False)

    def test_benchmark_loader_rejects_mixed_script_control_key_without_banning_unicode_values(self):
        with tempfile.TemporaryDirectory(prefix="r146-wire-confusable-") as directory:
            path = Path(directory) / "receipt.json"
            # First letter in the second key is Cyrillic ie, visually Latin e.
            path.write_text('{"eval_access":false,"еval_access":true}', encoding="utf-8")
            with self.assertRaises(bench.BenchmarkError):
                bench._json_file(path)

            path.write_text(json.dumps({"eval_access": False,
                                        "description": "Café — привет — 世界",
                                        "café": "allowed Latin-script key"}),
                            encoding="utf-8")
            result = bench._json_file(path)
            self.assertIs(result["eval_access"], False)
            self.assertEqual(result["description"], "Café — привет — 世界")
            self.assertEqual(result["café"], "allowed Latin-script key")

    def test_benchmark_loader_rejects_default_ignorable_control_key_aliases(self):
        with tempfile.TemporaryDirectory(prefix="r146-wire-ignorable-") as directory:
            path = Path(directory) / "receipt.json"
            for hidden in ("\u200d", "\u200b", "\u034f", "\u2060", "\u200c",
                           "\ufe0f", "\u202e", "\u00ad", "\U000e0061"):
                with self.subTest(hidden=ascii(hidden)):
                    ambiguous = "eval" + hidden + "_access"
                    path.write_text(json.dumps({"eval_access": False, ambiguous: True}),
                                    encoding="utf-8")
                    with self.assertRaises(bench.BenchmarkError):
                        bench._json_file(path)

            path.write_text(json.dumps({"eval_access": False,
                                        "description": "Café — привет — 世界 👩‍💻",
                                        "café": "Latin-accented key preserved"}),
                            encoding="utf-8")
            result = bench._json_file(path)
            self.assertIs(result["eval_access"], False)
            self.assertEqual(result["description"], "Café — привет — 世界 👩‍💻")
            self.assertEqual(result["café"], "Latin-accented key preserved")

    def test_benchmark_loader_rejects_ascii_control_character_keys(self):
        with tempfile.TemporaryDirectory(prefix="r146-wire-control-char-") as directory:
            path = Path(directory) / "receipt.json"
            for hidden in ("\x00", "\x1f", "\x7f", "\x85"):
                with self.subTest(hidden=ascii(hidden)):
                    path.write_text(json.dumps({"eval_access": False,
                                                "ev" + hidden + "al_access": True}),
                                    encoding="utf-8")
                    with self.assertRaisesRegex(bench.BenchmarkError,
                                                "ambiguous or malformed JSON evidence"):
                        bench._json_file(path)

    def test_qlora_preflight_rejects_escaped_duplicate_plan_keys(self):
        # Abort before touching source data/host by injecting valid upstream custody.
        with tempfile.TemporaryDirectory(prefix="r143-wire-plan-") as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            plan = root / "plan.json"
            plan.write_bytes(b'{"schema":"' + q.PLAN_SCHEMA.encode() +
                             b'","seed":41,"\\u0073eed":41}')
            args = SimpleNamespace(control="correct", run_id="safe-run",
                                   output_root=output, manifest=root / "manifest.json",
                                   producer_git_head=q.R16_PRODUCER_HEAD,
                                   cases=8, seed=41, learning_rate=0.00002,
                                   base_binding=root / "base.json", plan=plan)
            fake_base = {"model_id": q.BASE_ID, "revision": q.BASE_REVISION}
            with patch.object(q, "_hash_file", return_value=q.R16_V2_MANIFEST_SHA256), \
                 patch.object(q, "_initial_adapter_custody", return_value={}), \
                 patch.object(q, "_required_base", return_value=(fake_base, root)), \
                 self.assertRaisesRegex(AdmissionError, "duplicate|ambiguous"):
                q.preflight(args)

    def test_qlora_probe_rejects_escaped_duplicate_child_status(self):
        with tempfile.TemporaryDirectory(prefix="r143-wire-probe-") as directory:
            root = Path(directory)

            def fake_process(command, **kwargs):
                path = Path(command[command.index("--probe-result") + 1])
                path.write_bytes(b'{"status":"OOM","\\u0073tatus":"FIT"}')
                return SimpleNamespace(returncode=0)

            with patch.object(q.subprocess, "run", side_effect=fake_process), \
                 patch("trainingday.physical_launch_authority.require_physical_launch_authority"), \
                 self.assertRaisesRegex(AdmissionError, "duplicate|ambiguous"):
                q.probe_sweep(SimpleNamespace(base_binding=root / "base.json", seed=41,
                                              learning_rate=0.00002,
                                              initial_adapter=None), root,
                              {"probe_budget_seconds": 30})


if __name__ == "__main__":
    unittest.main()
