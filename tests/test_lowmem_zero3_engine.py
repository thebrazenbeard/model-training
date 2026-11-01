"""CPU-only source-contract checks. This suite never loads Qwen or CUDA."""

import json
import subprocess
import sys
import unittest

from trainingday.physical_launch_authority import PhysicalLaunchAuthorityUnavailable


class Zero3SourceContractTests(unittest.TestCase):
    def test_checkpoint_gradients_and_offload_are_not_falsely_qualified(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "trainingday"
                  / "lowmem_zero3_engine.py").read_text(encoding="utf-8")
        self.assertIn('"use_reentrant": False', source)
        self.assertIn("model.enable_input_require_grads()", source)
        self.assertIn('"cpu_offload_runtime_verified": False', source)
        self.assertNotIn('"actual_cpu_offload_configured": True', source)

    def test_import_does_not_import_training_stack(self):
        program = (
            "import sys; import trainingday.lowmem_zero3_engine; "
            "assert not {'torch', 'deepspeed', 'transformers', 'peft'} "
            ".intersection(sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", program],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_plan_uses_genuine_training_offload_and_reports_hold(self):
        from trainingday.lowmem_zero3_engine import source_plan, zero3_config

        plan = source_plan()
        self.assertFalse(plan["authorizes_training"])
        self.assertEqual(plan["optimizer_steps_observed"], 0)
        self.assertEqual(plan["data_access"], "SYNTHETIC_ONLY")
        self.assertEqual(plan["steps_requested"], 3)
        config = zero3_config()
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertEqual(config["zero_optimization"]["offload_param"]["device"], "cpu")
        self.assertEqual(config["zero_optimization"]["offload_optimizer"]["device"], "cpu")
        self.assertTrue(config["bf16"]["enabled"])
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 1)
        self.assertNotIn("device_map", config)

    def test_physical_entrypoint_holds_before_model_or_import(self):
        from trainingday.lowmem_zero3_engine import run_qualified_zero3_fit
        with self.assertRaisesRegex(PhysicalLaunchAuthorityUnavailable, "HARD HOLD"):
            run_qualified_zero3_fit("/never/read/private/model-binding.json")

    def test_synthetic_telemetry_requires_continuous_real_changes(self):
        from trainingday.lowmem_zero3_engine import (Zero3SourceError,
                                                     summarize_step_evidence)
        a, b, c, d = (str(i) * 64 for i in range(4))
        records = [
            {"before_sha256": a, "after_sha256": b, "loss": 3.0,
             "gradient_norm": 0.1, "cuda_peak_reserved_bytes": 1_000,
             "cuda_peak_allocated_bytes": 700, "rss_peak_bytes": 2_000,
             "elapsed_seconds": 1.0},
            {"before_sha256": b, "after_sha256": c, "loss": 2.8,
             "gradient_norm": 0.2, "cuda_peak_reserved_bytes": 1_050,
             "cuda_peak_allocated_bytes": 750, "rss_peak_bytes": 2_100,
             "elapsed_seconds": 1.1},
            {"before_sha256": c, "after_sha256": d, "loss": 2.7,
             "gradient_norm": 0.2, "cuda_peak_reserved_bytes": 1_000,
             "cuda_peak_allocated_bytes": 750, "rss_peak_bytes": 2_100,
             "elapsed_seconds": 1.0},
        ]
        summary = summarize_step_evidence(records, total_cuda_bytes=4_096,
                                          physical_attestation=None)
        self.assertEqual(summary["step_count"], 3)
        self.assertFalse(summary["physical_training_attested"])
        self.assertFalse(summary["proves_memory_plateau"])
        self.assertEqual(summary["adapter_final_sha256"], d)
        for broken in (
            records[:2],
            [records[0], {**records[1], "before_sha256": a}, records[2]],
            [records[0], {**records[1], "after_sha256": b}, records[2]],
            [records[0], {**records[1], "gradient_norm": 0.0}, records[2]],
            [records[0], {**records[1], "cuda_peak_reserved_bytes": 5_000}, records[2]],
        ):
            with self.subTest(broken=broken), self.assertRaises(Zero3SourceError):
                summarize_step_evidence(broken, total_cuda_bytes=4_096,
                                        physical_attestation=None)

    def test_cli_rejects_executable_flags(self):
        bad = subprocess.run([sys.executable, "-m", "trainingday.lowmem_zero3_engine",
                              "--execute"], text=True, capture_output=True, check=False)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")
        self.assertIn("HARD HOLD", bad.stderr)
        good = subprocess.run([sys.executable, "-m", "trainingday.lowmem_zero3_engine"],
                              text=True, capture_output=True, check=False)
        self.assertEqual(good.returncode, 0)
        self.assertFalse(json.loads(good.stdout)["authorizes_training"])


if __name__ == "__main__":
    unittest.main()
