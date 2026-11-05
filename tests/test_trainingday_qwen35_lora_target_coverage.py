"""Offline static source/index analysis: synthetic bytes, no model/GPU calls."""

import copy
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from trainingday.lowmem_experiment import one_step_kernel
from trainingday.qwen35_lora_target_coverage import (
    StaticCoverageError, assess_projection_inventory, audit_pinned_lowmem_source,
    extract_kernel_source_contract, inspect_lowmem_source,
)
from trainingday.qwen35_memory_source_admission import MetadataAdmissionError


def fixture():
    from tests.test_trainingday_qwen35_memory_source_admission import ArchitectureAdmissionTests
    shape = ArchitectureAdmissionTests()
    shape.setUp()
    return shape.config, shape.index


class StaticLoRACoverageTests(unittest.TestCase):
    def setUp(self):
        self.config, self.index = fixture()
        self.contract = inspect_lowmem_source()

    def assess(self, targets=None):
        contract = copy.deepcopy(self.contract)
        if targets is not None:
            contract["target_modules"] = targets
        return assess_projection_inventory(self.config, self.index, contract)

    def test_actual_kernel_and_advertised_source_plan_match(self):
        self.assertEqual(self.contract["target_modules"], ["q_proj", "v_proj"])
        self.assertEqual(self.contract["lora_rank"], 4)
        self.assertTrue(self.contract["load_in_4bit"])
        self.assertEqual(self.contract["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(self.contract["bnb_4bit_use_double_quant"])
        self.assertEqual(self.contract["bnb_4bit_compute_dtype"], "torch.bfloat16")
        self.assertTrue(self.contract["gradient_checkpointing_requested"])
        self.assertTrue(self.contract["physical_authority_guard_precedes_model_path"])

    def test_qv_only_is_eight_of_thirty_two_and_zero_gdn(self):
        out = self.assess()
        self.assertEqual(out["status"], "SOURCE_ONLY_HYBRID_DECODER_COVERAGE_INADEQUATE")
        self.assertEqual(out["targeted_decoder_layers"], 8)
        self.assertEqual(out["targeted_decoder_fraction"], 0.25)
        self.assertEqual(out["gdn_targeted_layers"], [])
        self.assertEqual(out["full_attention_targeted_layers"], [3, 7, 11, 15, 19, 23, 27, 31])
        self.assertEqual(out["matched_decoder_projection_modules"], 16)
        self.assertEqual(out["reference_projection_complete_decoder_layers"], 8)
        self.assertFalse(out["pinned_official_metadata_bytes_verified"])
        self.assertFalse(out["checked_in_kernel_source_inspected"])
        self.assertEqual(out["contract_origin"], "CALLER_SUPPLIED_SHAPE_TEST_ONLY")
        self.assertFalse(out["physical_launch_authorized"])
        self.assertEqual(out["optimizer_steps_observed"], 0)

    def test_qv_suffix_potentially_matches_auxiliary_mtp_not_decoder(self):
        out = self.assess()
        self.assertEqual(out["potential_non_decoder_suffix_matches"],
                         ["mtp.layers.0.self_attn.q_proj"])
        self.assertFalse(out["training_authorized"])

    def test_reference_expansion_covers_all_hybrid_layers_only_as_static_hypothesis(self):
        out = self.assess(["q_proj", "v_proj", "linear_attn.in_proj_qkv",
                           "linear_attn.in_proj_z", "linear_attn.out_proj"])
        self.assertEqual(out["targeted_decoder_layers"], 32)
        self.assertEqual(len(out["gdn_targeted_layers"]), 24)
        self.assertEqual(out["reference_projection_complete_decoder_layers"], 32)
        self.assertEqual(out["matched_decoder_projection_modules"], 88)
        self.assertEqual(out["status"],
                         "SOURCE_ONLY_REFERENCE_PROJECTIONS_COMPLETE_HARDWARE_UNKNOWN")
        self.assertFalse(out["official_origin_authenticated"])
        self.assertFalse(out["training_authorized"])

    def test_partial_full_attention_q_only_is_not_reference_complete(self):
        out = self.assess(["q_proj"])
        self.assertEqual(out["targeted_decoder_layers"], 8)
        self.assertEqual(out["reference_projection_complete_decoder_layers"], 0)
        self.assertEqual(out["matched_decoder_projection_modules"], 8)

    def test_unmatched_unknown_and_overbroad_targets_rejected(self):
        for targets in (["does_not_exist"], ["all-linear"], [".*"],
                        ["q_proj", "q_proj"], [], ["self_attn.q_proj", "does_not_exist"]):
            with self.subTest(targets=targets), self.assertRaises(StaticCoverageError):
                self.assess(targets)

    def test_wrong_decoder_type_or_foreign_decoder_match_rejected(self):
        self.config["text_config"]["layer_types"][0] = "full_attention"
        with self.assertRaises(StaticCoverageError):
            self.assess()
        self.config, self.index = fixture()
        self.index["weight_map"]["model.language_model.layers.0.self_attn.q_proj.weight"] = "s"
        with self.assertRaisesRegex(StaticCoverageError, "unexpected"):
            self.assess()

    def test_malformed_inventory_types_are_rejected_without_crashing(self):
        self.config["text_config"] = []
        with self.assertRaisesRegex(StaticCoverageError, "text decoder"):
            self.assess()
        self.config, self.index = fixture()
        self.index["weight_map"]["bad.value"] = 1
        with self.assertRaisesRegex(StaticCoverageError, "weight-map"):
            self.assess()

    def test_changed_nf4_double_quant_or_bf16_source_fails_closed(self):
        source = inspect.getsource(one_step_kernel)
        for old, new in (("bnb_4bit_quant_type=\"nf4\"", "bnb_4bit_quant_type=\"fp4\""),
                         ("bnb_4bit_use_double_quant=True", "bnb_4bit_use_double_quant=False"),
                         ("bnb_4bit_compute_dtype=torch.bfloat16",
                          "bnb_4bit_compute_dtype=torch.float16"),
                         ("device_map=0", "device_map='auto'"),
                         ("target_modules=[\"q_proj\", \"v_proj\"]",
                          "target_modules='all-linear'"),
                         ('require_physical_launch_authority(operation="low-memory synthetic fit experiment")',
                          'print("authority guard removed")')):
            self.assertIn(old, source)
            with self.subTest(old=old), self.assertRaises(StaticCoverageError):
                extract_kernel_source_contract(source.replace(old, new, 1))

    def test_quantization_limits_are_expressed_without_fit_claim(self):
        out = self.assess()
        self.assertIn("FOUR_GB_GPU_MEMORY_FIT_UNVERIFIED", out["quantization_static_limits"])
        self.assertIn("SIXTY_GB_PAGEFILE_DOES_NOT_INCREASE_VRAM",
                      out["quantization_static_limits"])
        self.assertIn("PEFT_RUNTIME_MODULE_INJECTION_UNVERIFIED", out["quantization_static_limits"])
        self.assertEqual(out["hardware_fit"], "UNKNOWN")

    def test_synthetic_metadata_cannot_pass_official_pinned_gate(self):
        fake_config = json.dumps(self.config).encode("utf-8")
        fake_index = json.dumps(self.index).encode("utf-8")
        with self.assertRaises(MetadataAdmissionError):
            audit_pinned_lowmem_source(fake_config, fake_index)

    def test_cli_missing_and_synthetic_metadata_hard_hold(self):
        base = [sys.executable, "-m", "trainingday.qwen35_lora_target_coverage"]
        missing = subprocess.run(base, capture_output=True, text=True, check=False)
        self.assertEqual(missing.returncode, 2)
        self.assertFalse(json.loads(missing.stdout)["training_authorized"])
        with tempfile.TemporaryDirectory() as temp:
            conf = Path(temp) / "config.json"
            index = Path(temp) / "index.json"
            conf.write_text(json.dumps(self.config), encoding="utf-8")
            index.write_text(json.dumps(self.index), encoding="utf-8")
            result = subprocess.run(base + ["--config", str(conf), "--index", str(index)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stdout)["status"], "SOURCE_REJECTED_HARD_HOLD")

    def test_module_import_has_no_torch_or_peft_side_effects(self):
        script = ("import sys; import trainingday.qwen35_lora_target_coverage; "
                  "assert 'torch' not in sys.modules and 'peft' not in sys.modules")
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, check=False)
        self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()
