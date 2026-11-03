"""Source-only candidate tests; no Torch, PEFT, weights, or GPU."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trainingday.qwen35_hybrid_lora_candidate import (
    EXPECTED_TARGETS, HYBRID_LINEAR_PROJECTIONS, HybridLoRACandidateError,
    audit_pinned_hybrid_candidate, propose_and_check,
)
from trainingday.qwen35_memory_source_admission import MetadataAdmissionError


def synthetic_shape():
    # Reuse the independently tested structural shape fixture, then populate
    # its intentional subset to include the known additional nn.Linear paths.
    from tests.test_trainingday_qwen35_memory_source_admission import ArchitectureAdmissionTests
    shape = ArchitectureAdmissionTests()
    shape.setUp()
    config = shape.config
    index = shape.index
    for layer, kind in enumerate(config["text_config"]["layer_types"]):
        for projection in HYBRID_LINEAR_PROJECTIONS[kind]:
            index["weight_map"][
                f"model.language_model.layers.{layer}.{projection}.weight"
            ] = shape.shards[layer % 2]
    index["weight_map"]["mtp.layers.0.self_attn.v_proj.weight"] = shape.shards[1]
    return config, index


class HybridLoRACandidateTests(unittest.TestCase):
    def setUp(self):
        self.config, self.index = synthetic_shape()

    def assess(self, targets=None):
        return propose_and_check(self.config, self.index, candidate_targets=targets)

    def test_all_known_attention_branch_linear_projections_covered(self):
        out = self.assess()
        targets = out["proposed_lora_config"]["target_modules"]
        self.assertEqual(EXPECTED_TARGETS, 152)
        self.assertEqual(len(targets), 152)
        self.assertEqual(len(set(targets)), 152)
        self.assertEqual(out["projected_matched_module_count"], 152)
        self.assertEqual(out["expected_hybrid_projection_module_count"], 152)
        self.assertEqual(out["full_attention_targeted_layers"], [3, 7, 11, 15, 19, 23, 27, 31])
        self.assertEqual(len(out["gdn_targeted_layers"]), 24)
        self.assertTrue(out["full_attention_linear_projections_complete"])
        self.assertTrue(out["gdn_linear_projections_complete"])
        self.assertTrue(out["projected_text_coverage_complete"])
        self.assertFalse(out["pinned_official_metadata_bytes_verified"])
        self.assertFalse(out["physical_launch_authorized"])

    def test_every_target_is_explicitly_qualified_for_projected_text_loader(self):
        out = self.assess()
        for target in out["proposed_lora_config"]["target_modules"]:
            self.assertRegex(target, r"^model\.layers\.(?:[0-9]|[12][0-9]|3[01])\.")
            self.assertFalse(target.startswith("model.language_model."))
        self.assertEqual(out["potential_mtp_vision_target_collisions"], [])
        self.assertEqual(out["baseline_qv_would_match_auxiliary_modules_if_loaded"],
                         ["mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.v_proj"])

    def test_candidate_is_not_installed_or_memory_fit(self):
        out = self.assess()
        self.assertEqual(out["status"],
                         "SOURCE_ONLY_ALL_32_ATTENTION_BRANCHES_TARGETED_UNQUALIFIED")
        self.assertIn("TARGETS_ARE_NOT_INSTALLED_IN_PROTECTED_LOWMEM_KERNEL",
                      out["source_only_limits"])
        self.assertIn("GDN_NF4_BNB_PEFT_COMPATIBILITY_UNPROVEN", out["source_only_limits"])
        self.assertEqual(out["hardware_fit"], "UNKNOWN")
        self.assertFalse(out["training_authorized"])
        self.assertEqual(out["optimizer_steps_observed"], 0)

    def test_partial_explicit_candidate_has_partial_status(self):
        targets = ["model.layers.0.linear_attn.in_proj_qkv"]
        out = self.assess(targets)
        self.assertEqual(out["projected_matched_module_count"], 1)
        self.assertEqual(out["gdn_targeted_layers"], [0])
        self.assertEqual(out["full_attention_targeted_layers"], [])
        self.assertFalse(out["projected_text_coverage_complete"])
        self.assertEqual(out["status"], "SOURCE_ONLY_PARTIAL_HYBRID_COVERAGE_HOLD")

    def test_broad_suffixes_foreign_or_duplicate_targets_rejected(self):
        bad_targets = (["q_proj", "v_proj"], ["linear_attn.in_proj_qkv"],
                       ["model.language_model.layers.0.linear_attn.in_proj_qkv"],
                       ["model.layers.0.linear_attn.in_proj_qkv"] * 2,
                       ["model.layers.31.linear_attn.in_proj_qkv"],
                       ["mtp.layers.0.self_attn.q_proj"], [], [".*"])
        for targets in bad_targets:
            with self.subTest(targets=targets), self.assertRaises(HybridLoRACandidateError):
                self.assess(targets)

    def test_accidental_mtp_suffix_collision_rejected_even_if_decoders_match(self):
        self.index["weight_map"][
            "mtp.model.layers.0.linear_attn.in_proj_qkv.weight"
        ] = "model.safetensors-00002-of-00002.safetensors"
        with self.assertRaisesRegex(HybridLoRACandidateError, "accidentally match"):
            self.assess()

    def test_missing_gdn_b_or_a_or_full_k_or_o_is_rejected(self):
        for tensor in (
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.3.self_attn.k_proj.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
        ):
            self.config, self.index = synthetic_shape()
            self.index["weight_map"].pop(tensor)
            with self.subTest(tensor=tensor), self.assertRaisesRegex(
                    HybridLoRACandidateError, "missing"):
                self.assess()

    def test_foreign_or_malformed_checkpoint_index_rejected(self):
        self.index["weight_map"]["foreign.decoder.weight"] = "dummy"
        with self.assertRaisesRegex(HybridLoRACandidateError, "non-text"):
            self.assess()
        self.config, self.index = synthetic_shape()
        self.index["weight_map"]["model.language_model.layers.0.linear_attn.out_proj.weight"] = 1
        with self.assertRaisesRegex(HybridLoRACandidateError, "index"):
            self.assess()

    def test_invalid_architecture_or_text_mapping_contract_rejected(self):
        self.config["text_config"]["layer_types"][3] = "linear_attention"
        with self.assertRaises(HybridLoRACandidateError):
            self.assess()
        self.config, self.index = synthetic_shape()
        with patch("trainingday.qwen35_hybrid_lora_candidate.TEXT_PREFIX", "wrong."):
            with self.assertRaisesRegex(HybridLoRACandidateError, "mapping"):
                self.assess()

    def test_synthetic_self_hashed_bytes_cannot_enter_pinned_gate(self):
        with self.assertRaises(MetadataAdmissionError):
            audit_pinned_hybrid_candidate(json.dumps(self.config).encode(),
                                          json.dumps(self.index).encode())

    def test_cli_missing_and_synthetic_inputs_hold(self):
        entry = [sys.executable, "-m", "trainingday.qwen35_hybrid_lora_candidate"]
        missing = subprocess.run(entry, capture_output=True, text=True, check=False)
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(json.loads(missing.stdout)["status"],
                         "OFFICIAL_METADATA_MISSING_HARD_HOLD")
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "config.json"
            ind = Path(temp) / "index.json"
            cfg.write_text(json.dumps(self.config), encoding="utf-8")
            ind.write_text(json.dumps(self.index), encoding="utf-8")
            got = subprocess.run(entry + ["--config", str(cfg), "--index", str(ind)],
                                 capture_output=True, text=True, check=False)
            self.assertEqual(got.returncode, 1)
            self.assertEqual(json.loads(got.stdout)["status"], "SOURCE_REJECTED_HARD_HOLD")

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"),
                         "POSIX FIFO behavior only")
    def test_cli_fifo_holds_without_blocking(self):
        entry = [sys.executable, "-m", "trainingday.qwen35_hybrid_lora_candidate"]
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "config.fifo"
            ind = Path(temp) / "index.json"
            os.mkfifo(cfg)
            ind.write_text(json.dumps(self.index), encoding="utf-8")
            got = subprocess.run(entry + ["--config", str(cfg), "--index", str(ind)],
                                 capture_output=True, text=True, check=False, timeout=2)
            self.assertEqual(got.returncode, 1)
            self.assertFalse(json.loads(got.stdout)["training_authorized"])

    def test_import_never_imports_torch_or_peft(self):
        statement = ("import sys; import trainingday.qwen35_hybrid_lora_candidate; "
                     "assert 'torch' not in sys.modules and 'peft' not in sys.modules")
        got = subprocess.run([sys.executable, "-c", statement],
                             capture_output=True, text=True, check=False)
        self.assertEqual(got.returncode, 0, got.stderr)


if __name__ == "__main__":
    unittest.main()
