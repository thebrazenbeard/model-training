"""Negative source-only Contract-B plan tests, with no host/GPU effects."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday import tier1_public_fit_plan as fit


KERNEL_SHA = "a" * 64


class Tier1PublicFitPlanTests(unittest.TestCase):
    def test_official_base_only_and_no_caller_model_choice(self):
        plan = fit.build_public_fit_plan(executable_sha256=KERNEL_SHA)
        self.assertEqual(plan["model_id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(plan["model_revision"], "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(len(plan["official_shard_lfs_oid_sha256"]), 2)
        self.assertEqual(plan["metadata_sha256_status"], "UNKNOWN_UNAUTHENTICATED")
        self.assertIs(plan["model_snapshot_authenticated"], False)
        self.assertNotIn("rodrigomt", json.dumps(plan).lower())
        with self.assertRaises(TypeError):
            fit.build_public_fit_plan(executable_sha256=KERNEL_SHA,
                                      model_id="rodrigomt/Qwen3.5-4B-Uncensored-Aggressive")

    def test_digest_independently_recomputed_over_exact_canonical_ascii(self):
        receipt = fit.candidate_receipt(executable_sha256=KERNEL_SHA)
        raw = json.dumps(receipt["plan"], sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
        self.assertEqual(receipt["plan_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt["plan_canonical_ascii_bytes"], len(raw))
        self.assertEqual(receipt["authorized"], False)
        self.assertEqual(receipt, fit.candidate_receipt(executable_sha256=KERNEL_SHA))

    def test_caller_cannot_enlarge_resource_budget(self):
        p = fit.build_public_fit_plan(executable_sha256=KERNEL_SHA)
        self.assertEqual((p["host"], p["tier"], p["policy"]),
                         ("LAPPY", "TIER1_PUBLIC_SYNTHETIC", "WDDM_BOUNDED_COEXISTENCE"))
        self.assertEqual((p["max_seconds"], p["vram_target_mib"]), (1800, 3072))
        self.assertEqual(p["vram_limit_semantics"],
                         "OBSERVATIONAL_TARGET_NOT_HARD_WDDM_QUOTA")
        with self.assertRaises(TypeError):
            fit.build_public_fit_plan(executable_sha256=KERNEL_SHA, max_seconds=3600)

    def test_no_protected_data_or_promotion_access(self):
        p = fit.build_public_fit_plan(executable_sha256=KERNEL_SHA)
        for name in ("protected_r201_access", "protected_dev_access", "nas_q6_write",
                     "checkpoint_promotion", "gpu_physical_exclusivity_claimed",
                     "blanket_wsl_distro_termination_allowed", "launch_path_implemented",
                     "linux_orphan_cleanup_qualified"):
            self.assertIs(p[name], False, name)
        self.assertTrue(p["broker_owner_signed_grant_required"])
        self.assertTrue(p["targeted_wsl_orphan_kill_and_readback_required"])
        self.assertEqual(p["one_optimizer_step_max"], 1)
        self.assertEqual(p["wsl_cleanup"], "TARGETED_WSL_PIDTREE_REQUIRED")
        self.assertEqual(p["protected_access"], "NONE")
        self.assertEqual(p["outputs"], "UNPROMOTED_ONLY")

    def test_bad_kernel_hash_rejected(self):
        for value in (None, True, b"a" * 64, "a" * 63, "A" * 64,
                      "a" * 63 + "g", "a" * 64 + "\n"):
            with self.subTest(value=value), self.assertRaises(fit.PublicFitPlanError):
                fit.build_public_fit_plan(executable_sha256=value)

    def test_upstream_shard_lfs_oids_are_not_local_byte_attestations(self):
        hashes = fit.OFFICIAL_SHARD_LFS_OID_SHA256
        self.assertEqual(hashes["model.safetensors-00001-of-00002.safetensors"],
                         "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61")
        self.assertEqual(hashes["model.safetensors-00002-of-00002.safetensors"],
                         "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188")
        plan = fit.build_public_fit_plan(executable_sha256=KERNEL_SHA)
        self.assertEqual(set(plan["metadata_files_needing_trusted_byte_pins"]), {
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        })
        self.assertNotIn("config.json", hashes)
        self.assertNotIn("tokenizer.json", hashes)
        self.assertNotIn("5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
                         json.dumps(plan))
        self.assertNotIn("ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
                         json.dumps(plan))

    def test_585_mib_free_wddm_holds_even_at_zero_utilization(self):
        a = fit.assess_observed_vram(total_mib=4096, free_mib=585,
                                     telemetry_complete=True)
        self.assertEqual(a["status"], "HOLD_INSUFFICIENT_OBSERVED_FREE_VRAM")
        self.assertIs(a["launch_authorized"], False)
        self.assertTrue(a["wddm_foreign_clients_allowed"])

    def test_complete_preflight_is_not_authority(self):
        a = fit.assess_observed_vram(total_mib=4096, free_mib=3778,
                                     telemetry_complete=True)
        self.assertEqual(a["status"], "RESOURCE_PREFLIGHT_CANDIDATE_ONLY_NO_GPU_LEASE")
        self.assertIs(a["launch_authorized"], False)
        self.assertIs(a["physical_exclusivity_proved"], False)

    def test_telemetry_incomplete_holds(self):
        a = fit.assess_observed_vram(total_mib=4096, free_mib=3778,
                                     telemetry_complete=False)
        self.assertEqual(a["status"], "HOLD_TELEMETRY_INCOMPLETE")

    def test_invalid_memory_observation_rejected(self):
        for total, free, complete in ((4096, 4097, True), (0, 0, True),
                                      (4096, -1, True), (True, 1, True),
                                      (4096, True, True), (4096, 2000, 1)):
            with self.subTest(case=(total, free, complete)):
                with self.assertRaises(fit.PublicFitPlanError):
                    fit.assess_observed_vram(total_mib=total, free_mib=free,
                                             telemetry_complete=complete)

    def test_hashes_fixed_adjacent_kernel_bytes_not_caller_path(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "lowmem_experiment.py"
            source.write_bytes(b"# exactly pinned candidate kernel bytes\n")
            with patch.object(fit, "__file__", str(directory / "tier1_public_fit_plan.py")):
                actual = fit.source_kernel_sha256()
            self.assertEqual(actual, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_missing_and_symlinked_kernel_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(fit, "__file__", str(directory / "tier1_public_fit_plan.py")):
                with self.assertRaises(fit.PublicFitPlanError):
                    fit.source_kernel_sha256()
                outside = directory / "regular.py"
                outside.write_text("pass\n", encoding="utf-8")
                (directory / "lowmem_experiment.py").symlink_to(outside)
                with self.assertRaises(fit.PublicFitPlanError):
                    fit.source_kernel_sha256()


if __name__ == "__main__":
    unittest.main()
