"""Source-only malicious-declaration and execution-negative regression tests."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from trainingday.official_qwen35_gguf_workflow import (
    AUXILIARY_FILES, BASE_ID, BASE_REVISION, CANONICAL_FILES,
    CUSTODY_SIDECAR_SHA256, EXPECTED_SNAPSHOT_BYTES, EXPECTED_SNAPSHOT_FILES,
    LLAMA_COMMIT, LLAMA_QUANTIZE_SHA256, LLAMA_SERVER_SHA256,
    R16_RUN, WorkflowHold, execute, expected_source_declaration,
    inspect_untrusted_declaration, main, plan,
)


class OfficialQwen35WorkflowTests(unittest.TestCase):
    def test_exact_official_model_and_snapshot_inventory(self):
        source = expected_source_declaration()
        self.assertEqual(source["model"]["id"], BASE_ID)
        self.assertEqual(source["model"]["revision"], BASE_REVISION)
        self.assertEqual(source["snapshot_file_count"], 26)
        self.assertEqual(source["snapshot_total_bytes"], 9_342_822_940)
        self.assertEqual(len(CANONICAL_FILES), 11)
        self.assertEqual(len(AUXILIARY_FILES), 15)
        self.assertEqual(len(set(AUXILIARY_FILES)), 15)
        self.assertEqual(len(set(name for name, _, _ in CANONICAL_FILES)), 11)
        self.assertEqual(EXPECTED_SNAPSHOT_FILES, 26)
        self.assertEqual(EXPECTED_SNAPSHOT_BYTES, 9_342_822_940)
        self.assertEqual(CUSTODY_SIDECAR_SHA256,
                         "06e730744a578cc2168d1b587711f82a8b7e691545182ff1b814dff95e450fca")

    def test_11_pin_shapes_and_canonical_bytes(self):
        pins = expected_source_declaration()["canonical_files"]
        self.assertEqual(len(pins), 11)
        for item in pins:
            self.assertGreater(item["bytes"], 0)
            self.assertEqual(len(item["sha256"]), 64)
            self.assertEqual(set(item), {"file", "bytes", "sha256"})
        indexed = {item["file"]: item for item in pins}
        self.assertEqual(indexed["tokenizer.json"]["sha256"],
                         "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42")
        self.assertEqual(indexed["model.safetensors-00001-of-00002.safetensors"]["bytes"],
                         5_329_398_688)
        self.assertEqual(indexed["model.safetensors-00002-of-00002.safetensors"]["bytes"],
                         3_990_429_408)
        self.assertLess(sum(item["bytes"] for item in pins), EXPECTED_SNAPSHOT_BYTES)

    def test_adapter_identity_is_correct_label_not_corrupted_control(self):
        adapter = expected_source_declaration()["r16_correct_label"]
        self.assertEqual(adapter["run"], R16_RUN)
        self.assertEqual(adapter["files"][0]["sha256"],
                         "322d91e034cd2b2cf63b077e98061559478a4271c0bf5d4abd8520dca195fc11")
        self.assertNotIn("corrupt8", str(adapter))

    def test_exact_toolchain_and_source_pins(self):
        runtime = plan()["toolchain"]
        self.assertEqual(runtime["commit"], LLAMA_COMMIT)
        self.assertEqual(runtime["release"], "b12098")
        self.assertEqual(runtime["windows_server_exe_sha256"], LLAMA_SERVER_SHA256)
        self.assertEqual(runtime["windows_quantize_exe_sha256"], LLAMA_QUANTIZE_SHA256)
        self.assertFalse(runtime["version_execution_qualified"])

    def test_recipe_bf16_excludes_nextn_and_quantizes_q4(self):
        c = plan()["conversion"]
        self.assertEqual(c["base"]["outtype"], "bf16")
        self.assertEqual(c["base"]["exclude_nextn"], "--no-mtp")
        self.assertTrue(c["base"]["vision_mmproj_required_for_multimodal"])
        self.assertEqual(c["quantize"]["output_format"], "Q4_K_M")
        self.assertEqual(c["correct8_lora"]["outtype"], "bf16")
        self.assertFalse(c["official_derived_gguf_verified"])
        self.assertFalse(c["correct8_adapter_derived_gguf_verified"])

    def test_three_inference_states_are_not_conflated(self):
        states = plan()["inference_identities"]
        self.assertEqual([s["state"] for s in states], ["base", "trained", "interface"])
        self.assertIsNone(states[0]["adapter"])
        self.assertEqual(states[1]["adapter"], R16_RUN)
        self.assertEqual(states[2]["adapter"], R16_RUN)
        self.assertTrue(all(s["runtime_verified"] is False for s in states))

    def test_declaration_equal_is_syntax_not_physical_custody(self):
        r = inspect_untrusted_declaration(expected_source_declaration())
        self.assertTrue(r["declaration_matches_source_pins"])
        self.assertFalse(r["source_files_freshly_hashed"])
        self.assertFalse(r["custody_sidecar_independently_authenticated"])
        self.assertFalse(r["conversion_lineage_verified"])
        self.assertFalse(r["physical_launch_allowed"])

    def test_forged_authority_boolean_rejected(self):
        forged = expected_source_declaration()
        forged["physical_launch_allowed"] = True
        with self.assertRaisesRegex(WorkflowHold, "exact typed"):
            inspect_untrusted_declaration(forged)

    def test_bool_as_integer_or_wrong_digest_rejected(self):
        forged = expected_source_declaration()
        forged["snapshot_file_count"] = True
        with self.assertRaises(WorkflowHold):
            inspect_untrusted_declaration(forged)
        forged = expected_source_declaration()
        forged["canonical_files"][7]["sha256"] = "0" * 64
        with self.assertRaises(WorkflowHold):
            inspect_untrusted_declaration(forged)

    def test_swapped_shards_and_missing_cache_metadata_rejected(self):
        forged = expected_source_declaration()
        forged["canonical_files"][3], forged["canonical_files"][4] = (
            forged["canonical_files"][4], forged["canonical_files"][3])
        with self.assertRaises(WorkflowHold):
            inspect_untrusted_declaration(forged)
        forged = expected_source_declaration()
        forged["auxiliary_file_names_without_pinned_sha"].pop()
        with self.assertRaises(WorkflowHold):
            inspect_untrusted_declaration(forged)

    def test_arbitrary_tool_path_and_corrupted_label_rejected(self):
        forged = expected_source_declaration()
        forged["llama_server"] = {"path": "C:\\evil.exe", "sha256": "0" * 64}
        with self.assertRaises(WorkflowHold):
            inspect_untrusted_declaration(forged)
        forged = expected_source_declaration()
        forged["r16_correct_label"]["run"] = "r16-v2-corrupt8-20261030-b"
        with self.assertRaises(WorkflowHold):
            inspect_untrusted_declaration(forged)

    def test_execute_unconditionally_refuses_even_with_forged_gates(self):
        with patch("subprocess.Popen") as launch:
            with patch("socket.socket") as listener:
                with self.assertRaisesRegex(WorkflowHold, "cannot launch"):
                    execute({"trusted": True, "owner_idle": True}, "--allow-everything")
                launch.assert_not_called()
                listener.assert_not_called()

    def test_plan_is_immutable_between_callers(self):
        a = plan()
        a["ready"] = True
        a["toolchain"]["version_execution_qualified"] = True
        a["source"]["canonical_files"].clear()
        b = plan()
        self.assertIs(b["ready"], False)
        self.assertIs(b["toolchain"]["version_execution_qualified"], False)
        self.assertEqual(len(b["source"]["canonical_files"]), 11)

    def test_cli_is_nonzero_hard_hold_and_no_effects(self):
        out = io.StringIO()
        with patch("subprocess.Popen") as launch:
            with redirect_stdout(out):
                code = main(["--json"])
            launch.assert_not_called()
        self.assertEqual(code, 1)
        self.assertIn('"ready":false', out.getvalue())
        self.assertIn('"physical_launch_attempted":false', out.getvalue())


if __name__ == "__main__":
    unittest.main()
