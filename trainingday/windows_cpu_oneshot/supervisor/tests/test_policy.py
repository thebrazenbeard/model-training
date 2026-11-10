"""Offline Linux/Windows policy tests; these are NOT Win32 launch tests."""

import json
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from policy import (GIB, Hold, MIN_FREE_BYTES, MODEL_SHA256, PROMPT,
                    REVIEWED_RUNNER_SHA256, REVIEWED_RUNTIME_FILES_SHA256,
                    TRIAL_ROOT, parse_plan, required_prelaunch_free,
                    state_root_for_mode, blocking_processes_for_mode,
                    ollama_inventory_empty, competing_model_workers)


def valid():
    return {
        "schema": "VERA_VN1_CPU_Q4_ONESHOT_V1",
        "host": "VERA-NODE-1",
        "plan_id": "0123456789abcdef0123456789abcdef",
        "executable": r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot\bin\llama-cli.exe",
        "executable_sha256": "f" * 64,
        "model": r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot\models\Qwen3.5-4B-Q4_K_M.gguf",
        "model_sha256": MODEL_SHA256,
        "context": 1024,
        "output_tokens": 64,
        "threads": 2,
        "wall_seconds": 180,
        "job_memory_bytes": 4 * GIB,
        "cpu_rate_percent": 20,
    }


class PolicyTests(unittest.TestCase):
    def parse(self, record):
        with patch("policy.REVIEWED_RUNNER_SHA256", "f" * 64), patch(
            "policy.REVIEWED_RUNTIME_FILES_SHA256", {"llama-cli.exe": "f" * 64}
        ):
            return parse_plan(json.dumps(record))

    def test_official_release_binary_manifest_is_frozen(self):
        self.assertEqual(len(REVIEWED_RUNTIME_FILES_SHA256), 19)
        self.assertEqual(REVIEWED_RUNNER_SHA256,
                         "02042615249dd4852f47cc2d4dd5caabba91766b5161dd4642ed6272f788172e")
        self.assertEqual(REVIEWED_RUNTIME_FILES_SHA256["llama-cli.exe"],
                         REVIEWED_RUNNER_SHA256)
        self.assertEqual(REVIEWED_RUNTIME_FILES_SHA256["llama.dll"],
                         "669c47bee16dcad5f6e6d0755f16c9d0969508c1ac1983f3e2f74280ab44822a")

    def test_normal_fixed_cpu_only_plan(self):
        plan = self.parse(valid())
        argv = plan.args()
        self.assertEqual(argv[0], valid()["executable"])
        self.assertEqual(argv[argv.index("--gpu-layers") + 1], "0")
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "1024")
        self.assertEqual(argv[argv.index("--predict") + 1], "64")
        self.assertEqual(argv[argv.index("--seed") + 1], "0")
        self.assertEqual(argv[argv.index("--temp") + 1], "0")
        self.assertEqual(argv[argv.index("--prompt") + 1], PROMPT)
        self.assertEqual(argv[argv.index("--fit") + 1], "off")
        self.assertEqual(argv[argv.index("--reasoning") + 1], "off")
        self.assertIn("--offline", argv)
        self.assertIn("--single-turn", argv)
        self.assertGreaterEqual(required_prelaunch_free(plan), MIN_FREE_BYTES)
        self.assertGreaterEqual(required_prelaunch_free(plan) - plan.job_memory_bytes,
                                1536 * 1024 ** 2)

    def test_gpu_memory_flags_cannot_be_supplied(self):
        for key, value in [("ngl", 99), ("extra_args", ["--server"]),
                           ("prompt", "read private chat"), ("port", 9000)]:
            with self.subTest(key=key):
                x = valid()
                x[key] = value
                with self.assertRaises(Hold):
                    self.parse(x)

    def test_flags_do_not_allow_cpu_expansion(self):
        for key, value in [
            ("context", 1025), ("output_tokens", 65), ("threads", 3),
            ("wall_seconds", 181), ("job_memory_bytes", 5 * GIB),
            ("cpu_rate_percent", 26), ("context", False),
        ]:
            with self.subTest(key=key, value=value):
                x = valid()
                x[key] = value
                with self.assertRaises(Hold):
                    self.parse(x)

    def test_artifact_pins_are_exact(self):
        for key, value in [
            ("model_sha256", "1" * 64),
            ("executable_sha256", "f" * 63),
            ("model", r"D:\VERA\Qwen3.5-4B-Q4_K_M.gguf"),
            ("model", r"C:\Users\Vera\BT2ModelArtifacts\..\other.gguf"),
            ("executable", r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot\bin\cmd.exe"),
            ("executable", r"C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot\bin\llama-cli.exe:ads"),
            ("executable", r"\\somehost\share\llama-cli.exe"),
            ("plan_id", "h" * 32),
        ]:
            with self.subTest(key=key, value=value):
                x = valid()
                x[key] = value
                with self.assertRaises(Hold):
                    self.parse(x)

    def test_wrong_host_or_unknown_plan_schema_hold(self):
        for key, value in [("host", "VN2"), ("schema", "ARBITRARY"),
                           ("context", "1024")]:
            with self.subTest(key=key):
                x = valid()
                x[key] = value
                with self.assertRaises(Hold):
                    self.parse(x)

    def test_reject_non_json_and_missing_fields(self):
        with self.assertRaises(Hold):
            parse_plan("bad json")
        x = valid()
        x.pop("model_sha256")
        with self.assertRaises(Hold):
            self.parse(x)

    def test_uninstalled_policy_holds_even_valid_manifest(self):
        with self.assertRaisesRegex(Hold, "not independently approved"):
            parse_plan(json.dumps(valid()))

    def test_harmless_selftest_does_not_populate_atomic_trial_leaf(self):
        selftest = state_root_for_mode("--self-test")
        model_run = state_root_for_mode("--run")
        self.assertNotEqual(selftest, model_run)
        self.assertNotEqual(selftest, TRIAL_ROOT)
        self.assertNotIn(TRIAL_ROOT, selftest.parents)
        self.assertEqual(model_run.parent, TRIAL_ROOT)
        self.assertEqual(selftest.name, "w56-q4-selftest")
        with self.assertRaises(Hold):
            state_root_for_mode("--unknown")

    def test_idle_ollama_daemon_exception_cannot_authorize_model_run(self):
        harmless = blocking_processes_for_mode("--self-test")
        physical = blocking_processes_for_mode("--run")
        self.assertNotIn("ollama.exe", harmless)
        self.assertIn("ollama.exe", physical)
        for actual_worker in ("ollama_llama_server.exe", "llama-cli.exe",
                              "llama-server.exe", "torchrun.exe", "koboldcpp.exe"):
            self.assertIn(actual_worker, harmless)
            self.assertIn(actual_worker, physical)
        with self.assertRaises(Hold):
            blocking_processes_for_mode("--unknown")

    def test_ollama_api_requires_unambiguous_empty_loaded_models(self):
        self.assertTrue(ollama_inventory_empty(b'{"models": []}'))
        for response in (
            b'{"models": [{"name": "qwen3.5:4b"}]}',
            b'{"models": null}', b'{}', b'{"running": []}',
            b'invalid JSON', b'["models"]', b'{"models": ""}',
            b' ' * 65537,
        ):
            with self.subTest(response=response[:70]):
                with self.assertRaises(Hold):
                    ollama_inventory_empty(response)

    def test_only_exact_owned_llama_cli_pid_is_exempt(self):
        processes = {"ollama.exe": {5300}, "llama-cli.exe": {111, 222}}
        self.assertEqual(competing_model_workers(processes), ["llama-cli.exe"])
        self.assertEqual(competing_model_workers(processes, 111), ["llama-cli.exe"])
        self.assertEqual(competing_model_workers({"llama-cli.exe": {111}}, 111), [])
        self.assertEqual(competing_model_workers({"llama-cli.exe": {222}}, 111), ["llama-cli.exe"])
        self.assertEqual(competing_model_workers({"ollama_llama_server.exe": {444}}, 111),
                         ["ollama_llama_server.exe"])
        with self.assertRaises(Hold):
            competing_model_workers({}, owner_pid=0)


if __name__ == "__main__":
    unittest.main()
