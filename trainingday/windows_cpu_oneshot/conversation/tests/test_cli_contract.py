"""Static exact b12125 CLI safety contract; no process is started."""

import unittest

from q4_conversation_eval.cli_contract import (
    CPU_BASE_FLAGS,
    WINDOWS_CPU_ARCHIVE,
    WINDOWS_CPU_ARCHIVE_SHA256,
    build_first_turn_args,
)


EXE = r"C:\VeraTest\llama-cli.exe"
MODEL = r"D:\VeraTest\Qwen3.5-4B-Q4_K_M.gguf"


class CliContractTests(unittest.TestCase):
    def test_exact_release_archive_digest_pin(self):
        self.assertEqual(WINDOWS_CPU_ARCHIVE, "llama-b12125-bin-win-cpu-x64.zip")
        self.assertEqual(WINDOWS_CPU_ARCHIVE_SHA256,
                         "968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8")

    def test_one_public_first_turn_no_network_persistent_service_or_implicit_fit(self):
        argv = build_first_turn_args(EXE, MODEL, "self_identification")
        self.assertEqual(argv[:3], [EXE, "--model", MODEL])
        self.assertIn("--single-turn", argv)
        self.assertIn("--prompt", argv)
        self.assertIn("What name should I use for you?", argv[-1])
        for flag, value in (
            ("--gpu-layers", "0"),
            ("--ctx-size", "1024"),
            ("--predict", "64"),
            ("--fit", "off"),
            ("--reasoning", "off"),
            ("--reasoning-budget", "0"),
        ):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        for flag in ("--offline", "--log-disable", "--no-display-prompt",
                     "--no-show-timings", "--simple-io", "--no-context-shift"):
            self.assertIn(flag, argv)
        for forbidden in ("--model-url", "--hf-repo", "--hf-token",
                          "--server-base", "--lora", "--log-prompts-dir",
                          "--output", "--log-file"):
            self.assertNotIn(forbidden, argv)
        self.assertEqual(tuple(argv[3:-3]), CPU_BASE_FLAGS)

    def test_can_only_represent_known_public_first_turns(self):
        self.assertIn("final_FINAL_v97", build_first_turn_args(
            EXE, MODEL, "dry_humor")[-1])
        for bad in ("private prompt", "", None, {"prompt": "secret"}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                build_first_turn_args(EXE, MODEL, bad)

    def test_rejects_uncanonical_file_paths_and_command_injection(self):
        for bad_exe in ("llama-cli.exe", r"\\server\share\llama-cli.exe",
                        r"C:\VeraTest\..\llama-cli.exe", "powershell.exe -Command",
                        r"C:\VeraTest\llama-cli.txt"):
            with self.subTest(exe=bad_exe), self.assertRaises(ValueError):
                build_first_turn_args(bad_exe, MODEL, "dry_humor")
        for bad_model in ("Q4_K_M.gguf", r"D:\VeraTest\..\Q4_K_M.gguf",
                          r"\\server\share\Q4_K_M.gguf", "https://remote/model.gguf"):
            with self.subTest(model=bad_model), self.assertRaises(ValueError):
                build_first_turn_args(EXE, bad_model, "dry_humor")


if __name__ == "__main__":
    unittest.main()
