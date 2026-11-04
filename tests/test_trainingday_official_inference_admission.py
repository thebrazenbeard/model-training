"""Offline fixture tests; no HF downloads, host access, or model execution."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainingday import official_inference_admission as impl


class OfficialInferenceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="qwen-admission-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_plan_pins_revision_all_files_and_never_downloads(self):
        with patch.object(impl, "preflight", return_value={"resource_preflight_pass": False}):
            report = impl.download_plan(self.root / "new-official", "VN1")
        command = report["operator_command_argv_not_executed"]
        self.assertEqual(command[:3], ["hf", "download", "Qwen/Qwen3.5-4B"])
        self.assertEqual(command[3:3 + len(impl.MODEL_FILENAMES)],
                         list(impl.MODEL_FILENAMES))
        self.assertEqual(command[command.index("--revision") + 1], impl.OFFICIAL_REVISION)
        self.assertEqual(len(impl.MODEL_FILENAMES), 9)
        self.assertFalse(report["downloaded"])
        self.assertFalse(report["authorized"])
        self.assertFalse((self.root / "new-official").exists())

    def test_existing_destination_collision_is_reported(self):
        snapshot = self.root / "occupied"
        snapshot.mkdir()
        (snapshot / "do-not-touch").write_bytes(b"user data")
        with patch.object(impl, "preflight", return_value={"resource_preflight_pass": False}):
            report = impl.download_plan(snapshot, "VN1")
        self.assertTrue(report["collision"])
        self.assertIsNone(report["operator_command_argv_not_executed"])
        self.assertEqual((snapshot / "do-not-touch").read_bytes(), b"user data")

    def test_generic_streaming_fixture_hash_and_size(self):
        sample = self.root / "synthetic.bin"
        sample.write_bytes(b"fixture-not-official-weights")
        result = impl.measure_file(
            sample, expected_bytes=sample.stat().st_size,
            expected_sha256=hashlib.sha256(sample.read_bytes()).hexdigest())
        self.assertTrue(result["verified"])
        self.assertFalse(impl.measure_file(sample, expected_bytes=5329398688)["verified"])
        self.assertFalse(impl.measure_file(sample, expected_sha256="0" * 64)["verified"])

    def test_official_snapshot_rejects_tiny_fake_shards(self):
        for name in impl.MODEL_FILENAMES:
            (self.root / name).write_bytes(b"synthetic")
        report = impl.verify_snapshot(self.root)
        self.assertFalse(report["shards_verified"])
        self.assertFalse(report["official_checkpoint_admitted"])
        self.assertFalse(report["metadata_full_byte_pins_admitted"])
        self.assertFalse(report["inference_executed"])
        self.assertEqual(len(report["files"]), 9)

    def test_symlinks_and_relative_paths_are_rejected(self):
        source = self.root / "source"
        source.write_bytes(b"synthetic")
        (self.root / "alias").symlink_to(source)
        for candidate in (Path("relative"), self.root / "alias",
                          self.root / "alias" / "nested"):
            with self.assertRaises(impl.AdmissionError):
                impl.safe_absolute(candidate)

    def test_resource_guard_is_fail_closed_on_unknown_and_conflicts(self):
        with (patch.object(impl.socket, "gethostname", return_value="VERA-NODE-1"),
              patch.object(impl, "_disk_free_bytes", return_value=100 * 1024**3),
              patch.object(impl, "_available_ram_bytes", return_value=16 * 1024**3),
              patch.object(impl, "_cpu_fraction", return_value=0.1),
              patch.object(impl, "_running_model_processes", return_value=[])):
            report = impl.preflight("VN1", self.root, stage=True)
            self.assertTrue(report["resource_preflight_pass"])
            self.assertFalse(report["host_authorized"])
            self.assertFalse(report["may_start_workload"])
        with (patch.object(impl.socket, "gethostname", return_value="VERA-NODE-2"),
              patch.object(impl, "_disk_free_bytes", return_value=None),
              patch.object(impl, "_available_ram_bytes", return_value=None),
              patch.object(impl, "_cpu_fraction", return_value=None),
              patch.object(impl, "_running_model_processes",
                           return_value=[{"pid": 1234, "name": "llama-server"}])):
            report = impl.preflight("VN2", self.root, stage=False)
            self.assertFalse(report["resource_preflight_pass"])
            self.assertFalse(report["guards"]["no_conflicting_model_process"])
            self.assertFalse(report["guards"]["disk_headroom"])

    def test_benchmark_plan_is_text_only_dry_run(self):
        gguf = self.root / "synthetic.gguf"
        gguf.write_bytes(b"GGUF" + b"synthetic-test")
        binary = self.root / "synthetic-llama-cli"
        binary.write_bytes(b"fake executable")
        with patch.object(impl, "preflight", return_value={"resource_preflight_pass": False}):
            report = impl.benchmark_plan("VN1", gguf, binary, "llama.cpp")
            self.assertTrue(report["text_only"])
            self.assertEqual(report["binary"]["version_argv"],
                             [str(binary), "--version"])
            self.assertFalse(report["binary"]["version_probe_executed"])
            command = report["operator_command_argv_not_executed"]
            self.assertEqual(command[0:3], [str(binary), "-m", str(gguf)])
            self.assertIn("--single-turn", command)
            self.assertFalse(report["inference_executed"])
            self.assertFalse(report["gguf_provenance_from_official_checkpoint_proven"])
        with patch.object(impl, "preflight", return_value={}):
            kobold = impl.benchmark_plan("VN1", gguf, binary, "KoboldCPP")
            self.assertIsNone(kobold["operator_command_argv_not_executed"])

    def test_non_gguf_file_cannot_generate_benchmark_command(self):
        gguf = self.root / "synthetic.gguf"
        gguf.write_bytes(b"WRONG")
        binary = self.root / "llama-cli"
        binary.write_bytes(b"fake")
        with self.assertRaises(impl.AdmissionError):
            impl.benchmark_plan("VN1", gguf, binary, "llama.cpp")

    def test_telemetry_parser_separates_untrusted_logs_from_inference_proof(self):
        stdout = "Four.\n"
        stderr = ("llama_perf_context_print: prompt eval time = 24.0 ms / 5 tokens "
                  "(4.8 ms per token, 208.33 tokens per second)\n"
                  "llama_perf_context_print: eval time = 66.0 ms / 6 tokens "
                  "(11.0 ms per token, 90.91 tokens per second)")
        record = impl.parse_operator_logs(stdout, stderr, returncode=0,
                                          wall_ms=103, gguf_sha256="a" * 64)
        self.assertEqual(record["generated_tokens_timing"], 6)
        self.assertEqual(record["generation_tokens_per_second"], 90.91)
        self.assertEqual(record["prompt_tokens_timing"], 5)
        self.assertTrue(record["decode_claim_in_logs"])
        self.assertFalse(record["decode_verified_by_trusted_witness"])
        self.assertIsNone(record["peak_vram_bytes"])

    def test_record_rejects_oversize_logs_and_cli_never_starts_model(self):
        log = self.root / "log.txt"
        log.write_bytes(b"x" * (impl.MAX_LOG_BYTES + 1))
        with self.assertRaises(impl.AdmissionError):
            impl._read_bounded_log(log)
        with patch.object(impl, "preflight", return_value={"resource_preflight_pass": False}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = impl.main(["plan", "--snapshot", str(self.root / "new")])
            self.assertEqual(exit_code, 0)
            self.assertFalse(json.loads(output.getvalue())["downloaded"])


if __name__ == "__main__":
    unittest.main()
