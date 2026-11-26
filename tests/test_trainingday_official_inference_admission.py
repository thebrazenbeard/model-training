"""Offline fixture tests; no HF downloads, host access, or model execution."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
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

    def test_late_first_shard_path_replacement_holds_entire_hash_audit(self):
        """W29: first FD hashes correctly, then its pathname is replaced."""
        first_name, second_name = (name for name, _, _ in impl.SHARDS)
        first = self.root / first_name
        first.write_bytes(b"A")
        (self.root / second_name).write_bytes(b"B")
        for name in impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
            (self.root / name).write_bytes(b"metadata")
        tiny_pins = ((first_name, 1, hashlib.sha256(b"A").hexdigest()),
                     (second_name, 1, hashlib.sha256(b"B").hexdigest()))
        actual_measure = impl._measure_fd
        calls = [0]

        def swap_after_hash(fd, **kwargs):
            measured = actual_measure(fd, **kwargs)
            calls[0] += 1
            if calls[0] == 1:
                first.rename(self.root / "first-shard-old-inode")
                first.write_bytes(b"X")
            return measured

        with (patch.object(impl, "SHARDS", tiny_pins),
              patch.object(impl, "_measure_fd", side_effect=swap_after_hash),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO()) as stderr):
            exit_code = impl.main(["hash-audit", "--snapshot", str(self.root)])
        self.assertGreaterEqual(calls[0], 1)
        self.assertEqual(exit_code, 2, stderr.getvalue())
        self.assertIn('"status": "HOLD"', stderr.getvalue())

    def test_late_snapshot_ancestor_swap_holds_lexical_path_audit(self):
        """W30: retained snapshot dir stays valid after its parent path escapes."""
        parent = self.root / "snapshot-parent"
        snapshot = parent / "official"
        snapshot.mkdir(parents=True)
        bogus_parent = self.root / "bogus-parent"
        bogus = bogus_parent / "official"
        bogus.mkdir(parents=True)
        tiny = []
        for letter, (name, _, _) in zip((b"A", b"B"), impl.SHARDS):
            (snapshot / name).write_bytes(letter)
            (bogus / name).write_bytes(b"X")
            tiny.append((name, 1, hashlib.sha256(letter).hexdigest()))
        for name in impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
            (snapshot / name).write_bytes(b"official-metadata")
            (bogus / name).write_bytes(b"bogus-metadata")
        actual_measure = impl._measure_fd
        calls = [0]

        def swap_after_first_hash(fd, **kwargs):
            result = actual_measure(fd, **kwargs)
            calls[0] += 1
            if calls[0] == 1:
                parent.rename(self.root / "snapshot-parent-original")
                parent.symlink_to(bogus_parent, target_is_directory=True)
            return result

        with (patch.object(impl, "SHARDS", tuple(tiny)),
              patch.object(impl, "_measure_fd", side_effect=swap_after_first_hash),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO()) as stderr):
            exit_code = impl.main(["hash-audit", "--snapshot", str(snapshot)])
        self.assertGreaterEqual(calls[0], 1)
        self.assertEqual(exit_code, 2, stderr.getvalue())
        self.assertIn('"status": "HOLD"', stderr.getvalue())

    def test_late_snapshot_ancestor_plain_directory_replacement_holds(self):
        """Final absolute rewalk checks inode, not merely absence of symlinks."""
        parent = self.root / "parent"
        snapshot = parent / "official"
        snapshot.mkdir(parents=True)
        tiny = []
        for letter, (name, _, _) in zip((b"A", b"B"), impl.SHARDS):
            (snapshot / name).write_bytes(letter)
            tiny.append((name, 1, hashlib.sha256(letter).hexdigest()))
        for name in impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
            (snapshot / name).write_bytes(b"metadata")
        actual_measure = impl._measure_fd
        swapped = [False]

        def replace_parent(fd, **kwargs):
            result = actual_measure(fd, **kwargs)
            if not swapped[0]:
                swapped[0] = True
                parent.rename(self.root / "old-parent")
                (parent / "official").mkdir(parents=True)
            return result

        with (patch.object(impl, "SHARDS", tuple(tiny)),
              patch.object(impl, "_measure_fd", side_effect=replace_parent)):
            with self.assertRaisesRegex(impl.AdmissionError, "absolute snapshot pathname"):
                impl.verify_snapshot(snapshot)
        self.assertTrue(swapped[0])

    def test_late_metadata_path_replacement_holds_entire_hash_audit(self):
        """Metadata names are revalidated too, not merely the two shards."""
        tiny = []
        for letter, (name, _, _) in zip((b"A", b"B"), impl.SHARDS):
            (self.root / name).write_bytes(letter)
            tiny.append((name, 1, hashlib.sha256(letter).hexdigest()))
        for name in impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
            (self.root / name).write_bytes(b"metadata")
        first_metadata = self.root / impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS[0]
        actual_measure = impl._measure_fd
        count = [0]

        def swap_after_last_hash(fd, **kwargs):
            result = actual_measure(fd, **kwargs)
            count[0] += 1
            if count[0] == len(impl.MODEL_FILENAMES):
                first_metadata.rename(self.root / "previous-metadata-inode")
                first_metadata.write_bytes(b"malicious")
            return result

        with (patch.object(impl, "SHARDS", tuple(tiny)),
              patch.object(impl, "_measure_fd", side_effect=swap_after_last_hash)):
            with self.assertRaisesRegex(impl.AdmissionError, "changed after read|pathname replaced"):
                impl.verify_snapshot(self.root)
        self.assertEqual(count[0], len(impl.MODEL_FILENAMES))

    def test_shard_write_between_hash_return_and_final_audit_holds(self):
        first_name, second_name = (name for name, _, _ in impl.SHARDS)
        first = self.root / first_name
        first.write_bytes(b"A")
        (self.root / second_name).write_bytes(b"B")
        for name in impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
            (self.root / name).write_bytes(b"metadata")
        tiny = ((first_name, 1, hashlib.sha256(b"A").hexdigest()),
                (second_name, 1, hashlib.sha256(b"B").hexdigest()))
        actual_measure = impl._measure_fd
        calls = [0]

        def mutate_after_first_hash(fd, **kwargs):
            result = actual_measure(fd, **kwargs)
            calls[0] += 1
            if calls[0] == 1:
                before = first.stat()
                first.write_bytes(b"X")  # Same inode, changed contents.
                os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result

        with (patch.object(impl, "SHARDS", tiny),
              patch.object(impl, "_measure_fd", side_effect=mutate_after_first_hash)):
            with self.assertRaisesRegex(impl.AdmissionError, "changed during shard audit"):
                impl.verify_snapshot(self.root)

    def test_successful_synthetic_hash_audit_marks_nonatomic_revalidation(self):
        tiny = []
        for letter, (name, _, _) in zip((b"A", b"B"), impl.SHARDS):
            (self.root / name).write_bytes(letter)
            tiny.append((name, 1, hashlib.sha256(letter).hexdigest()))
        for name in impl.METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS:
            (self.root / name).write_bytes(b"metadata")
        with patch.object(impl, "SHARDS", tuple(tiny)):
            receipt = impl.verify_snapshot(self.root)
        self.assertTrue(receipt["shards_verified"])
        self.assertTrue(receipt["all_snapshot_entries_revalidated"])
        self.assertTrue(receipt["absolute_snapshot_directory_revalidated"])
        self.assertFalse(receipt["official_checkpoint_admitted"])
        self.assertIn("non-atomic", receipt["file_identity_limit"])

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

    def test_ancestor_swap_cannot_redirect_open_into_external_file(self):
        parent = self.root / "safe"
        parent.mkdir()
        original = parent / "model.bin"
        original.write_bytes(b"A" * 32)
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "model.bin").write_bytes(b"B" * 32)
        real_open = impl.os.open
        swapped = [False]

        def race(path, flags, *args, **kwargs):
            if path == "model.bin" and not swapped[0]:
                swapped[0] = True
                parent.rename(self.root / "safe_old")
                parent.symlink_to(outside, target_is_directory=True)
            return real_open(path, flags, *args, **kwargs)

        with patch.object(impl.os, "open", side_effect=race):
            receipt = impl.measure_file(original)
        self.assertTrue(swapped[0])
        self.assertEqual(receipt["sha256_measured"], hashlib.sha256(b"A" * 32).hexdigest())

    def test_ancestor_symlink_before_component_open_fails_closed(self):
        parent = self.root / "safe"
        parent.mkdir()
        file = parent / "model.bin"
        file.write_bytes(b"A")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "model.bin").write_bytes(b"B")
        real_open = impl.os.open
        swapped = [False]

        def race(path, flags, *args, **kwargs):
            if path == "safe" and not swapped[0]:
                swapped[0] = True
                parent.rename(self.root / "safe_old")
                parent.symlink_to(outside, target_is_directory=True)
            return real_open(path, flags, *args, **kwargs)

        with patch.object(impl.os, "open", side_effect=race):
            with self.assertRaises(impl.AdmissionError):
                impl.measure_file(file)
        self.assertTrue(swapped[0])

    def test_log_symlink_replacement_after_size_check_stays_bounded(self):
        log = self.root / "log.txt"
        log.write_text("ok")
        outside = self.root / "large.txt"
        outside.write_bytes(b"X" * (impl.MAX_LOG_BYTES + 1))
        real_read = impl.os.read
        swapped = [False]

        def race(fd, size):
            if not swapped[0]:
                swapped[0] = True
                log.rename(self.root / "log_old")
                log.symlink_to(outside)
            return real_read(fd, size)

        with patch.object(impl.os, "read", side_effect=race):
            with self.assertRaisesRegex(impl.AdmissionError, "log changed"):
                impl._read_bounded_log(log)
        self.assertTrue(swapped[0])

    def test_in_place_rewrite_restored_mtime_rejected_by_ctime(self):
        file = self.root / "data.bin"
        file.write_bytes(b"A" * (2 * 1024 * 1024))
        before = file.stat()
        real_read = impl.os.read
        calls = [0]

        def rewrite(fd, size):
            block = real_read(fd, size)
            calls[0] += 1
            if calls[0] == 1:
                with file.open("r+b") as output:
                    output.seek(1024 * 1024)
                    output.write(b"B" * (1024 * 1024))
                os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
            return block

        with patch.object(impl.os, "read", side_effect=rewrite):
            with self.assertRaisesRegex(impl.AdmissionError, "changed during read"):
                impl.measure_file(file)

    def test_hash_audit_is_separate_from_admission_exit_code(self):
        synthetic = {"shards_verified": True,
                     "official_checkpoint_admitted": False,
                     "status": "LOCAL_SHARD_HASH_AUDIT_PASS_NOT_ADMISSION"}
        with patch.object(impl, "verify_snapshot", return_value=synthetic):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, impl.main(["hash-audit", "--snapshot", str(self.root)]))
                self.assertEqual(3, impl.main(["verify", "--snapshot", str(self.root)]))

    def test_malformed_timing_is_null_not_exception_or_nan(self):
        for prefix, fields in (
            ("eval time", ("generation_tokens_per_second", "generated_tokens_timing")),
            ("prompt eval time", ("prompt_tokens_per_second", "prompt_tokens_timing")),
        ):
            for malformed in ("1e309", "1.2.3", "NaN", ".."):
                with self.subTest(prefix=prefix, malformed=malformed):
                    line = f"{prefix} = 1 ms / 1 tokens ({malformed} tokens per second)"
                    result = impl.parse_operator_logs(
                        line, "", returncode=0, wall_ms=1, gguf_sha256="0" * 64)
                    for field in fields:
                        self.assertIsNone(result[field])

    def test_complete_parenthetical_timing_accepts_valid_llama_format(self):
        output = ("eval time = 66.0 ms / 6 tokens "
                  "(11.0 ms per token, 90.91 tokens per second)\n"
                  "prompt eval time = 24.0 ms / 5 tokens "
                  "(4.8 ms per token, 208.33 tokens per second)")
        record = impl.parse_operator_logs(output, "", returncode=0,
                                          wall_ms=90, gguf_sha256="0" * 64)
        self.assertEqual(record["generation_tokens_per_second"], 90.91)
        self.assertEqual(record["generated_tokens_timing"], 6)
        self.assertEqual(record["prompt_tokens_per_second"], 208.33)
        self.assertEqual(record["prompt_tokens_timing"], 5)

    def test_windows_without_safe_handle_opener_fails_closed(self):
        file = self.root / "tiny.bin"
        file.write_bytes(b"tiny")
        with patch.object(impl, "_anchored_fd_supported", return_value=False):
            with self.assertRaisesRegex(impl.AdmissionError, "unavailable"):
                impl.measure_file(file)
            with self.assertRaisesRegex(impl.AdmissionError, "unavailable"):
                impl._read_bounded_log(file)


if __name__ == "__main__":
    unittest.main()
