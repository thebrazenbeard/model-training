"""Source-only Windows Job Object adapter recovery contract tests."""

import math
import unittest
from dataclasses import asdict
from unittest.mock import patch

from trainingday.r82_windows_job_object_adapter import (
    FakeJobBinding,
    FakeWindowsJobBackend,
    JobRequest,
    SourceOnlyWindowsJobLauncher,
    WindowsAdapterUnavailable,
    exercise_fake_windows_binding,
)


def request(**changes):
    defaults = dict(
        run_id="training-run-001",
        executable=r"C:\Models\Trainer.exe",
        argv=("--batch", "8"),
        cwd=r"C:\Training\Runs",
        budget_seconds=120.0,
        memory_bytes=4 * 1024 ** 3,
        operator_authorized=True,
    )
    defaults.update(changes)
    return JobRequest(**defaults)


class WindowsJobObjectSourceOnlyTests(unittest.TestCase):
    def test_valid_fake_binding_has_no_live_execution_authority(self):
        fake = FakeWindowsJobBackend()
        binding = exercise_fake_windows_binding(request(), backend=fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(binding.run_id, "training-run-001")
        self.assertEqual(binding.executable_identity, r"c:\models\trainer.exe")
        self.assertEqual(binding.cwd_identity, r"c:\training\runs")
        self.assertEqual(binding.provenance, "SOURCE_ONLY_FAKE_JOB_NOT_LIVE")
        self.assertFalse(binding.authorizes_process_launch)
        self.assertFalse(binding.enforces_live_memory_limit)
        self.assertFalse(binding.enforces_live_time_limit)

    def test_production_launcher_unconditionally_fails_closed(self):
        launcher = SourceOnlyWindowsJobLauncher()
        for req in (request(), request(operator_authorized=False)):
            with self.subTest(authorized=req.operator_authorized):
                with self.assertRaises(WindowsAdapterUnavailable):
                    launcher.launch(req)

    def test_bad_budget_rejected_before_fake_backend(self):
        class IntSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        for budget in (
            float("nan"), float("inf"), -float("inf"), -1, 0,
            -0.0, 86401, 86400.0001, True, False, "100",
            IntSubclass(120), FloatSubclass(120.0), 10 ** 500,
            None, [], {}, complex(5, 0),
        ):
            with self.subTest(budget=repr(budget)[:80]):
                fake = FakeWindowsJobBackend()
                with self.assertRaises(ValueError):
                    exercise_fake_windows_binding(request(budget_seconds=budget), backend=fake)
                self.assertEqual(fake.calls, [])

    def test_valid_budget_boundaries_are_synthetic_only(self):
        for budget in (math.ldexp(1.0, -1074), 0.1, 86400, 86400.0):
            with self.subTest(budget=budget):
                fake = FakeWindowsJobBackend()
                bound = exercise_fake_windows_binding(request(budget_seconds=budget), backend=fake)
                self.assertEqual(bound.budget_seconds, budget)
                self.assertEqual(len(fake.calls), 1)

    def test_executable_and_cwd_reject_win32_device_and_noncanonical_paths(self):
        invalid = (
            r"C:\CON.txt", r"C:\CON .txt", r"C:\prn", r"C:\AUX.ini", r"C:\NUL",
            r"C:\COM1.txt", r"C:\COM1 .txt", r"C:\LPT9", "C:\\COM¹.txt", "C:\\lpt².dat",
            "C:\\LPT² .dat",
            "C:\\com³.log", r"C:\folder.\app.exe", r"C:\folder \app.exe",
            r"C:\ folder\app.exe", r"C:\..\app.exe", r"C:\.\app.exe",
            r"C:\\app.exe", r"C:/model/app.exe", r"C:relative.exe",
            r"\\server\share\app.exe", r"\\?\C:\app.exe",
            "C:\\bad\x01\\app.exe", "C:\\nul\x00.exe",
            r"C:\data:stream\app.exe",
        )
        for path in invalid:
            for field in ("executable", "cwd"):
                with self.subTest(field=field, path=repr(path)):
                    fake = FakeWindowsJobBackend()
                    with self.assertRaises(ValueError):
                        exercise_fake_windows_binding(request(**{field: path}), backend=fake)
                    self.assertEqual(fake.calls, [])

    def test_malformed_ids_argv_memory_authority_rejected(self):
        bad_fields = (
            ("run_id", "../bad"),
            ("run_id", "bad\x00id"),
            ("run_id", True),
            ("argv", ["--batch", "8"]),
            ("argv", ("--batch", "bad\x00arg")),
            ("argv", ("--batch", 8)),
            ("memory_bytes", True),
            ("memory_bytes", -1),
            ("memory_bytes", 10 ** 500),
            ("operator_authorized", 1),
            ("operator_authorized", False),
        )
        for field, value in bad_fields:
            with self.subTest(field=field, value=repr(value)[:70]):
                fake = FakeWindowsJobBackend()
                with self.assertRaises(ValueError):
                    exercise_fake_windows_binding(request(**{field: value}), backend=fake)
                self.assertEqual(fake.calls, [])

    def test_custom_backend_and_request_subclasses_rejected(self):
        class EvilBackend(FakeWindowsJobBackend):
            pass

        class EvilRequest(JobRequest):
            pass

        with self.assertRaises(ValueError):
            exercise_fake_windows_binding(request(), backend=EvilBackend())
        with self.assertRaises(ValueError):
            exercise_fake_windows_binding(EvilRequest(**asdict(request())), backend=FakeWindowsJobBackend())

    def test_direct_fake_backend_method_cannot_bypass_request_validation(self):
        fake = FakeWindowsJobBackend()
        with self.assertRaises(ValueError):
            fake.bind(request(budget_seconds=float("nan")),
                      r"c:\models\trainer.exe", r"c:\training\runs")
        self.assertEqual(fake.calls, [])
        with self.assertRaises(ValueError):
            fake.bind(request(), r"c:\different\trainer.exe", r"c:\training\runs")
        self.assertEqual(fake.calls, [])

    def test_instance_method_monkeypatch_cannot_forge_live_receipt(self):
        fake = FakeWindowsJobBackend()
        forged = FakeJobBinding(
            run_id="forged", executable_identity="forged", cwd_identity="forged",
            budget_seconds=1, memory_bytes=1,
            provenance="LIVE_WINDOWS_JOB_VERIFIED",
            authorizes_process_launch=True, enforces_live_memory_limit=True,
            enforces_live_time_limit=True,
        )
        fake.bind = lambda *_args: forged
        observed = exercise_fake_windows_binding(request(), backend=fake)
        self.assertIsNot(observed, forged)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(observed.provenance, "SOURCE_ONLY_FAKE_JOB_NOT_LIVE")
        self.assertFalse(observed.authorizes_process_launch)
        self.assertFalse(observed.enforces_live_memory_limit)
        self.assertFalse(observed.enforces_live_time_limit)

    def test_class_method_monkeypatch_cannot_forge_live_receipt(self):
        fake = FakeWindowsJobBackend()
        forged = FakeJobBinding(
            run_id="forged", executable_identity="forged", cwd_identity="forged",
            budget_seconds=1, memory_bytes=1,
            authorizes_process_launch=True, enforces_live_memory_limit=True,
            enforces_live_time_limit=True,
        )
        with patch.object(FakeWindowsJobBackend, "bind", return_value=forged):
            observed = exercise_fake_windows_binding(request(), backend=fake)
        self.assertIsNot(observed, forged)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(observed.provenance, "SOURCE_ONLY_FAKE_JOB_NOT_LIVE")
        self.assertFalse(observed.authorizes_process_launch)

    def test_forged_recording_store_rejected_before_binding(self):
        class DiscardingList(list):
            def __init__(self):
                super().__init__()
                self.append_calls = 0

            def append(self, _value):
                self.append_calls += 1

        for store in (DiscardingList(), (), {}, None, "fake"):
            with self.subTest(store_type=type(store).__name__):
                fake = FakeWindowsJobBackend()
                fake.calls = store
                with self.assertRaises(ValueError):
                    exercise_fake_windows_binding(request(), backend=fake)
                with self.assertRaises(ValueError):
                    fake.bind(request(), r"c:\models\trainer.exe", r"c:\training\runs")
                if isinstance(store, DiscardingList):
                    self.assertEqual(store.append_calls, 0)

    def test_repeated_canonical_fake_binding_records_exact_receipts(self):
        fake = FakeWindowsJobBackend()
        first = exercise_fake_windows_binding(request(), backend=fake)
        second = exercise_fake_windows_binding(request(run_id="training-run-002"), backend=fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertIs(fake.calls[0], first)
        self.assertIs(fake.calls[1], second)


if __name__ == "__main__":
    unittest.main()
