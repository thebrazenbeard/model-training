"""Zero-GPU controls for the separate memory telemetry classification."""
import inspect
import subprocess
import sys
import unittest
from unittest.mock import patch

from trainingday.lowmem_budget_telemetry import (
    MemoryTelemetryError, capture_memory_phase, parse_linux_kib,
    summarize_memory_timeline,
)
from trainingday.lowmem_experiment import one_step_kernel


class MockCuda:
    def __init__(self):
        self.synchronized = 0
        self.observed_device_indices = []

    def synchronize(self, device):
        self.synchronized += 1
        self.observed_device_indices.append(device)

    def mem_get_info(self, device):
        self.observed_device_indices.append(device)
        return (1_000_000_000, 4_000_000_000)

    def memory_allocated(self, device):
        self.observed_device_indices.append(device)
        return 400_000_000

    def memory_reserved(self, device):
        self.observed_device_indices.append(device)
        return 500_000_000

    def max_memory_allocated(self, device):
        self.observed_device_indices.append(device)
        return 600_000_000

    def max_memory_reserved(self, device):
        self.observed_device_indices.append(device)
        return 700_000_000


class MockTorch:
    def __init__(self):
        self.cuda = MockCuda()


class MemoryTelemetryTests(unittest.TestCase):
    def test_capture_has_device_wide_and_process_scope(self):
        torch = MockTorch()
        with patch("trainingday.lowmem_budget_telemetry.read_linux_host_memory",
                   return_value={"host_mem_available_bytes": 10_000_000_000,
                                 "process_resident_bytes": 200_000_000}):
            records = [capture_memory_phase(torch, phase=phase, device_index=0)
                       for phase in ("before_load", "after_load", "after_step")]
        result = summarize_memory_timeline(records)
        self.assertEqual(torch.cuda.synchronized, 3)
        self.assertEqual(torch.cuda.observed_device_indices, [0] * 18)
        self.assertEqual(result["device_index"], 0)
        self.assertEqual(result["device_total_bytes"], 4_000_000_000)
        self.assertEqual(result["minimum_sampled_device_free_bytes"], 1_000_000_000)
        self.assertEqual(result["maximum_process_cuda_peak_reserved_bytes"], 700_000_000)
        self.assertEqual(result["four_gib_training_fit"], "UNKNOWN")
        self.assertFalse(result["authority_or_weight_custody_proven"])
        self.assertFalse(result["training_authorized"])

    def test_proc_parser_strict_single_positive_kib(self):
        self.assertEqual(parse_linux_kib("MemTotal: 20 kB\nMemAvailable: 15 kB",
                                         "MemAvailable"), 15 * 1024)
        for bad in ("MemAvailable: 0 kB", "MemAvailable: -1 kB",
                    "MemAvailable: 9 MB", "MemAvailable: 9 kB\nMemAvailable: 10 kB",
                    "MemTot: 4 kB"):
            with self.subTest(bad=bad), self.assertRaises(MemoryTelemetryError):
                parse_linux_kib(bad, "MemAvailable")

    def test_reject_gpu_counter_conflicts_and_missing_phase(self):
        with patch("trainingday.lowmem_budget_telemetry.read_linux_host_memory",
                   return_value={"host_mem_available_bytes": 9_000_000_000,
                                 "process_resident_bytes": 200_000_000}):
            records = [capture_memory_phase(MockTorch(), phase=phase, device_index=0)
                       for phase in ("before_load", "after_load", "after_step")]
        cases = (
            (0, "device_free_bytes", 4_000_000_001),
            (1, "process_cuda_reserved_bytes", 4_000_000_001),
            (1, "device_total_bytes", 8_000_000_000),
            (2, "host_mem_available_bytes", None),
            (2, "process_cuda_peak_reserved_bytes", -1),
        )
        for idx, field, bad in cases:
            corrupt = [dict(record) for record in records]
            corrupt[idx][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(
                    MemoryTelemetryError):
                summarize_memory_timeline(corrupt)
        with self.assertRaises(MemoryTelemetryError):
            summarize_memory_timeline(records[:2])
        with self.assertRaises(MemoryTelemetryError):
            summarize_memory_timeline([records[1], records[0], records[2]])
        # GPU 0 and GPU 1 can have exactly equal physical memory capacity.
        # Capacity equality must never substitute for device identity.
        same_capacity_other_gpu = [dict(record) for record in records]
        same_capacity_other_gpu[1]["device_index"] = 1
        with self.assertRaises(MemoryTelemetryError):
            summarize_memory_timeline(same_capacity_other_gpu)
        with self.assertRaises(MemoryTelemetryError):
            summarize_memory_timeline([dict(record, device_index=True)
                                       for record in records])
        for bad_device in (True, -1, "0", None):
            with self.subTest(bad_device=bad_device), self.assertRaises(
                    MemoryTelemetryError):
                capture_memory_phase(MockTorch(), phase="before_load",
                                     device_index=bad_device)

    def test_protected_kernel_collects_each_phase_without_relaxing_hold(self):
        source = inspect.getsource(one_step_kernel)
        hold = source.index("require_physical_launch_authority(")
        first_import = source.index("    import torch")
        self.assertLess(hold, first_import)
        for phase in ("before_load", "after_load", "after_step"):
            self.assertIn(f'capture_memory_phase(torch, phase="{phase}", device_index=0)', source)
        self.assertIn('witness["device_host_memory"] = summarize_memory_timeline(', source)
        from trainingday.physical_launch_authority import PhysicalLaunchAuthorityUnavailable
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            one_step_kernel("/absent/official")

    def test_import_has_no_torch_or_cuda(self):
        proc = subprocess.run([sys.executable, "-c",
                               "import sys; import trainingday.lowmem_budget_telemetry; "
                               "assert 'torch' not in sys.modules"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
