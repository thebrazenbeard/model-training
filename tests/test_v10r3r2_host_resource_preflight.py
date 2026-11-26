from __future__ import annotations

import pytest

from successor.experiments.preflight_v10r3r2_host_resources import (
    HostResourceHold,
    validate_host_resources,
)


def test_host_resource_gate_passes_with_safe_headroom() -> None:
    result = validate_host_resources(
        {
            "available_physical_gib": 12.0,
            "commit_headroom_gib": 14.0,
            "committed_percent": 62.0,
            "gpu_memory_used_mib": 0,
            "gpu_memory_total_mib": 4096,
            "gpu_utilization_percent": 0,
            "competing_trainers": [],
        },
        min_available_physical_gib=8.0,
        min_commit_headroom_gib=8.0,
    )

    assert result["status"] == "PASS"
    assert result["available_physical_gib"] == 12.0
    assert result["commit_headroom_gib"] == 14.0


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        (
            {
                "available_physical_gib": 7.9,
                "commit_headroom_gib": 14.0,
                "committed_percent": 62.0,
                "gpu_memory_used_mib": 0,
                "gpu_memory_total_mib": 4096,
                "gpu_utilization_percent": 0,
                "competing_trainers": [],
            },
            "available physical memory below minimum",
        ),
        (
            {
                "available_physical_gib": 12.0,
                "commit_headroom_gib": 7.9,
                "committed_percent": 88.0,
                "gpu_memory_used_mib": 0,
                "gpu_memory_total_mib": 4096,
                "gpu_utilization_percent": 0,
                "competing_trainers": [],
            },
            "commit headroom below minimum",
        ),
        (
            {
                "available_physical_gib": 12.0,
                "commit_headroom_gib": 14.0,
                "committed_percent": 62.0,
                "gpu_memory_used_mib": 256,
                "gpu_memory_total_mib": 4096,
                "gpu_utilization_percent": 0,
                "competing_trainers": [],
            },
            "gpu memory is not free",
        ),
        (
            {
                "available_physical_gib": 12.0,
                "commit_headroom_gib": 14.0,
                "committed_percent": 62.0,
                "gpu_memory_used_mib": 0,
                "gpu_memory_total_mib": 4096,
                "gpu_utilization_percent": 0,
                "competing_trainers": [{"pid": 1234, "command": "train_v10r2_dev.py"}],
            },
            "competing trainer detected",
        ),
    ],
)
def test_host_resource_gate_holds_unsafe_launch(metrics: dict, reason: str) -> None:
    with pytest.raises(HostResourceHold, match=reason):
        validate_host_resources(
            metrics,
            min_available_physical_gib=8.0,
            min_commit_headroom_gib=8.0,
        )



class _RunResult:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_gpu_compute_process_scan_parses_rows(monkeypatch) -> None:
    import successor.experiments.preflight_v10r3r2_host_resources as host

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _RunResult("5888, python.exe, 3915\n1234, train.exe, 128\n")

    monkeypatch.setattr(host.subprocess, "run", fake_run)

    assert host.read_competing_trainers() == [
        {"pid": 5888, "process_name": "python.exe", "used_memory_mib": 3915},
        {"pid": 1234, "process_name": "train.exe", "used_memory_mib": 128},
    ]
    assert captured["args"] == [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    assert captured["kwargs"]["timeout"] == 10


def test_gpu_compute_process_scan_empty_is_empty(monkeypatch) -> None:
    import successor.experiments.preflight_v10r3r2_host_resources as host

    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *args, **kwargs: _RunResult(""),
    )
    assert host.read_competing_trainers() == []


def test_gpu_compute_process_scan_accepts_windows_na_memory(monkeypatch) -> None:
    import successor.experiments.preflight_v10r3r2_host_resources as host

    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *args, **kwargs: _RunResult(
            "5888, C:\\ProgramData\\ProRun\\python\\python.exe, [N/A]\n"
        ),
    )

    assert host.read_competing_trainers() == [
        {
            "pid": 5888,
            "process_name": "C:\\ProgramData\\ProRun\\python\\python.exe",
            "used_memory_mib": None,
        }
    ]


def test_preflight_main_returns_structured_hold_on_probe_failure(
    monkeypatch,
    capsys,
) -> None:
    import successor.experiments.preflight_v10r3r2_host_resources as host

    monkeypatch.setattr(
        host,
        "read_live_metrics",
        lambda: (_ for _ in ()).throw(
            host.HostResourceHold("probe failed cleanly")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "preflight_v10r3r2_host_resources.py",
            "--min-available-physical-gib",
            "8",
            "--min-commit-headroom-gib",
            "8",
        ],
    )

    assert host.main() == 2
    output = capsys.readouterr().out
    assert '"status": "HOLD"' in output
    assert "probe failed cleanly" in output
