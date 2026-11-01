import json
from contextlib import contextmanager

from successor.experiments import qualify_v10r3r3_stage0_runtime as stage0


def test_stage0_step_telemetry_appends_durable_jsonl(tmp_path):
    path = tmp_path / "stage0-step-telemetry.jsonl"
    first = {
        "step": 1,
        "step_time_seconds": 10.5,
        "gpu_temperature_c": 67,
        "commit_headroom_mib": 12000.0,
        "microbatches_completed": 8,
    }
    second = {
        "step": 2,
        "step_time_seconds": 11.0,
        "gpu_temperature_c": 68,
        "commit_headroom_mib": 11000.0,
        "microbatches_completed": 16,
    }

    stage0.append_stage0_step_telemetry(path, first)
    stage0.append_stage0_step_telemetry(path, second)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [first, second]


def test_stage0_optimizer_soak_forwards_telemetry_path(monkeypatch, tmp_path):
    observed = {}

    @contextmanager
    def fake_lock(path):
        # This proves wrapper argument forwarding, not Windows lock custody.
        observed["requested_lock_path"] = path
        yield {"synthetic_lock_for_source_test_only": True}

    monkeypatch.setattr(stage0, "exclusive_stage0_gpu_lock", fake_lock)

    def fake_unlocked(repo_root, **kwargs):
        observed["repo_root"] = repo_root
        observed.update(kwargs)
        return {"status": "SENTINEL"}

    monkeypatch.setattr(
        stage0,
        "_execute_stage0_optimizer_soak_unlocked",
        fake_unlocked,
    )
    # Fake lock isolates source forwarding from native Windows lock custody.
    telemetry_path = tmp_path / "trace.jsonl"
    result = stage0.execute_stage0_optimizer_soak(
        tmp_path,
        minimum_commit_headroom_mib=8192.0,
        commit_headroom_provider=lambda: 16384.0,
        gpu_lock_path=tmp_path / "gpu.lock",
        telemetry_path=telemetry_path,
    )

    assert result == {"status": "SENTINEL"}
    assert observed["requested_lock_path"] == tmp_path / "gpu.lock"
    assert observed["telemetry_path"] == telemetry_path
