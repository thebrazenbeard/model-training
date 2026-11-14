import hashlib
from pathlib import Path

import pytest


def test_r7_runner_resource_gate_ignores_wddm_memory_but_enforces_safety():
    from successor.experiments import run_v10r3r7_development as runner

    ok = runner.evaluate_resource_sample(
        available_physical_gib=12.0,
        commit_headroom_gib=26.0,
        gpu_utilization_percent=0,
        gpu_memory_used_mib=3900,
        compute_apps=[],
        gpu_temperature_c=70,
        prelaunch=True,
    )
    assert ok["status"] == "PASS"

    low_commit = runner.evaluate_resource_sample(
        available_physical_gib=12.0,
        commit_headroom_gib=7.9,
        gpu_utilization_percent=100,
        gpu_memory_used_mib=4000,
        compute_apps=["trainer"],
        gpu_temperature_c=70,
        prelaunch=False,
    )
    assert low_commit["status"] == "HOLD"
    assert "commit" in low_commit["reason"].lower()

    hot = runner.evaluate_resource_sample(
        available_physical_gib=12.0,
        commit_headroom_gib=20.0,
        gpu_utilization_percent=100,
        gpu_memory_used_mib=4000,
        compute_apps=["trainer"],
        gpu_temperature_c=88,
        prelaunch=False,
    )
    assert hot["status"] == "HOLD"
    assert "temperature" in hot["reason"].lower()


def test_r7_runner_verifies_exact_corpus_bytes_and_rows(tmp_path):
    from successor.experiments import run_v10r3r7_development as runner

    path = tmp_path / "train.jsonl"
    payload = b'{"x":1}\n{"x":2}\n'
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    receipt = runner.verify_corpus_file(path, expected_sha256=digest, expected_rows=2)
    assert receipt["status"] == "EXACT_CORPUS_VERIFIED"

    with pytest.raises(runner.R7Hold, match="sha256"):
        runner.verify_corpus_file(path, expected_sha256="0" * 64, expected_rows=2)


def test_r7_runner_requires_fresh_output_namespace(tmp_path):
    from successor.experiments import run_v10r3r7_development as runner

    fresh = tmp_path / "fresh"
    runner.require_fresh_output_namespace(fresh)
    fresh.mkdir()
    with pytest.raises(runner.R7Hold, match="already exists"):
        runner.require_fresh_output_namespace(fresh)


def test_r8_runner_uses_module_safe_child_invocation(tmp_path):
    from successor.experiments import run_v10r3r7_development as runner

    command = runner.build_training_command(
        repo_root=tmp_path,
        train_jsonl=tmp_path / "train.jsonl",
        spec_path=tmp_path / "spec.json",
    )

    assert command[:3] == [
        runner.sys.executable,
        "-m",
        "successor.experiments.train_v10r2_dev",
    ]
    assert str(tmp_path) in command
