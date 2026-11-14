from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import successor.experiments.run_durable_process_r3 as durable_module
from successor.experiments.run_durable_process_r3 import (
    DurableProcessHold,
    run_durable_process,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_REL = (
    "successor/experiments/"
    "V10R3R3_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261005_V1.json"
)
RUNTIME_REL = (
    "successor/experiments/"
    "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
)


def _lf_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _metadata() -> dict[str, str]:
    spec = json.loads((ROOT / SPEC_REL).read_text(encoding="utf-8"))
    runtime = json.loads((ROOT / RUNTIME_REL).read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {
        "repo_head": head,
        "spec_path": SPEC_REL,
        "spec_sha256": _lf_sha(ROOT / SPEC_REL),
        "runtime_binding_sha256": spec["source_subject"]["runtime_binding_sha256"],
        "runtime_python_path": runtime["target"]["python_path"],
    }


def _args(log_dir: Path, metadata: dict[str, str]) -> list[str]:
    runtime_python = metadata.get("runtime_python_path", sys.executable)
    return [
        "--log-dir", str(log_dir),
        "--watch-seconds", "0.04",
        "--expected-optimizer-steps", "20",
        "--metadata-json", json.dumps(metadata),
        "--cwd", str(ROOT),
        "--",
        runtime_python,
        "-c", "print('SHOULD_NOT_RUN')",
        "--spec", SPEC_REL,
    ]


def test_r3_requires_runtime_python_metadata(tmp_path: Path) -> None:
    metadata = _metadata()
    del metadata["runtime_python_path"]
    log_dir = tmp_path / "missing-runtime-python"

    with pytest.raises(DurableProcessHold, match="runtime_python_path"):
        run_durable_process(_args(log_dir, metadata))
    assert not log_dir.exists()


def test_r3_rejects_wrong_wrapper_runtime(tmp_path: Path) -> None:
    metadata = _metadata()
    log_dir = tmp_path / "wrong-wrapper-runtime"

    with pytest.raises(DurableProcessHold, match="runtime_python_path wrapper mismatch"):
        run_durable_process(_args(log_dir, metadata))
    assert not log_dir.exists()


def test_r3_rejects_unavailable_cuda_before_log_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata()
    runtime_python = metadata["runtime_python_path"]
    monkeypatch.setattr(durable_module.sys, "executable", runtime_python)
    monkeypatch.setattr(
        durable_module,
        "_runtime_cuda_observation",
        lambda _: {
            "executable": runtime_python,
            "python": "3.12.10",
            "torch": "2.14.0+cu130",
            "torch_cuda": "13.0",
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
        },
    )
    log_dir = tmp_path / "cuda-unavailable"

    with pytest.raises(DurableProcessHold, match="runtime CUDA unavailable"):
        run_durable_process(_args(log_dir, metadata))
    assert not log_dir.exists()


def test_r3_rejects_runtime_version_mismatch_before_log_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata()
    runtime_python = metadata["runtime_python_path"]
    monkeypatch.setattr(durable_module.sys, "executable", runtime_python)
    monkeypatch.setattr(
        durable_module,
        "_runtime_cuda_observation",
        lambda _: {
            "executable": runtime_python,
            "python": "3.12.10",
            "torch": "WRONG",
            "torch_cuda": "13.0",
            "cuda_available": True,
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        },
    )
    log_dir = tmp_path / "runtime-version-mismatch"

    with pytest.raises(DurableProcessHold, match="runtime torch mismatch"):
        run_durable_process(_args(log_dir, metadata))
    assert not log_dir.exists()
