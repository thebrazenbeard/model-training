from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "successor" / "experiments"
PROTOCOL = EXP / "V10R3R3_DURABLE_CONTINUOUS20_PROTOCOL_20261005_V1.json"
SPEC = EXP / "V10R3R3_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261005_V1.json"
HARNESS = EXP / "run_durable_process_r3.py"
RUNTIME = EXP / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
INCIDENT = EXP / "receipts" / "V10R3R2_EXECUTION_INCIDENT_20261005_V1.json"

RUNTIME_SHA = "44b01efc6c2d0e1716208e04e7132433561b5562430adae821fd29e6a1493224"
RUNTIME_PYTHON = r"C:\ProgramData\ProRun\model-env\Scripts\python.exe"


def _lf_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_r3_protocol_binds_dedicated_harness_spec_runtime_and_incident() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = protocol["source_subject"]
    assert source["execution_spec_committed_sha256"] == _lf_sha(SPEC)
    assert source["durable_harness_path"] == "successor/experiments/run_durable_process_r3.py"
    assert source["durable_harness_committed_sha256"] == _lf_sha(HARNESS)
    assert source["runtime_binding_sha256"] == RUNTIME_SHA
    assert source["runtime_python_path"] == RUNTIME_PYTHON
    assert protocol["predecessor_incident"]["lf_normalized_sha256"] == _lf_sha(INCIDENT)
    assert protocol["predecessor_incident"]["r2_attempt_limit_exhausted"] is True
    assert protocol["predecessor_incident"]["r2_retry_prohibited"] is True


def test_r3_runtime_binding_matches_frozen_target() -> None:
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    assert runtime["binding_sha256"] == RUNTIME_SHA
    assert runtime["target"]["python_path"] == RUNTIME_PYTHON
    assert runtime["target"]["python_version"] == "3.12.10"
    assert runtime["packages"]["torch"] == "2.14.0+cu130"
    assert runtime["target"]["cuda_runtime"] == "13.0"
    assert runtime["target"]["gpu"] == "NVIDIA GeForce RTX 3050 Laptop GPU"


def test_r3_training_math_is_unchanged_and_namespaces_are_fresh() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    trainer = spec["trainer"]
    assert trainer["max_optimizer_steps"] == 20
    assert trainer["per_device_train_batch_size"] == 1
    assert trainer["gradient_accumulation_steps"] == 8
    assert trainer["max_length"] == 512
    assert trainer["optimizer"] == "adamw_bnb_8bit"
    assert trainer["learning_rate"] == 2e-5
    assert trainer["lr_scheduler_type"] == "cosine"
    assert trainer["warmup_optimizer_steps"] == 0
    assert trainer["shuffle_dataset"] is False
    assert trainer["train_sampling_strategy"] == "sequential"
    assert trainer["gradient_checkpointing"] is True
    assert trainer["gradient_checkpointing_use_reentrant"] is False
    assert spec["development_window"] == {"start_row": 0, "row_count": 160}
    assert spec["source_subject"]["train_sha256"] == "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
    assert spec["source_subject"]["train_rows"] == 50000
    assert spec["source_subject"]["runtime_python_path"] == RUNTIME_PYTHON
    assert spec["output"]["namespace"] == r"D:\VERA\models\adapters\v10r3r3-lane-a-continuous20-sequential-20261005"
    assert spec["comparison"]["r2_attempt_reuse"] is False


def test_r3_protocol_is_single_attempt_and_requires_runtime_python_metadata() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["execution"]["attempt_limit"] == 1
    assert protocol["execution"]["automatic_retry"] is False
    assert protocol["execution"]["fresh_process_required"] is True
    assert protocol["execution"]["fresh_adapter_required"] is True
    assert protocol["execution"]["resume_adapter_forbidden"] is True
    assert protocol["execution"]["panel_use"] == "PROHIBITED"
    assert protocol["observability"]["log_namespace"] == r"D:\VERA\logs\training\v10r3r3-cont20-20261005-v1"
    assert protocol["observability"]["launch_metadata_required_fields"] == [
        "repo_head", "spec_path", "spec_sha256", "runtime_binding_sha256", "runtime_python_path"
    ]
    assert protocol["recipe_semantics"]["training_math_change_from_r2"] == "NONE"
