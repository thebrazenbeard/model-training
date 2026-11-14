from __future__ import annotations

import json
from pathlib import Path

from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "successor" / "experiments"
R1 = EXP / "V10R3R1_CONTINUOUS20_EXECUTION_SPEC_20261004_V1.json"
R2 = EXP / "V10R3R2_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261004_V1.json"


def test_r2_preserves_r1_training_recipe_projection() -> None:
    r1 = json.loads(R1.read_text(encoding="utf-8"))
    r2 = json.loads(R2.read_text(encoding="utf-8"))

    for key in (
        "source_subject",
        "trainer",
        "quantization",
        "lora",
        "model_load",
        "qualification",
        "experiment_class",
        "development_window",
    ):
        assert r2[key] == r1[key]

    for key in (
        "exact_order",
        "full_matched_order_record_ids_sha256",
        "order_manifest_path",
        "order_manifest_sha256",
        "expected_initial_trainable_parameter_digest",
        "immutable_staged_control_receipt_sha256",
        "immutable_staged_control_adapter_sha256",
        "evaluation_panel_use",
    ):
        assert r2["comparison"][key] == r1["comparison"][key]

    assert r2["output"]["save_final_adapter"] == r1["output"]["save_final_adapter"]
    assert r2["output"]["overwrite_forbidden"] == r1["output"]["overwrite_forbidden"]
    assert r2["output"]["namespace"] != r1["output"]["namespace"]

    assert r2["comparison"]["r1_attempt_reuse"] is False
    assert r2["comparison"]["r1_status"] == "FAILED_INCOMPLETE_ONE_ATTEMPT_EXHAUSTED"
    assert r2["comparison"]["r1_last_completed_optimizer_step_observed"] == 13


def test_r2_spec_is_executable_under_current_training_authority() -> None:
    spec = load_and_validate_dev_spec(R2)

    assert spec["status"] == "AUTHORIZED_LOCAL_DEVELOPMENT_TRAINING"
    assert spec["trainer"]["max_optimizer_steps"] == 20
    assert spec["development_window"] == {"start_row": 0, "row_count": 160}
    assert spec["output"]["namespace"].endswith(
        "v10r3r2-lane-a-continuous20-sequential-20261004"
    )
