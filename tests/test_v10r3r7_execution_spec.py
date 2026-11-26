import json
from pathlib import Path

from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec


def test_r7_execution_spec_is_exact_continuous20_fresh_adapter():
    path = Path(
        "successor/experiments/"
        "V10R3R7_CONTINUOUS20_EXECUTION_SPEC_20261007_V1.json"
    )
    spec = load_and_validate_dev_spec(path)

    assert spec["status"] == "AUTHORIZED_LOCAL_DEVELOPMENT_TRAINING"
    assert spec["experiment_class"] == "CONTINUOUS_STATE_RECIPE_SEMANTICS"
    assert spec["development_window"] == {"start_row": 0, "row_count": 160}
    assert spec.get("resume_adapter") is None

    subject = spec["source_subject"]
    assert subject["train_sha256"] == (
        "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
    )
    assert subject["train_rows"] == 50000

    trainer = spec["trainer"]
    assert trainer["max_optimizer_steps"] == 20
    assert trainer["optimizer"] == "adamw_bnb_8bit"
    assert trainer["gradient_accumulation_steps"] == 8
    assert trainer["shuffle_dataset"] is False
    assert trainer["train_sampling_strategy"] == "sequential"

    assert spec["output"]["namespace"] == (
        r"D:\VERA\models\adapters\v10r3r7-lane-a-continuous20-sequential-20261007"
    )
    assert spec["output"]["overwrite_forbidden"] is True

    receipt = json.loads(
        Path(
            "successor/experiments/receipts/"
            "V10R3R7_CORPUS_REBIND_20261007_V1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "EXACT_CORPUS_REBOUND"
    assert receipt["train_sha256"] == subject["train_sha256"]
    assert receipt["train_rows"] == subject["train_rows"]
