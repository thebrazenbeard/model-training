import hashlib
import json
from pathlib import Path

from successor.experiments import run_v10r3r7_development as runner
from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec


def test_r8_successor_keeps_training_math_and_repairs_invocation_only(tmp_path):
    protocol = json.loads(
        Path(
            "successor/experiments/"
            "V10R3R8_CONTINUOUS20_DEVELOPMENT_PROTOCOL_20261007_V1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["mechanism_change"]["training_math_change"] == "NONE"
    assert protocol["mechanism_change"]["invocation_change"] == (
        "SCRIPT_PATH_TO_PACKAGE_MODULE"
    )
    assert protocol["resource_gate"]["min_prelaunch_commit_headroom_gib"] == 25.0
    assert protocol["execution"]["attempt_status"] == "UNUSED"
    assert protocol["execution"]["automatic_retry"] is False

    spec = load_and_validate_dev_spec(
        Path(
            "successor/experiments/"
            "V10R3R8_CONTINUOUS20_EXECUTION_SPEC_20261007_V1.json"
        )
    )
    assert spec["development_window"] == {"start_row": 0, "row_count": 160}
    assert spec["trainer"]["max_optimizer_steps"] == 20
    assert spec["trainer"]["optimizer"] == "adamw_bnb_8bit"
    assert spec.get("resume_adapter") is None
    assert "v10r3r8" in spec["output"]["namespace"].lower()

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


def test_r8_watchdog_accepts_r8_fresh_namespace():
    root = Path(".").resolve()
    spec_path = root / (
        "successor/experiments/"
        "V10R3R8_CONTINUOUS20_EXECUTION_SPEC_20261007_V1.json"
    )
    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    info = runner._validate_spec(
        root,
        spec_path,
        expected_spec_sha256=spec_sha,
        expected_train_sha256=(
            "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
        ),
        expected_train_rows=50000,
    )

    assert "v10r3r8" in info["output_namespace"].lower()
