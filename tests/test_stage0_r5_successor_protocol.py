import json
from pathlib import Path


def test_r5_successor_protocol_freezes_host_pressure_hypothesis():
    path = Path(
        "successor/experiments/"
        "V10R3R5_STAGE0_SUCCESSOR_PROTOCOL_20261006_V1.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["schema"] == "V10R3R5_STAGE0_SUCCESSOR_PROTOCOL_V1"
    assert protocol["status"] == "AUTHORIZED_LOCAL_DEVELOPMENT_STAGE0_POLICY_GATED"
    assert protocol["predecessor"]["r4_evidence_head"] == (
        "4f2aa26f50ce723bf32774238f6fb55b531dc5ad"
    )
    assert protocol["hypothesis"]["leading"] == "HOST_COMMIT_PRESSURE"
    assert protocol["hypothesis"]["training_math_change"] == "NONE"
    assert protocol["hypothesis"]["stage0_mechanism_change"] == "TELEMETRY_ONLY"

    gate = protocol["resource_gate"]
    assert gate["min_prelaunch_commit_headroom_gib"] == 40.0
    assert gate["in_run_min_commit_headroom_mib"] == 8192.0
    assert gate["min_available_physical_gib"] == 8.0
    assert gate["r4_observed_commit_pressure_gib_through_step5"] == 25.043
    assert gate["same_pressure_minimum_prelaunch_gib"] == 33.043

    execution = protocol["execution"]
    assert execution["attempt_limit"] == 1
    assert execution["attempt_status"] == "UNUSED"
    assert execution["automatic_retry"] is False
    assert execution["synthetic_input_only"] is True
    assert execution["corpus_bearing_input_prohibited"] is True
    assert execution["optimizer_steps"] == 20
    assert execution["runtime_python_path"] == (
        r"C:\ProgramData\ProRun\model-env\Scripts\python.exe"
    )
    assert execution["telemetry_path"].endswith(".jsonl")

    assert protocol["promotion"]["preflight_fail"] == "HOLD_NO_ATTEMPT_CONSUMED"
    assert protocol["promotion"]["stage0_pass"] == (
        "PREPARE_SEPARATE_FRESH_CORPUS_TRAINING_SUBJECT"
    )
