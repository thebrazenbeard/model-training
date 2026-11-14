import json
from pathlib import Path


def test_r5_v2_uses_user_selected_32_gib_prelaunch_gate():
    path = Path(
        "successor/experiments/"
        "V10R3R5_STAGE0_SUCCESSOR_PROTOCOL_20261006_V2.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["schema"] == "V10R3R5_STAGE0_SUCCESSOR_PROTOCOL_V2"
    assert protocol["authority"]["selected_prelaunch_commit_headroom_gib"] == 32.0
    assert protocol["authority"]["supersedes_threshold_from_v1_gib"] == 40.0

    gate = protocol["resource_gate"]
    assert gate["min_prelaunch_commit_headroom_gib"] == 32.0
    assert gate["in_run_min_commit_headroom_mib"] == 8192.0
    assert gate["same_pressure_minimum_prelaunch_gib"] == 33.043
    assert gate["selected_gate_delta_vs_same_pressure_minimum_gib"] == -1.043
    assert gate["selected_gate_below_same_pressure_estimate"] is True
    assert gate["threshold_is_completion_guarantee"] is False

    execution = protocol["execution"]
    assert execution["attempt_limit"] == 1
    assert execution["attempt_status"] == "UNUSED"
    assert execution["automatic_retry"] is False
    assert execution["optimizer_steps"] == 20
    assert execution["synthetic_input_only"] is True
    assert execution["corpus_bearing_input_prohibited"] is True
    assert execution["telemetry_path"].endswith(".jsonl")

    assert protocol["promotion"]["preflight_fail"] == "HOLD_NO_ATTEMPT_CONSUMED"
    assert protocol["promotion"]["stage0_pass"] == (
        "PREPARE_SEPARATE_FRESH_CORPUS_TRAINING_SUBJECT"
    )
