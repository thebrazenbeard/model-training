import json
from pathlib import Path


def test_r6_v2_uses_user_selected_25_gib_prelaunch_gate():
    path = Path(
        "successor/experiments/"
        "V10R3R6_STAGE0_SUCCESSOR_PROTOCOL_20261007_V2.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["schema"] == "V10R3R6_STAGE0_SUCCESSOR_PROTOCOL_V2"
    assert protocol["authority"]["selected_prelaunch_commit_headroom_gib"] == 25.0
    assert protocol["authority"]["supersedes_threshold_from_v1_gib"] == 32.0

    gate = protocol["resource_gate"]
    assert gate["min_prelaunch_commit_headroom_gib"] == 25.0
    assert gate["in_run_min_commit_headroom_mib"] == 8192.0
    assert gate["min_available_physical_gib"] == 8.0
    assert gate["gpu_utilization_percent_required"] == 0
    assert gate["wddm_reported_memory_used_is_informational"] is True

    execution = protocol["execution"]
    assert execution["attempt_limit"] == 1
    assert execution["attempt_status"] == "UNUSED"
    assert execution["automatic_retry"] is False
    assert execution["optimizer_steps"] == 20
    assert execution["max_wall_seconds"] == 1200.0
    assert execution["synthetic_input_only"] is True
    assert execution["corpus_bearing_input_prohibited"] is True
