import json
from pathlib import Path


def test_r6_successor_protocol_freezes_walltime_only_successor():
    path = Path(
        "successor/experiments/"
        "V10R3R6_STAGE0_SUCCESSOR_PROTOCOL_20261007_V1.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["schema"] == "V10R3R6_STAGE0_SUCCESSOR_PROTOCOL_V1"
    assert protocol["status"] == "AUTHORIZED_LOCAL_DEVELOPMENT_STAGE0_POLICY_GATED"

    predecessor = protocol["predecessor"]
    assert predecessor["r5_evidence_head"] == (
        "bd83999f9bec0520184c7467143222b90f2b3839"
    )
    assert predecessor["r5_status"] == "FAILED_INCOMPLETE_ONE_ATTEMPT_EXHAUSTED"
    assert predecessor["r5_optimizer_steps_completed"] == 17
    assert predecessor["r5_failure_classification"] == "WALL_TIME_LIMIT_ONLY"

    hypothesis = protocol["hypothesis"]
    assert hypothesis["training_math_change"] == "NONE"
    assert hypothesis["stage0_mechanism_change"] == "WALL_BUDGET_ONLY"

    gate = protocol["resource_gate"]
    assert gate["min_available_physical_gib"] == 8.0
    assert gate["min_prelaunch_commit_headroom_gib"] == 32.0
    assert gate["in_run_min_commit_headroom_mib"] == 8192.0
    assert gate["competing_trainers_required"] == 0
    assert gate["gpu_utilization_percent_required"] == 0
    assert gate["wddm_reported_memory_used_is_informational"] is True

    execution = protocol["execution"]
    assert execution["optimizer_steps"] == 20
    assert execution["max_wall_seconds"] == 1200.0
    assert execution["gpu_temperature_abort_c"] == 88
    assert execution["attempt_limit"] == 1
    assert execution["attempt_status"] == "UNUSED"
    assert execution["automatic_retry"] is False
    assert execution["synthetic_input_only"] is True
    assert execution["corpus_bearing_input_prohibited"] is True

    timing = protocol["r5_timing_evidence"]
    assert timing["slowest_step_seconds"] == 52.219
    assert timing["projected_20_step_elapsed_at_slowest_observed_seconds"] == 1099.761
    assert timing["r6_wall_margin_seconds"] == 100.239

    assert protocol["promotion"]["preflight_fail"] == "HOLD_NO_ATTEMPT_CONSUMED"
    assert protocol["promotion"]["stage0_pass"] == (
        "PREPARE_SEPARATE_FRESH_CORPUS_TRAINING_SUBJECT"
    )
