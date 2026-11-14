import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "successor"
    / "experiments"
    / "V10R3R4_STAGE0_SUCCESSOR_PROTOCOL_20261006_V2.json"
)


def _load():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_r4_v2_binds_newer_preactive_window_policy_and_keeps_attempt_unused():
    protocol = _load()
    assert protocol["schema"] == "V10R3R4_STAGE0_SUCCESSOR_PROTOCOL_V2"
    assert protocol["status"] == "AUTHORIZED_LOCAL_DEVELOPMENT_STAGE0_POLICY_GATED"
    assert protocol["claim_ceiling"] == (
        "STAGE0_RUNTIME_QUALIFICATION_ONLY_NOT_CORPUS_TRAINING_NOT_DEPLOYMENT"
    )

    supersedes = protocol["supersedes"]
    assert supersedes["path"].endswith(
        "V10R3R4_STAGE0_SUCCESSOR_PROTOCOL_20261006_V1.json"
    )
    assert supersedes["frozen_head"] == "d62a055ab0c808a62419ca7195a78d93e21b4f14"
    assert supersedes["reason"] == "NEWER_PREACTIVE_AUTONOMY_WINDOW_INVALIDATED_WATCHDOG_HANDOFF"

    execution = protocol["execution"]
    assert execution["attempt_limit"] == 1
    assert execution["attempt_status"] == "UNUSED"
    assert execution["automatic_retry"] is False
    assert execution["synthetic_input_only"] is True
    assert execution["corpus_bearing_input_prohibited"] is True

    policy = protocol["preactive_window_policy"]
    assert policy["policy_sha256"] == (
        "28c7a9e56b3ee785445917cf1b73c32f10aabfd6bd73330fbabd0ed397791fe4"
    )
    assert policy["apply_policy_sha256"] == (
        "94027fdf9ad0635aba9302928e565c6590530c860ea322ac922dfcdedb605fa2"
    )
    assert policy["supervisor_sha256"] == (
        "0bafb43e77a151a46854422bafc44d9d6585a9240f9011ea0824a042b48609c4"
    )
    assert policy["close_sha256"] == (
        "a3f2102567f447048c78fd33e609fd5bbef7b8371b12da5404fcf134f25de75f"
    )
    assert policy["authority"] == "Patrick present instruction 2026-10-06"
    assert policy["start_local"] == "00:00"
    assert policy["enforcement_cutoff_local"] == "13:59"
    assert policy["outside_window"] == "STOP_PRE_ACTIVE_DAEMON_AND_QWEN"

    arbitration = protocol["resource_arbitration"]
    assert arbitration["current_gate"] == "HOLD_INSIDE_NEWER_AUTONOMY_WINDOW"
    assert arbitration["preemption_rule"] == (
        "NO_PREEMPT_INSIDE_ACTIVE_WINDOW_WITHOUT_NEWER_EXPLICIT_PATRICK_OVERRIDE"
    )
    assert arbitration["supervisor_task"] == "PreActive Window Supervisor"
    assert arbitration["window_close_task"] == "PreActive Window Close"
    assert arbitration["qwen_task"] == "PreActive Qwen Endpoint"
    assert arbitration["daemon_task"] == "PreActive Daemon"
    assert arbitration["stop_order_if_newer_override"] == [
        "PreActive Window Supervisor",
        "PreActive Qwen Endpoint",
    ]
    assert arbitration["restore_order_if_newer_override"] == [
        "PreActive Qwen Endpoint",
        "PreActive Window Supervisor",
    ]
