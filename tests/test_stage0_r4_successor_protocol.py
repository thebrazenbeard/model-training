import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "successor" / "experiments" / "V10R3R4_STAGE0_SUCCESSOR_PROTOCOL_20261006_V1.json"


def _load():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_r4_stage0_is_distinct_fail_closed_successor():
    protocol = _load()
    assert protocol["schema"] == "V10R3R4_STAGE0_SUCCESSOR_PROTOCOL_V1"
    assert protocol["status"] == "AUTHORIZED_LOCAL_DEVELOPMENT_STAGE0"
    assert protocol["claim_ceiling"] == "STAGE0_RUNTIME_QUALIFICATION_ONLY_NOT_CORPUS_TRAINING_NOT_DEPLOYMENT"

    predecessor = protocol["predecessor"]
    assert predecessor["subject"] == "lane-a/stage0-runtime-qualification-v6"
    assert predecessor["head"] == "8fcb8428a9b6ef20d0052866aac40ea45d1385ce"
    assert predecessor["status"] == "FAILED_INCOMPLETE_ONE_ATTEMPT_EXHAUSTED"
    assert predecessor["retry_in_place_forbidden"] is True

    execution = protocol["execution"]
    assert execution["attempt_limit"] == 1
    assert execution["automatic_retry"] is False
    assert execution["synthetic_input_only"] is True
    assert execution["corpus_bearing_input_prohibited"] is True
    assert execution["module"] == "successor.experiments.qualify_v10r3r3_stage0_runtime"

    runtime = protocol["runtime"]
    assert runtime["python_path"] == r"C:\ProgramData\ProRun\model-env\Scripts\python.exe"
    assert runtime["cuda_required"] is True
    assert runtime["min_commit_headroom_gib"] == 8
    assert runtime["gpu_memory_used_mib_required"] == 0

    handoff = protocol["preactive_handoff"]
    assert handoff["watchdog_task"] == "PreActive Watchdog"
    assert handoff["qwen_task"] == "PreActive Qwen Endpoint"
    assert handoff["daemon_task"] == "PreActive Daemon"
    assert handoff["daemon_stop_required"] is False
    assert handoff["restore_required_after_terminal_state"] is True
    assert handoff["stop_order"] == ["PreActive Watchdog", "PreActive Qwen Endpoint"]
    assert handoff["restore_order"] == ["PreActive Qwen Endpoint", "PreActive Watchdog"]
