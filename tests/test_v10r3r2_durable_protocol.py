from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "successor" / "experiments"
PROTOCOL = EXP / "V10R3R2_DURABLE_CONTINUOUS20_PROTOCOL_20261004_V1.json"
SPEC_REL = "successor/experiments/V10R3R2_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261004_V1.json"
HARNESS_REL = "successor/experiments/run_durable_process.py"
HOST_GUARD_REL = "successor/experiments/preflight_v10r3r2_host_resources.py"
INCIDENT = EXP / "receipts" / "V10R3R1_EXECUTION_INCIDENT_20261004_V1.json"


def _committed_sha(relative: str) -> str:
    payload = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])
    return hashlib.sha256(payload).hexdigest()


def test_r2_protocol_binds_exact_recipe_subject_and_harness() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["source_subject"]["recipe_source_head"] == (
        "4197205466f3213c4d3e3c0e3e7557e4569bc276"
    )
    assert protocol["source_subject"]["execution_spec_committed_sha256"] == _committed_sha(
        SPEC_REL
    )
    assert protocol["source_subject"]["durable_harness_committed_sha256"] == _committed_sha(
        HARNESS_REL
    )
    assert protocol["source_subject"]["host_resource_guard_path"] == HOST_GUARD_REL
    assert protocol["source_subject"]["host_resource_guard_committed_sha256"] == _committed_sha(
        HOST_GUARD_REL
    )
    assert protocol["source_subject"]["launch_guard_source_head"] == (
        "1304f1ce34a11fcd3e5752d7f4a09eb95d66a145"
    )
    assert protocol["source_subject"]["observability_source_head"] == (
        "d64dac5c5b7cc1fb2997c272e432ef61682b10e6"
    )
    assert protocol["source_subject"]["host_guard_source_head"] == (
        "1304f1ce34a11fcd3e5752d7f4a09eb95d66a145"
    )
    assert protocol["source_subject"]["hash_semantics"] == (
        "UTF8_TEXT_LF_NORMALIZED_COMMITTED_CONTENT"
    )


def test_r2_protocol_binds_r1_incident_and_forbids_retry() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    incident_bytes = INCIDENT.read_bytes().replace(b"\r\n", b"\n")

    assert protocol["predecessor_incident"]["lf_normalized_sha256"] == hashlib.sha256(
        incident_bytes
    ).hexdigest()
    assert protocol["predecessor_incident"]["r1_attempt_limit_exhausted"] is True
    assert protocol["predecessor_incident"]["r1_retry_prohibited"] is True
    assert protocol["execution"]["attempt_limit"] == 1
    assert protocol["execution"]["automatic_retry"] is False
    assert protocol["execution"]["panel_use"] == "PROHIBITED"
    assert protocol["observability"]["watchdog_latest_optimizer_step_required"] is True
    assert protocol["observability"]["launch_metadata_required_fields"] == [
        "repo_head",
        "spec_path",
        "spec_sha256",
        "runtime_binding_sha256",
    ]
    assert protocol["execution"]["host_resource_gate"] == {
        "required": True,
        "min_available_physical_gib": 8.0,
        "min_commit_headroom_gib": 8.0,
        "gpu_memory_used_mib_required": 0,
        "competing_trainer_count_required": 0,
    }
