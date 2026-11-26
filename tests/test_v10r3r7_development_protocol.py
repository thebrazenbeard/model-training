import json
from pathlib import Path


def test_r7_real_corpus_subject_is_fresh_and_waits_only_for_corpus_rebind():
    path = Path(
        "successor/experiments/"
        "V10R3R7_CONTINUOUS20_DEVELOPMENT_PROTOCOL_20261007_V1.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["schema"] == "V10R3R7_CONTINUOUS20_DEVELOPMENT_PROTOCOL_V1"
    assert protocol["status"] == "PREPARED_R6_PASS_BOUND_PENDING_CORPUS_REBIND"

    predecessor = protocol["predecessor_gate"]
    assert predecessor["r6_pass_not_yet_bound"] is False
    assert predecessor["r6_evidence_head"] == (
        "21a72bf0f834b1e2555393e82bc74440df2e29ef"
    )
    assert predecessor["r6_receipt_sha256"] == (
        "2b30e33d27415139768508bf59866489c1a69b1a5bb06cc0562be9113f90bb4d"
    )

    assert protocol["authority"]["actor_id"] == "PATRICK_USER_AUTHORITY"
    assert protocol["authority"]["source"] == "EXPLICIT_CURRENT_USER_INSTRUCTION"
    assert protocol["resource_gate"]["min_prelaunch_commit_headroom_gib"] == 25.0
    assert protocol["resource_gate"]["min_available_physical_gib"] == 8.0
    assert protocol["resource_gate"]["single_gpu_mutating_lane"] == "lane-a"

    subject = protocol["training_subject"]
    assert subject["train_sha256"] == (
        "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
    )
    assert subject["train_rows"] == 50000
    assert subject["window"] == {"start_row": 0, "row_count": 160}
    assert subject["optimizer_steps"] == 20
    assert subject["optimizer"] == "adamw_bnb_8bit"
    assert subject["fresh_adapter"] is True
    assert "v10r3r7" in subject["output_namespace"].lower()
    assert subject["legacy_h07_parent_used"] is False

    assert protocol["execution"]["automatic_retry"] is False
    assert protocol["execution"]["attempt_limit"] == 1
    assert protocol["execution"]["executable_now"] is False
