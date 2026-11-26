import hashlib
import json
from pathlib import Path


CAPSULE = Path(
    "successor/evaluation/recursive_experience/"
    "VERA_RECURSIVE_EXPERIENCE_001_R8_AUTHORITY_RUNTIME_20261007_V1.json"
)


def test_recursive_experience_001_is_unadmitted_and_self_consistent():
    value = json.loads(CAPSULE.read_text(encoding="utf-8"))

    assert value["schema"] == "VERA_RECURSIVE_EXPERIENCE_CAPSULE_V1"
    assert value["status"] == "CANDIDATE_UNADMITTED"
    assert value["teacher"]["model"] == "GPT-5.6 Sol"
    assert value["teacher"]["reasoning_effort"] == "High"
    assert value["teacher"]["same_model_recursive_loop"] is True
    assert value["teacher"]["independent_review"] is False

    trials = {row["mode"]: row for row in value["student_trials"]}
    assert trials["NO_SYSTEM"]["teacher_score"] == 4
    assert trials["NO_SYSTEM"]["cap_applied"] is True
    assert trials["MINIMAL_VERA_SYSTEM"]["teacher_score"] == 7
    assert trials["MINIMAL_VERA_SYSTEM"]["cap_applied"] is False

    assert value["target"]["teacher_expected_score"] == 10
    assert value["admission"]["automatic_training_use"] == "PROHIBITED"
    assert value["admission"]["independent_review_required"] is True
    assert value["admission"]["ground_truth_reverification_required_before_admission"] is True
    assert value["admission"]["status"] == "PENDING_INDEPENDENT_REVIEW"

    unsigned = dict(value)
    claimed = unsigned.pop("capsule_sha256")
    raw = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == claimed


def test_recursive_experience_001_binds_exact_r8_evidence():
    value = json.loads(CAPSULE.read_text(encoding="utf-8"))
    subject = value["subject"]

    assert subject["r8_result_sha256"] == (
        "f5375f17163ff27525dd21a7bda8a4d149bce040c4e27e54c1e99c8c0e5e17f5"
    )
    assert subject["r8_execution_spec_sha256"] == (
        "7f64429473bd6a23aa0b86117a4402939fe40db8ac7a63a896968ebb1b9e4005"
    )
    assert subject["adapter_model_sha256"] == (
        "debe3bf4a806858eeece821a7eb49be8f9daada109d78c5c10cf5a952baf1092"
    )

    truth = value["task"]["objective_ground_truth"]
    assert truth["r8_optimizer_steps"] == 20
    assert truth["r8_status"] == "DEVELOPMENT_TRAINING_PASS"
    assert truth["r8_weight_digest_changed"] is True
    assert truth["r8_qualification_status"] == "NOT_EXTERNALLY_EVALUATED"
    assert truth["deploy_activate_authorized"] is False
    assert truth["route_selection_evidence_supplied"] is False
    assert truth["runtime_consumption_evidence_supplied"] is False
