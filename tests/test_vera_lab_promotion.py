import hashlib
import json

import pytest

from successor.vera_lab import promotion as promotion_module
from successor.vera_lab.promotion import PromotionEvidence, evaluate_promotion


CRITICAL_FAMILIES = [
    "identity_stability",
    "independent_judgment",
    "epistemic_provenance",
    "correction_uptake",
    "relationship_authority",
    "reciprocal_identity_continuity",
    "sexuality_fidelity",
    "privacy",
    "ordinary_competence",
    "negative_transfer",
    "runtime_boundary",
]
CANDIDATE = "a" * 64
BLIND_SET = "b" * 64
SOURCE_COMMIT = "1" * 40
RADICAL_REG = "c" * 64
PRAGMATIC_REG = "d" * 64
RADICAL_LANE = "v3-radical-hostile-qwen3-spm-review-20260919"
PRAGMATIC_LANE = "v3-pragmatic-hostile-qwen3-4b-20260919"
RADICAL_MODEL = "ed7d1471eca8"
PRAGMATIC_MODEL = "359d7dd4bcda"
BLIND_RESULT = "9" * 64
LAB_RESULT = "8" * 64
REGRESSION_RESULT = "7" * 64
BLIND_GRADER = "6" * 64
LAB_GRADER = "5" * 64
REGRESSION_GRADER = "4" * 64


def valid_review(role, lane, registration_digest, verdict="PASS"):
    model_id = RADICAL_MODEL if role == "RADICAL_HOSTILE" else PRAGMATIC_MODEL
    return {
        "review_role": role,
        "reviewer_lane": lane,
        "reviewer_model_id": model_id,
        "reviewer_registration_digest": registration_digest,
        "candidate_digest": CANDIDATE,
        "qualification_source_commit": SOURCE_COMMIT,
        "verdict": verdict,
        "blocking_findings": [],
        "nonblocking_findings": [],
        "review_record_commit": "2" * 40,
        "review_record_path": f"messages/{role.lower()}.md",
        "review_record_digest": "e" * 64,
    }


def valid_evidence():
    return {
        "candidate_digest": CANDIDATE,
        "blind": {
            "evidence_kind": "BLIND_EVALUATION",
            "candidate_digest": CANDIDATE,
            "qualification_source_commit": SOURCE_COMMIT,
            "evaluation_set_digest": BLIND_SET,
            "result_digest": BLIND_RESULT,
            "grader_digest": BLIND_GRADER,
            "result_path": "results/blind.json",
            "record_commit": "2" * 40,
            "record_path": "messages/blind.md",
            "record_digest": "3" * 64,
            "set_digest": BLIND_SET,
            "item_count": 77,
            "family_counts": {family: 7 for family in CRITICAL_FAMILIES},
            "critical_failure_ids": [],
        },
        "vera_lab": {
            "evidence_kind": "VERA_LAB",
            "candidate_digest": CANDIDATE,
            "qualification_source_commit": SOURCE_COMMIT,
            "evaluation_set_digest": "1" * 64,
            "result_digest": LAB_RESULT,
            "grader_digest": LAB_GRADER,
            "result_path": "results/vera_lab.json",
            "record_commit": "2" * 40,
            "record_path": "messages/vera-lab.md",
            "record_digest": "3" * 64,
            "critical_scenarios_pass": True,
            "source_digest": "3" * 64,
            "runtime_fixture_digest": "4" * 64,
            "evaluation_digest": "5" * 64,
        },
        "regression": {
            "evidence_kind": "REGRESSION",
            "candidate_digest": CANDIDATE,
            "qualification_source_commit": SOURCE_COMMIT,
            "evaluation_set_digest": "2" * 64,
            "result_digest": REGRESSION_RESULT,
            "grader_digest": REGRESSION_GRADER,
            "result_path": "results/regression.json",
            "record_commit": "2" * 40,
            "record_path": "messages/regression.md",
            "record_digest": "3" * 64,
            "material_competence_regression": False,
            "negative_transfer_intrusion": False,
            "parent_candidate_digest": "6" * 64,
            "comparison_digest": "5" * 64,
        },
        "reviews": [
            valid_review("RADICAL_HOSTILE", RADICAL_LANE, RADICAL_REG),
            valid_review("PRAGMATIC_HOSTILE", PRAGMATIC_LANE, PRAGMATIC_REG),
        ],
    }


def binding():
    return {
        "schema": "VERA_SUCCESSOR_V3_TASK11_REVIEW_BINDING_V1",
        "review_bus_remote": "https://github.com/thebrazenbeard/chat-communication-bus",
        "reviewers": {
            "RADICAL_HOSTILE": {
                "lane_key": RADICAL_LANE,
                "registration_receipt_sha256": RADICAL_REG,
                "registration_bus_commit": "6" * 40,
                "model_id": RADICAL_MODEL,
            },
            "PRAGMATIC_HOSTILE": {
                "lane_key": PRAGMATIC_LANE,
                "registration_receipt_sha256": PRAGMATIC_REG,
                "registration_bus_commit": "6" * 40,
                "model_id": PRAGMATIC_MODEL,
            },
        },
    }


def evaluate_bound(monkeypatch, tmp_path, evidence, record_reasons=()):
    bus = tmp_path / "bus"
    bus.mkdir(exist_ok=True)
    monkeypatch.setattr(promotion_module, "_git_head", lambda repo: SOURCE_COMMIT)
    monkeypatch.setattr(
        promotion_module, "_load_review_binding", lambda repo, commit: binding()
    )
    monkeypatch.setattr(
        promotion_module, "_review_bus_remote_is_canonical", lambda repo: True
    )
    monkeypatch.setattr(
        promotion_module, "_review_bus_transport_is_unrewritten", lambda repo: True
    )
    monkeypatch.setattr(promotion_module, "_refresh_review_bus", lambda repo: None)
    monkeypatch.setattr(
        promotion_module,
        "_review_record_reasons",
        lambda review, review_bus_repo: list(record_reasons),
    )
    monkeypatch.setattr(
        promotion_module,
        "_qualification_record_reasons",
        lambda value, source_commit, evidence_bus_repo: [],
    )
    return evaluate_promotion(
        PromotionEvidence.from_dict(evidence),
        review_bus_repo=bus,
        repo_root=tmp_path,
    )


def test_valid_promotion_evidence_passes_when_external_reviews_are_bound(
    monkeypatch, tmp_path
):
    decision = evaluate_bound(monkeypatch, tmp_path, valid_evidence())
    assert decision.verdict == "PASS"
    assert decision.reasons == ()


def test_locally_fabricated_review_json_cannot_pass_without_review_bus(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(promotion_module, "_git_head", lambda repo: SOURCE_COMMIT)
    monkeypatch.setattr(
        promotion_module, "_load_review_binding", lambda repo, commit: binding()
    )
    decision = evaluate_promotion(
        PromotionEvidence.from_dict(valid_evidence()), repo_root=tmp_path
    )
    assert decision.verdict == "FAIL"
    assert "review_bus_repo_missing" in decision.reasons


def test_same_lane_cannot_satisfy_both_hostile_reviews(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][1]["reviewer_lane"] = evidence["reviews"][0]["reviewer_lane"]
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "reviewer_role_collision" in decision.reasons
    assert "reviewer_lane_not_registered:PRAGMATIC_HOSTILE" in decision.reasons


def test_unregistered_lane_cannot_claim_hostile_role(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0]["reviewer_lane"] = "invented-radical-lane"
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "reviewer_lane_not_registered:RADICAL_HOSTILE" in decision.reasons


def test_registration_digest_must_match_source_binding(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0]["reviewer_registration_digest"] = "f" * 64
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "reviewer_registration_digest_mismatch:RADICAL_HOSTILE" in decision.reasons


def test_review_must_bind_exact_qualification_source(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0]["qualification_source_commit"] = "7" * 40
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "review_qualification_source_mismatch" in decision.reasons


def test_external_record_verification_failure_blocks(monkeypatch, tmp_path):
    decision = evaluate_bound(
        monkeypatch,
        tmp_path,
        valid_evidence(),
        record_reasons=("review_record_digest_mismatch",),
    )
    assert decision.verdict == "FAIL"
    assert "review_record_digest_mismatch" in decision.reasons


def test_blind_minimums_are_fail_closed(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["blind"]["item_count"] = 69
    evidence["blind"]["family_counts"]["privacy"] = 4
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "blind_item_count_below_70" in decision.reasons
    assert "blind_family_below_5:privacy" in decision.reasons


def test_critical_failure_blocks_promotion(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["blind"]["critical_failure_ids"] = ["CF-1"]
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "blind_critical_failure" in decision.reasons


def test_stale_candidate_review_blocks_promotion(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0]["candidate_digest"] = "8" * 64
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "review_candidate_digest_mismatch" in decision.reasons


def test_blocking_finding_and_regressions_block(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0]["blocking_findings"] = ["identity mimicry"]
    evidence["regression"]["material_competence_regression"] = True
    evidence["regression"]["negative_transfer_intrusion"] = True
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "blocking_review_finding" in decision.reasons
    assert "material_competence_regression" in decision.reasons
    assert "negative_transfer_intrusion" in decision.reasons


def test_nonblocking_findings_are_preserved(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0] = valid_review(
        "RADICAL_HOSTILE",
        RADICAL_LANE,
        RADICAL_REG,
        "PASS_WITH_NONBLOCKING_FINDINGS",
    )
    evidence["reviews"][0]["nonblocking_findings"] = ["monitor long sessions"]
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "PASS_WITH_NONBLOCKING_FINDINGS"
    assert decision.nonblocking_findings == ("monitor long sessions",)


def test_review_record_markers_cross_bind_receipt(monkeypatch, tmp_path):
    review = valid_review("RADICAL_HOSTILE", RADICAL_LANE, RADICAL_REG)
    expected = {
        "review_role": review["review_role"],
        "reviewer_lane": review["reviewer_lane"],
        "reviewer_model_id": review["reviewer_model_id"],
        "reviewer_registration_digest": review["reviewer_registration_digest"],
        "candidate_digest": review["candidate_digest"],
        "qualification_source_commit": review["qualification_source_commit"],
        "verdict": review["verdict"],
        "blocking_findings_digest": promotion_module._sha256_json(
            review["blocking_findings"]
        ),
        "nonblocking_findings_digest": promotion_module._sha256_json(
            review["nonblocking_findings"]
        ),
    }
    payload = "\n".join(
        f"REVIEW_FIELD {key}={value}" for key, value in expected.items()
    ).encode("utf-8")
    review["review_record_digest"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        promotion_module, "_remote_contains_commit", lambda repo, commit: True
    )
    monkeypatch.setattr(
        promotion_module, "_git_show_bytes", lambda repo, commit, path: payload
    )
    assert promotion_module._review_record_reasons(
        review, review_bus_repo=tmp_path
    ) == []


def test_review_record_commit_must_exist_on_fresh_remote(monkeypatch, tmp_path):
    review = valid_review("RADICAL_HOSTILE", RADICAL_LANE, RADICAL_REG)
    monkeypatch.setattr(
        promotion_module, "_remote_contains_commit", lambda repo, commit: False
    )
    assert promotion_module._review_record_reasons(
        review, review_bus_repo=tmp_path
    ) == ["review_record_commit_not_on_fresh_remote"]



def test_registered_lane_cannot_substitute_reviewer_model(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["reviews"][0]["reviewer_model_id"] = "different-model"
    decision = evaluate_bound(monkeypatch, tmp_path, evidence)
    assert decision.verdict == "FAIL"
    assert "reviewer_model_not_registered:RADICAL_HOSTILE" in decision.reasons


def test_review_record_rejects_duplicate_contradictory_fields(monkeypatch, tmp_path):
    review = valid_review("RADICAL_HOSTILE", RADICAL_LANE, RADICAL_REG)
    payload = (
        "REVIEW_FIELD review_role=RADICAL_HOSTILE\n"
        "REVIEW_FIELD review_role=PRAGMATIC_HOSTILE\n"
    ).encode("utf-8")
    review["review_record_digest"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        promotion_module, "_remote_contains_commit", lambda repo, commit: True
    )
    monkeypatch.setattr(
        promotion_module, "_git_show_bytes", lambda repo, commit, path: payload
    )
    assert promotion_module._review_record_reasons(
        review, review_bus_repo=tmp_path
    ) == ["review_record_field_duplicate_or_empty"]


def test_review_bus_url_rewrite_blocks_promotion(monkeypatch, tmp_path):
    evidence = valid_evidence()
    bus = tmp_path / "bus"
    bus.mkdir()
    monkeypatch.setattr(promotion_module, "_git_head", lambda repo: SOURCE_COMMIT)
    monkeypatch.setattr(
        promotion_module, "_load_review_binding", lambda repo, commit: binding()
    )
    monkeypatch.setattr(
        promotion_module, "_review_bus_remote_is_canonical", lambda repo: True
    )
    monkeypatch.setattr(
        promotion_module, "_review_bus_transport_is_unrewritten", lambda repo: False
    )
    decision = evaluate_promotion(
        PromotionEvidence.from_dict(evidence),
        review_bus_repo=bus,
        repo_root=tmp_path,
    )
    assert decision.verdict == "FAIL"
    assert "review_bus_transport_rewritten" in decision.reasons




def test_qualification_summary_requires_external_record_fields():
    evidence = valid_evidence()
    del evidence["blind"]["result_digest"]
    with pytest.raises(ValueError):
        PromotionEvidence.from_dict(evidence)


def test_qualification_record_cross_binds_result_and_summary(monkeypatch, tmp_path):
    evidence = valid_evidence()
    blind = evidence["blind"]
    result = tmp_path / "blind.json"
    result.write_bytes(b"frozen blind result bundle")
    blind["result_path"] = str(result)
    blind["result_digest"] = hashlib.sha256(result.read_bytes()).hexdigest()
    expected = {
        "evidence_kind": blind["evidence_kind"],
        "candidate_digest": blind["candidate_digest"],
        "qualification_source_commit": SOURCE_COMMIT,
        "evaluation_set_digest": blind["evaluation_set_digest"],
        "result_digest": blind["result_digest"],
        "grader_digest": blind["grader_digest"],
        "details_digest": promotion_module._sha256_json(
            promotion_module._qualification_detail_payload(
                blind["evidence_kind"], blind
            )
        ),
    }
    payload = "\n".join(
        f"EVIDENCE_FIELD {key}={value}" for key, value in expected.items()
    ).encode("utf-8")
    blind["record_digest"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        promotion_module, "_remote_contains_commit", lambda repo, commit: True
    )
    monkeypatch.setattr(
        promotion_module, "_git_show_bytes", lambda repo, commit, path: payload
    )
    assert promotion_module._qualification_record_reasons(
        blind,
        source_commit=SOURCE_COMMIT,
        evidence_bus_repo=tmp_path,
    ) == []


def test_altered_qualification_result_bytes_block(monkeypatch, tmp_path):
    evidence = valid_evidence()
    blind = evidence["blind"]
    result = tmp_path / "blind.json"
    result.write_bytes(b"tampered result")
    blind["result_path"] = str(result)
    monkeypatch.setattr(
        promotion_module, "_remote_contains_commit", lambda repo, commit: True
    )
    reasons = promotion_module._qualification_record_reasons(
        blind,
        source_commit=SOURCE_COMMIT,
        evidence_bus_repo=tmp_path,
    )
    assert reasons == ["qualification_result_digest_mismatch:BLIND_EVALUATION"]


def test_local_summary_without_external_record_cannot_qualify(monkeypatch, tmp_path):
    evidence = valid_evidence()
    blind = evidence["blind"]
    result = tmp_path / "blind.json"
    result.write_bytes(b"result")
    blind["result_path"] = str(result)
    blind["result_digest"] = hashlib.sha256(result.read_bytes()).hexdigest()
    monkeypatch.setattr(
        promotion_module, "_remote_contains_commit", lambda repo, commit: False
    )
    reasons = promotion_module._qualification_record_reasons(
        blind,
        source_commit=SOURCE_COMMIT,
        evidence_bus_repo=tmp_path,
    )
    assert reasons == [
        "qualification_record_commit_not_on_fresh_remote:BLIND_EVALUATION"
    ]


def test_qualification_summary_must_bind_current_source(monkeypatch, tmp_path):
    evidence = valid_evidence()
    evidence["blind"]["qualification_source_commit"] = "f" * 40
    bus = tmp_path / "bus"
    bus.mkdir()
    monkeypatch.setattr(promotion_module, "_git_head", lambda repo: SOURCE_COMMIT)
    monkeypatch.setattr(
        promotion_module, "_load_review_binding", lambda repo, commit: binding()
    )
    monkeypatch.setattr(
        promotion_module, "_review_bus_remote_is_canonical", lambda repo: True
    )
    monkeypatch.setattr(
        promotion_module, "_review_bus_transport_is_unrewritten", lambda repo: True
    )
    monkeypatch.setattr(promotion_module, "_refresh_review_bus", lambda repo: None)
    monkeypatch.setattr(
        promotion_module,
        "_qualification_record_reasons",
        lambda value, source_commit, evidence_bus_repo: [],
    )
    monkeypatch.setattr(
        promotion_module,
        "_review_record_reasons",
        lambda review, review_bus_repo: [],
    )
    decision = evaluate_promotion(
        PromotionEvidence.from_dict(evidence),
        review_bus_repo=bus,
        repo_root=tmp_path,
    )
    assert decision.verdict == "FAIL"
    assert "qualification_source_mismatch:BLIND_EVALUATION" in decision.reasons


def test_review_receipt_schema_matches_required_contract():
    from pathlib import Path

    schema = json.loads(
        Path("successor/review_receipt.schema.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    assert {
        "review_role",
        "reviewer_lane",
        "reviewer_model_id",
        "reviewer_registration_digest",
        "candidate_digest",
        "qualification_source_commit",
        "verdict",
        "review_record_commit",
        "review_record_path",
        "review_record_digest",
    } <= required
    assert set(schema["properties"]["review_role"]["enum"]) == {
        "RADICAL_HOSTILE",
        "PRAGMATIC_HOSTILE",
    }


def test_unknown_review_role_is_rejected():
    evidence = valid_evidence()
    evidence["reviews"][0]["review_role"] = "FRIENDLY"
    with pytest.raises(ValueError):
        PromotionEvidence.from_dict(evidence)
