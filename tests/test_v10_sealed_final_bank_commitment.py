from __future__ import annotations

from successor.experiments.sealed_final_bank_commitment import (
    DIMENSIONS,
    validate_sealed_final_bank_commitment,
)


def _sha(ch: str) -> str:
    return ch * 64


def _commitment() -> dict:
    return {
        "schema": "V10_SEALED_FINAL_BANK_COMMITMENT_V1",
        "bank_id": "bank-v1",
        "status": "SEALED_PRETRAINING_FINAL_BANK",
        "lane_counts": {
            "behavioral": 10000,
            "adversarial": 2000,
            "retention": 1500,
        },
        "behavioral_dimension_counts": {d: 500 for d in DIMENSIONS},
        "adversarial_dimension_counts": {d: 100 for d in DIMENSIONS},
        "behavioral_family_counts": {d: 50 for d in DIMENSIONS},
        "plaintext_artifacts": {
            "bank_sha256": _sha("1"),
            "grader_bundle_sha256": _sha("2"),
            "review_bundle_sha256": _sha("3"),
            "source_manifest_sha256": _sha("4"),
            "contamination_receipt_sha256": _sha("5"),
            "sealed_archive_sha256": _sha("6"),
        },
        "custody": {
            "plaintext_exposed_to_training_lane": False,
            "custodian_ids": ["custodian-a", "custodian-b", "custodian-h"],
            "human_reviewer_id": "reviewer-h",
            "training_lane_receives_hashes_only": True,
        },
        "admission": {
            "exact_normalized_exclusion": "PASS",
            "semantic_contamination": "PASS",
            "independent_review": "PASS",
            "bank_frozen": True,
            "post_freeze_case_mutation": False,
        },
    }


def test_valid_nonplaintext_commitment_passes() -> None:
    result = validate_sealed_final_bank_commitment(_commitment())
    assert result["status"] == "SEALED_BANK_GATE_PASS"
    assert result["reasons"] == []


def test_plaintext_field_fails_closed() -> None:
    value = _commitment()
    value["prompt"] = "secret final prompt"
    result = validate_sealed_final_bank_commitment(value)
    assert result["status"] == "HOLD"
    assert any("plaintext_field_present" in x for x in result["reasons"])


def test_wrong_lane_or_dimension_counts_hold() -> None:
    value = _commitment()
    value["lane_counts"]["behavioral"] = 9999
    value["behavioral_dimension_counts"]["H03"] = 499
    result = validate_sealed_final_bank_commitment(value)
    assert result["status"] == "HOLD"
    assert any("lane_count:behavioral" in x for x in result["reasons"])
    assert any("behavioral_dimension_count:H03" in x for x in result["reasons"])


def test_missing_family_coverage_holds() -> None:
    value = _commitment()
    value["behavioral_family_counts"]["H12"] = 49
    result = validate_sealed_final_bank_commitment(value)
    assert result["status"] == "HOLD"
    assert "behavioral_family_count:H12:49<50" in result["reasons"]


def test_bad_hash_or_custody_exposure_holds() -> None:
    value = _commitment()
    value["plaintext_artifacts"]["bank_sha256"] = "bad"
    value["custody"]["plaintext_exposed_to_training_lane"] = True
    result = validate_sealed_final_bank_commitment(value)
    assert result["status"] == "HOLD"
    assert "invalid_hash:bank_sha256" in result["reasons"]
    assert "plaintext_exposed_to_training_lane" in result["reasons"]


def test_failed_admission_receipt_holds() -> None:
    value = _commitment()
    value["admission"]["semantic_contamination"] = "HOLD"
    result = validate_sealed_final_bank_commitment(value)
    assert result["status"] == "HOLD"
    assert "admission:semantic_contamination:HOLD" in result["reasons"]
