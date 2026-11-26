from __future__ import annotations

import copy

from successor.experiments.final_bank_retention_binding import (
    validate_retention_lane_binding,
)


def _valid() -> dict:
    return {
        "schema": "V10_FINAL_BANK_RETENTION_LANE_BINDING_V1",
        "status": "RETENTION_LANE_ADMITTED_FOR_FINAL_BANK",
        "lane": "retention",
        "lane_count": 1500,
        "allocation": {
            "knowledge_factuality": 300,
            "reasoning_math": 300,
            "coding": 250,
            "instruction_following": 250,
            "extraction_structured": 200,
            "truthfulness_factual_calibration": 200,
        },
        "source": {
            "repository": "thebrazenbeard/vera_model_training",
            "branch": "work/v10-retention-bank-salvage-20261001",
            "head_sha": "b" * 40,
            "admission_status": "RETENTION_ADMITTED_ARCH_V7",
        },
        "artifacts": {
            "candidate_sha256": "1" * 64,
            "candidate_manifest_sha256": "2" * 64,
            "candidate_verify_sha256": "3" * 64,
            "admission_result_sha256": "4" * 64,
            "result_manifest_sha256": "5" * 64,
            "semantic_receipt_sha256": "6" * 64,
            "witness_adjudication_sha256": "7" * 64,
        },
        "boundaries": {
            "internal_evidence_backed_admission": True,
            "independent_external_review": False,
            "methodology_history_independent": False,
            "full_final_bank_admitted": False,
            "training_authorized": False,
            "weights_changed": False,
        },
    }


def test_exact_admitted_retention_binding_passes() -> None:
    result = validate_retention_lane_binding(_valid())
    assert result["status"] == "RETENTION_LANE_BINDING_PASS"
    assert result["reasons"] == []
    assert result["lane_count"] == 1500


def test_wrong_allocation_fails_closed() -> None:
    value = copy.deepcopy(_valid())
    value["allocation"]["coding"] = 249
    result = validate_retention_lane_binding(value)
    assert result["status"] == "HOLD"
    assert "allocation:coding:249!=250" in result["reasons"]


def test_unadmitted_or_unbound_source_fails_closed() -> None:
    value = copy.deepcopy(_valid())
    value["source"]["admission_status"] = "RETENTION_HOLD_ARCH_V7"
    value["source"]["head_sha"] = "not-a-sha"
    result = validate_retention_lane_binding(value)
    assert result["status"] == "HOLD"
    assert "source_admission_status:RETENTION_HOLD_ARCH_V7" in result["reasons"]
    assert "source_head_sha_invalid" in result["reasons"]


def test_binding_cannot_claim_full_bank_or_training_effect() -> None:
    value = copy.deepcopy(_valid())
    value["boundaries"]["full_final_bank_admitted"] = True
    value["boundaries"]["training_authorized"] = True
    result = validate_retention_lane_binding(value)
    assert result["status"] == "HOLD"
    assert "boundary_full_final_bank_admitted_not_false" in result["reasons"]
    assert "boundary_training_authorized_not_false" in result["reasons"]
