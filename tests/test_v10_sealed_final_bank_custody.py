from __future__ import annotations

import hashlib
import json

import pytest

from successor.experiments.sealed_final_bank_custody import (
    build_public_commitment,
    verify_reveal_manifest,
    verify_file_sha256,
)


def _sha(ch: str) -> str:
    return ch * 64


def _custodian_manifest() -> dict:
    dims = {f"H{i:02d}": 500 for i in range(1, 21)}
    adv = {f"H{i:02d}": 100 for i in range(1, 21)}
    fam = {f"H{i:02d}": 50 for i in range(1, 21)}
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_MANIFEST_V1",
        "bank_id": "v10-final-bank-v1",
        "status": "FROZEN_SEALED_PRETRAINING",
        "lane_counts": {
            "behavioral": 10000,
            "adversarial": 2000,
            "retention": 1500,
        },
        "behavioral_dimension_counts": dims,
        "adversarial_dimension_counts": adv,
        "behavioral_family_counts": fam,
        "artifacts": {
            "bank_sha256": _sha("1"),
            "grader_bundle_sha256": _sha("2"),
            "review_bundle_sha256": _sha("3"),
            "source_manifest_sha256": _sha("4"),
            "contamination_receipt_sha256": _sha("5"),
            "sealed_archive_sha256": _sha("6"),
        },
        "custody": {
            "custodian_ids": ["cust-a", "cust-b", "cust-h"],
            "human_reviewer_id": "review-h",
            "plaintext_exposed_to_training_lane": False,
            "training_lane_receives_hashes_only": True,
            "sealed_storage": {
                "mode": "INDEPENDENT_ACCESS_CONTROL",
                "training_lane_access": False,
                "key_holder": "NOT_APPLICABLE",
            },
        },
        "admission": {
            "exact_normalized_exclusion": "PASS",
            "semantic_contamination": "PASS",
            "independent_review": "PASS",
            "bank_frozen": True,
            "post_freeze_case_mutation": False,
        },
        "freeze": {
            "freeze_subject_digest": _sha("7"),
            "post_freeze_mutation": False,
        },
    }


def test_build_public_commitment_is_deterministic_and_valid() -> None:
    manifest = _custodian_manifest()
    a = build_public_commitment(manifest)
    b = build_public_commitment(json.loads(json.dumps(manifest)))
    assert a == b
    assert a["schema"] == "V10_SEALED_FINAL_BANK_COMMITMENT_V1"
    assert a["status"] == "SEALED_PRETRAINING_FINAL_BANK"
    assert len(a["commitment_sha256"]) == 64
    assert "sealed_storage" not in a["custody"]


def test_build_public_commitment_refuses_training_lane_plaintext_exposure() -> None:
    manifest = _custodian_manifest()
    manifest["custody"]["plaintext_exposed_to_training_lane"] = True
    with pytest.raises(RuntimeError, match="plaintext_exposed"):
        build_public_commitment(manifest)


def test_build_public_commitment_refuses_plaintext_fields() -> None:
    manifest = _custodian_manifest()
    manifest["prompt"] = "secret"
    with pytest.raises(RuntimeError, match="plaintext"):
        build_public_commitment(manifest)


def test_verify_reveal_manifest_accepts_exact_commitment() -> None:
    commitment = build_public_commitment(_custodian_manifest())
    reveal = {
        "schema": "V10_FINAL_BANK_REVEAL_MANIFEST_V1",
        "bank_id": commitment["bank_id"],
        "lane_counts": commitment["lane_counts"],
        "behavioral_dimension_counts": commitment["behavioral_dimension_counts"],
        "adversarial_dimension_counts": commitment["adversarial_dimension_counts"],
        "behavioral_family_counts": commitment["behavioral_family_counts"],
        "plaintext_artifacts": commitment["plaintext_artifacts"],
        "freeze_subject_digest": commitment["freeze_subject_digest"],
    }
    result = verify_reveal_manifest(commitment, reveal)
    assert result["status"] == "REVEAL_MATCH"
    assert result["reasons"] == []


def test_verify_reveal_manifest_holds_on_archive_substitution() -> None:
    commitment = build_public_commitment(_custodian_manifest())
    reveal = {
        "schema": "V10_FINAL_BANK_REVEAL_MANIFEST_V1",
        "bank_id": commitment["bank_id"],
        "lane_counts": commitment["lane_counts"],
        "behavioral_dimension_counts": commitment["behavioral_dimension_counts"],
        "adversarial_dimension_counts": commitment["adversarial_dimension_counts"],
        "behavioral_family_counts": commitment["behavioral_family_counts"],
        "plaintext_artifacts": {
            **commitment["plaintext_artifacts"],
            "sealed_archive_sha256": _sha("9"),
        },
        "freeze_subject_digest": commitment["freeze_subject_digest"],
    }
    result = verify_reveal_manifest(commitment, reveal)
    assert result["status"] == "HOLD"
    assert "artifact_hash_mismatch:sealed_archive_sha256" in result["reasons"]


def test_verify_file_sha256_binds_exact_bytes(tmp_path) -> None:
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed-final-bank")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_file_sha256(path, expected) == expected
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        verify_file_sha256(path, "0" * 64)