from __future__ import annotations

import json
from pathlib import Path

import pytest

from successor.experiments.materialize_final_bank_custodian_binding import (
    build_custodian_binding,
)


ROOT = Path(__file__).parents[1]
EXP = ROOT / "successor" / "experiments"


def _pair() -> dict:
    return json.loads(
        (EXP / "V10_FINAL_BANK_SYNTHETIC_CUSTODIANS_V1.json").read_text(
            encoding="utf-8"
        )
    )


def _contract() -> dict:
    return json.loads(
        (EXP / "V10_FINAL_BANK_CUSTODIAN_CONTRACT_V1.json").read_text(
            encoding="utf-8"
        )
    )


def test_materializes_valid_nonplaintext_binding_from_real_identity_receipts() -> None:
    value = build_custodian_binding(
        contract=_contract(),
        synthetic_pair=_pair(),
        human_authors=[{
            "actor_id": "human-author-A",
            "private_identity_attestation_sha256": "a" * 64,
        }],
        human_reviewers=[{
            "actor_id": "human-reviewer-B",
            "private_identity_attestation_sha256": "b" * 64,
        }],
        custody={
            "surface_id": "independent-custody-01",
            "surface_class": "INDEPENDENT_CUSTODY_EVALUATION_LANE",
            "training_lane_access": False,
            "candidate_development_access": False,
            "hash_only_publication_before_candidate_freeze": True,
            "plaintext_sealed_until_candidate_freeze": True,
            "access_control_receipt_sha256": "c" * 64,
        },
    )
    assert value["schema"] == "V10_FINAL_BANK_CUSTODIAN_BINDING_V1"
    assert value["status"] == "BOUND_FOR_PRETRAINING_SEALED_BANK"
    assert value["synthetic_custodians"] == _pair()["custodians"]
    assert value["human_authors"][0]["actor_id"] == "human-author-A"
    assert value["human_reviewers"][0]["actor_id"] == "human-reviewer-B"
    assert value["custody"]["surface_id"] == "independent-custody-01"
    serialized = json.dumps(value).casefold()
    assert '"prompt"' not in serialized
    assert '"answer"' not in serialized
    assert '"case"' not in serialized


def test_refuses_missing_human_or_custody_evidence() -> None:
    with pytest.raises(RuntimeError, match="custodian binding HOLD"):
        build_custodian_binding(
            contract=_contract(),
            synthetic_pair=_pair(),
            human_authors=[],
            human_reviewers=[],
            custody={
                "surface_id": None,
                "surface_class": "INDEPENDENT_CUSTODY_EVALUATION_LANE",
                "training_lane_access": False,
                "candidate_development_access": False,
                "hash_only_publication_before_candidate_freeze": True,
                "plaintext_sealed_until_candidate_freeze": True,
                "access_control_receipt_sha256": None,
            },
        )


def test_refuses_author_reviewer_overlap() -> None:
    actor = {
        "actor_id": "same-human",
        "private_identity_attestation_sha256": "d" * 64,
    }
    with pytest.raises(RuntimeError, match="human_author_reviewer_overlap"):
        build_custodian_binding(
            contract=_contract(),
            synthetic_pair=_pair(),
            human_authors=[actor],
            human_reviewers=[actor],
            custody={
                "surface_id": "independent-custody-01",
                "surface_class": "INDEPENDENT_CUSTODY_EVALUATION_LANE",
                "training_lane_access": False,
                "candidate_development_access": False,
                "hash_only_publication_before_candidate_freeze": True,
                "plaintext_sealed_until_candidate_freeze": True,
                "access_control_receipt_sha256": "e" * 64,
            },
        )
