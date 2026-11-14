from __future__ import annotations

import copy

import pytest

from successor.experiments.final_bank_generation_work_order import (
    prepare_generation_work_order,
)


def _sha(ch: str) -> str:
    return ch * 64


def _binding() -> dict:
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_V1",
        "status": "BOUND_FOR_PRETRAINING_SEALED_BANK",
        "composition_subject": {
            "proposal_blob_sha": "3b157d536279c17f1fcfa22cb7529fe87f5b83a9",
            "generation_spec_sha256": _sha("1"),
            "grader_spec_sha256": _sha("2"),
        },
        "synthetic_custodians": {
            "A": {
                "actor_id": "synthetic-a",
                "provider": "provider-a",
                "model": "model-a",
                "model_family": "family-a",
                "revision": "rev-a",
                "runtime": "runtime-a",
                "seed": 101,
                "sampling": {"temperature": 0.6},
                "identity_receipt_sha256": _sha("3"),
            },
            "B": {
                "actor_id": "synthetic-b",
                "provider": "provider-b",
                "model": "model-b",
                "model_family": "family-b",
                "revision": "rev-b",
                "runtime": "runtime-b",
                "seed": 202,
                "sampling": {"temperature": 0.6},
                "identity_receipt_sha256": _sha("4"),
            },
        },
        "human_authors": [{
            "actor_id": "human-author-1",
            "private_identity_attestation_sha256": _sha("5"),
        }],
        "human_reviewers": [{
            "actor_id": "human-reviewer-1",
            "private_identity_attestation_sha256": _sha("6"),
        }],
        "custody": {
            "surface_id": "sealed-custody-surface-v1",
            "surface_class": "INDEPENDENT_CUSTODY_EVALUATION_LANE",
            "training_lane_access": False,
            "candidate_development_access": False,
            "hash_only_publication_before_candidate_freeze": True,
            "plaintext_sealed_until_candidate_freeze": True,
            "access_control_receipt_sha256": _sha("7"),
        },
    }


def test_work_order_has_exact_frozen_family_allocation_without_plaintext() -> None:
    order = prepare_generation_work_order(_binding())
    assert order["status"] == "READY_FOR_CUSTODY_GENERATION"
    assert order["family_count"] == 1400
    assert order["lane_family_counts"] == {"behavioral": 1000, "adversarial": 400}
    assert order["planned_case_count"] == 12000
    assert len({row["family_id"] for row in order["families"]}) == 1400
    assert all("prompt" not in row and "case" not in row for row in order["families"])

    for dimension in [f"H{i:02d}" for i in range(1, 21)]:
        behavioral = [
            row for row in order["families"]
            if row["lane"] == "behavioral" and row["dimension"] == dimension
        ]
        adversarial = [
            row for row in order["families"]
            if row["lane"] == "adversarial" and row["dimension"] == dimension
        ]
        assert len(behavioral) == 50
        assert len(adversarial) == 20
        assert [r["provenance_class"] for r in behavioral].count("synthetic_A") == 20
        assert [r["provenance_class"] for r in behavioral].count("synthetic_B") == 20
        assert [r["provenance_class"] for r in behavioral].count("human_seeded") == 10
        assert [r["provenance_class"] for r in adversarial].count("synthetic_A") == 8
        assert [r["provenance_class"] for r in adversarial].count("synthetic_B") == 8
        assert [r["provenance_class"] for r in adversarial].count("human_seeded") == 4


def test_work_order_refuses_unbound_human_or_custody_state() -> None:
    value = copy.deepcopy(_binding())
    value["human_reviewers"] = []
    with pytest.raises(RuntimeError, match="custodian binding HOLD"):
        prepare_generation_work_order(value)
