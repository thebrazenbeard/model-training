from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.verify_v10_final_bank_custodian_handoff import (
    verify_handoff_files,
)


ROOT = Path(__file__).parents[1]
EXP = ROOT / "successor" / "experiments"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_live_binding() -> dict:
    pair = json.loads(
        (EXP / "V10_FINAL_BANK_SYNTHETIC_CUSTODIANS_V1.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_V1",
        "status": "BOUND_FOR_PRETRAINING_SEALED_BANK",
        "composition_subject": {
            "proposal_blob_sha": "3b157d536279c17f1fcfa22cb7529fe87f5b83a9",
            "generation_spec_sha256": _sha(
                EXP / "V10_H01_H20_FINAL_BANK_GENERATION_SPEC_V1.json"
            ),
            "grader_spec_sha256": _sha(
                EXP / "V10_H01_H20_FINAL_BANK_GRADER_SPEC_V1.json"
            ),
            "family_manifest_spec_sha256": _sha(
                EXP / "V10_FINAL_BANK_FAMILY_MANIFEST_SPEC_V1.json"
            ),
            "diversity_diagnostic_spec_sha256": _sha(
                EXP / "V10_FINAL_BANK_DIVERSITY_DIAGNOSTIC_SPEC_V1.json"
            ),
        },
        "synthetic_custodians": pair["custodians"],
        "human_authors": [{
            "actor_id": "fixture-human-author",
            "private_identity_attestation_sha256": "a" * 64,
        }],
        "human_reviewers": [{
            "actor_id": "fixture-human-reviewer",
            "private_identity_attestation_sha256": "b" * 64,
        }],
        "custody": {
            "surface_id": "fixture-independent-surface",
            "surface_class": "INDEPENDENT_CUSTODY_EVALUATION_LANE",
            "training_lane_access": False,
            "candidate_development_access": False,
            "hash_only_publication_before_candidate_freeze": True,
            "plaintext_sealed_until_candidate_freeze": True,
            "access_control_receipt_sha256": "c" * 64,
        },
    }


def test_repository_frozen_spec_bytes_match_custody_contract(tmp_path: Path) -> None:
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(
        json.dumps(_valid_live_binding()),
        encoding="utf-8",
    )
    result = verify_handoff_files(
        binding_path=binding_path,
        contract_path=EXP / "V10_FINAL_BANK_CUSTODIAN_CONTRACT_V1.json",
        proposal_path=(
            ROOT
            / "research/measurement/"
            "V10_H01_H20_FINAL_BANK_COMPOSITION_PROPOSAL_20261001_V1.md"
        ),
        generation_spec_path=EXP / "V10_H01_H20_FINAL_BANK_GENERATION_SPEC_V1.json",
        grader_spec_path=EXP / "V10_H01_H20_FINAL_BANK_GRADER_SPEC_V1.json",
        family_manifest_spec_path=EXP / "V10_FINAL_BANK_FAMILY_MANIFEST_SPEC_V1.json",
        diversity_spec_path=EXP / "V10_FINAL_BANK_DIVERSITY_DIAGNOSTIC_SPEC_V1.json",
    )
    assert result["status"] == "HANDOFF_READY_FOR_INDEPENDENT_CUSTODY"
    assert result["reasons"] == []
