from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.verify_v10_final_bank_custodian_handoff import (
    git_blob_sha,
    verify_handoff_files,
)


def _sha(ch: str) -> str:
    return ch * 64


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _binding(subject: dict) -> dict:
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_V1",
        "status": "BOUND_FOR_PRETRAINING_SEALED_BANK",
        "composition_subject": dict(subject),
        "synthetic_custodians": {
            "A": {
                "actor_id": "synthetic-a",
                "provider": "provider-a",
                "model": "model-a",
                "model_family": "family-a",
                "revision": "rev-a",
                "runtime": "runtime-a",
                "seed": 101,
                "sampling": {"temperature": 0.7},
                "identity_receipt_sha256": _sha("1"),
            },
            "B": {
                "actor_id": "synthetic-b",
                "provider": "provider-b",
                "model": "model-b",
                "model_family": "family-b",
                "revision": "rev-b",
                "runtime": "runtime-b",
                "seed": 202,
                "sampling": {"temperature": 0.7},
                "identity_receipt_sha256": _sha("2"),
            },
        },
        "human_authors": [
            {
                "actor_id": "human-author",
                "private_identity_attestation_sha256": _sha("3"),
            }
        ],
        "human_reviewers": [
            {
                "actor_id": "human-reviewer",
                "private_identity_attestation_sha256": _sha("4"),
            }
        ],
        "custody": {
            "surface_id": "custody-surface",
            "surface_class": "INDEPENDENT_CUSTODY_EVALUATION_LANE",
            "training_lane_access": False,
            "candidate_development_access": False,
            "hash_only_publication_before_candidate_freeze": True,
            "plaintext_sealed_until_candidate_freeze": True,
            "access_control_receipt_sha256": _sha("5"),
        },
    }


def _fixture(tmp_path: Path):
    proposal = tmp_path / "proposal.md"
    generation = tmp_path / "generation.json"
    grader = tmp_path / "grader.json"
    family_manifest_spec = tmp_path / "family-manifest-spec.json"
    diversity_spec = tmp_path / "diversity-spec.json"
    proposal_raw = b"# amended proposal\n"
    generation_raw = b'{"schema":"generation"}\n'
    grader_raw = b'{"schema":"grader"}\n'
    family_manifest_raw = b'{"schema":"family-manifest"}\n'
    diversity_raw = b'{"schema":"diversity"}\n'
    _write(proposal, proposal_raw)
    _write(generation, generation_raw)
    _write(grader, grader_raw)
    _write(family_manifest_spec, family_manifest_raw)
    _write(diversity_spec, diversity_raw)

    subject = {
        "proposal_blob_sha": git_blob_sha(proposal_raw),
        "generation_spec_sha256": hashlib.sha256(
            generation_raw
        ).hexdigest(),
        "grader_spec_sha256": hashlib.sha256(grader_raw).hexdigest(),
        "family_manifest_spec_sha256": hashlib.sha256(
            family_manifest_raw
        ).hexdigest(),
        "diversity_diagnostic_spec_sha256": hashlib.sha256(
            diversity_raw
        ).hexdigest(),
    }
    contract = {
        "schema": "V10_FINAL_BANK_CUSTODIAN_CONTRACT_V1",
        "status": "FROZEN_REQUIREMENTS_NO_CUSTODIANS_BOUND",
        "composition_subject": dict(subject),
    }
    binding = _binding(subject)
    contract_path = tmp_path / "contract.json"
    binding_path = tmp_path / "binding.json"
    _write(contract_path, contract)
    _write(binding_path, binding)
    return (
        binding_path,
        contract_path,
        proposal,
        generation,
        grader,
        family_manifest_spec,
        diversity_spec,
    )


def test_git_blob_sha_uses_git_blob_object_identity() -> None:
    raw = b"abc"
    expected = hashlib.sha1(b"blob 3\0abc").hexdigest()
    assert git_blob_sha(raw) == expected


def test_verify_handoff_files_passes_exact_bound_subject(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = verify_handoff_files(
        binding_path=paths[0],
        contract_path=paths[1],
        proposal_path=paths[2],
        generation_spec_path=paths[3],
        grader_spec_path=paths[4],
        family_manifest_spec_path=paths[5],
        diversity_spec_path=paths[6],
    )
    assert result["status"] == "HANDOFF_READY_FOR_INDEPENDENT_CUSTODY"
    assert result["reasons"] == []


def test_verify_handoff_files_detects_generation_spec_substitution(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths[3].write_bytes(b'{"schema":"substituted"}\n')
    result = verify_handoff_files(
        binding_path=paths[0],
        contract_path=paths[1],
        proposal_path=paths[2],
        generation_spec_path=paths[3],
        grader_spec_path=paths[4],
        family_manifest_spec_path=paths[5],
        diversity_spec_path=paths[6],
    )
    assert result["status"] == "HOLD"
    assert "generation_spec_file_sha256_mismatch" in result["reasons"]


def test_verify_handoff_files_detects_proposal_substitution(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths[2].write_bytes(b"# substituted proposal\n")
    result = verify_handoff_files(
        binding_path=paths[0],
        contract_path=paths[1],
        proposal_path=paths[2],
        generation_spec_path=paths[3],
        grader_spec_path=paths[4],
        family_manifest_spec_path=paths[5],
        diversity_spec_path=paths[6],
    )
    assert result["status"] == "HOLD"
    assert any(
        reason.startswith("proposal_blob_sha_file_subject_mismatch:")
        for reason in result["reasons"]
    )



def test_verify_handoff_files_detects_family_manifest_spec_substitution(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths[5].write_bytes(b'{"schema":"substituted-family"}\n')
    result = verify_handoff_files(
        binding_path=paths[0],
        contract_path=paths[1],
        proposal_path=paths[2],
        generation_spec_path=paths[3],
        grader_spec_path=paths[4],
        family_manifest_spec_path=paths[5],
        diversity_spec_path=paths[6],
    )
    assert result["status"] == "HOLD"
    assert "family_manifest_spec_file_sha256_mismatch" in result["reasons"]


def test_verify_handoff_files_detects_diversity_spec_substitution(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths[6].write_bytes(b'{"schema":"substituted-diversity"}\n')
    result = verify_handoff_files(
        binding_path=paths[0],
        contract_path=paths[1],
        proposal_path=paths[2],
        generation_spec_path=paths[3],
        grader_spec_path=paths[4],
        family_manifest_spec_path=paths[5],
        diversity_spec_path=paths[6],
    )
    assert result["status"] == "HOLD"
    assert "diversity_diagnostic_spec_file_sha256_mismatch" in result["reasons"]
