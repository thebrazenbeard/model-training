from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.final_bank_custodian_binding import (
    validate_custodian_handoff,
)


def git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def verify_handoff_files(
    *,
    binding_path: Path,
    contract_path: Path,
    proposal_path: Path,
    generation_spec_path: Path,
    grader_spec_path: Path,
    family_manifest_spec_path: Path,
    diversity_spec_path: Path,
) -> dict:
    binding = _read_json(binding_path)
    contract = _read_json(contract_path)

    proposal_raw = proposal_path.read_bytes()
    proposal_blob_sha = git_blob_sha(proposal_raw)
    generation_spec_sha256 = sha256_file(generation_spec_path)
    grader_spec_sha256 = sha256_file(grader_spec_path)
    family_manifest_spec_sha256 = sha256_file(
        family_manifest_spec_path
    )
    diversity_diagnostic_spec_sha256 = sha256_file(
        diversity_spec_path
    )

    result = validate_custodian_handoff(
        binding=binding,
        contract=contract,
        proposal_blob_sha=proposal_blob_sha,
        generation_spec_sha256=generation_spec_sha256,
        grader_spec_sha256=grader_spec_sha256,
        family_manifest_spec_sha256=family_manifest_spec_sha256,
        diversity_diagnostic_spec_sha256=(
            diversity_diagnostic_spec_sha256
        ),
    )
    return {
        **result,
        "verified_files": {
            "binding_path": str(binding_path),
            "contract_path": str(contract_path),
            "proposal_path": str(proposal_path),
            "proposal_git_blob_sha": proposal_blob_sha,
            "generation_spec_path": str(generation_spec_path),
            "generation_spec_sha256": generation_spec_sha256,
            "grader_spec_path": str(grader_spec_path),
            "grader_spec_sha256": grader_spec_sha256,
            "family_manifest_spec_path": str(
                family_manifest_spec_path
            ),
            "family_manifest_spec_sha256": (
                family_manifest_spec_sha256
            ),
            "diversity_spec_path": str(diversity_spec_path),
            "diversity_diagnostic_spec_sha256": (
                diversity_diagnostic_spec_sha256
            ),
        },
        "effect": "READ_ONLY_CUSTODIAN_HANDOFF_VERIFICATION",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--generation-spec", type=Path, required=True)
    parser.add_argument("--grader-spec", type=Path, required=True)
    parser.add_argument(
        "--family-manifest-spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--diversity-spec",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)

    try:
        result = verify_handoff_files(
            binding_path=args.binding,
            contract_path=args.contract,
            proposal_path=args.proposal,
            generation_spec_path=args.generation_spec,
            grader_spec_path=args.grader_spec,
            family_manifest_spec_path=args.family_manifest_spec,
            diversity_spec_path=args.diversity_spec,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "V10_FINAL_BANK_CUSTODIAN_HANDOFF_CHECK_V1",
            "status": "HOLD",
            "reasons": [f"handoff_input_error:{exc}"],
            "effect": "READ_ONLY_CUSTODIAN_HANDOFF_VERIFICATION",
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "HANDOFF_READY_FOR_INDEPENDENT_CUSTODY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
