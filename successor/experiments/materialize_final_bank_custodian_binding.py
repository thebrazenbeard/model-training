from __future__ import annotations

import argparse
import json
from pathlib import Path

from successor.experiments.final_bank_custodian_binding import (
    validate_custodian_binding,
)


def _read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def build_custodian_binding(
    *,
    contract: dict,
    synthetic_pair: dict,
    human_authors: list[dict],
    human_reviewers: list[dict],
    custody: dict,
) -> dict:
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    if contract.get("schema") != "V10_FINAL_BANK_CUSTODIAN_CONTRACT_V1":
        raise ValueError("contract schema mismatch")
    if (
        contract.get("status")
        != "FROZEN_REQUIREMENTS_NO_CUSTODIANS_BOUND"
    ):
        raise ValueError("contract status mismatch")

    if not isinstance(synthetic_pair, dict):
        raise ValueError("synthetic pair must be an object")
    if synthetic_pair.get("status") != "SYNTHETIC_CUSTODIANS_BOUND":
        raise RuntimeError("synthetic custodians are not bound")
    custodians = synthetic_pair.get("custodians")
    if not isinstance(custodians, dict) or set(custodians) != {"A", "B"}:
        raise ValueError("synthetic custodian pair invalid")

    subject = contract.get("composition_subject")
    if not isinstance(subject, dict):
        raise ValueError("contract composition subject missing")

    value = {
        "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_V1",
        "status": "BOUND_FOR_PRETRAINING_SEALED_BANK",
        "composition_subject": {
            "proposal_blob_sha": subject.get("proposal_blob_sha"),
            "generation_spec_sha256": subject.get(
                "generation_spec_sha256"
            ),
            "grader_spec_sha256": subject.get("grader_spec_sha256"),
            "family_manifest_spec_sha256": subject.get(
                "family_manifest_spec_sha256"
            ),
            "diversity_diagnostic_spec_sha256": subject.get(
                "diversity_diagnostic_spec_sha256"
            ),
        },
        "synthetic_custodians": custodians,
        "human_authors": human_authors,
        "human_reviewers": human_reviewers,
        "custody": custody,
        "boundaries": {
            "contains_final_plaintext": False,
            "final_plaintext_generated": False,
            "bank_admitted": False,
            "training_authorized_by_this_binding": False,
        },
        "claim_ceiling": (
            "CUSTODIAN_IDENTITY_AND_ACCESS_BINDING_ONLY / "
            "NO FINAL PLAINTEXT / NO BANK ADMISSION / NO TRAINING"
        ),
    }

    check = validate_custodian_binding(value)
    if check["status"] != "CUSTODIAN_BINDING_PASS":
        raise RuntimeError(
            "custodian binding HOLD: " + " | ".join(check["reasons"])
        )
    return value


def materialize_binding(
    *,
    contract_path: Path,
    synthetic_pair_path: Path,
    human_authors_path: Path,
    human_reviewers_path: Path,
    custody_path: Path,
    output_path: Path,
    write: bool,
) -> dict:
    contract = _read_json(contract_path)
    synthetic_pair = _read_json(synthetic_pair_path)
    human_authors = _read_json(human_authors_path)
    human_reviewers = _read_json(human_reviewers_path)
    custody = _read_json(custody_path)

    if not isinstance(human_authors, list):
        raise ValueError("human authors input must be a JSON array")
    if not isinstance(human_reviewers, list):
        raise ValueError("human reviewers input must be a JSON array")
    if not isinstance(custody, dict):
        raise ValueError("custody input must be a JSON object")

    binding = build_custodian_binding(
        contract=contract,
        synthetic_pair=synthetic_pair,
        human_authors=human_authors,
        human_reviewers=human_reviewers,
        custody=custody,
    )
    if not write:
        return {
            "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_PLAN_V1",
            "status": "READY_TO_WRITE_BOUND_CUSTODY",
            "effect": "READ_ONLY_NO_BINDING_WRITTEN",
            "binding": binding,
        }

    if output_path.exists():
        raise RuntimeError(f"output already exists:{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_MATERIALIZATION_V1",
        "status": "BOUND_CUSTODY_WRITTEN",
        "effect": "CUSTODY_BINDING_METADATA_WRITE_ONLY_NO_FINAL_PLAINTEXT",
        "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--synthetic-pair", type=Path, required=True)
    parser.add_argument("--human-authors", type=Path, required=True)
    parser.add_argument("--human-reviewers", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = materialize_binding(
            contract_path=args.contract,
            synthetic_pair_path=args.synthetic_pair,
            human_authors_path=args.human_authors,
            human_reviewers_path=args.human_reviewers,
            custody_path=args.custody,
            output_path=args.output,
            write=args.write,
        )
        code = 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        result = {
            "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_MATERIALIZATION_V1",
            "status": "HOLD",
            "effect": "NO_BINDING_WRITTEN_NO_FINAL_PLAINTEXT",
            "reasons": [str(exc)],
        }
        code = 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
