from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10_qwen35_retention_candidate import (
    build_candidate,
    canonical_json,
    load_exclusion_hashes,
)
from successor.experiments.retention_graders import (
    check_instruction_contract,
    invalid_instruction_output,
    reference_coding_source,
    reference_instruction_output,
    run_python_test_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    *,
    candidate_path: Path,
    manifest_path: Path,
    exclusion_path: Path,
) -> dict:
    observed_rows = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rebuilt_rows, rebuilt_manifest = build_candidate(
        exclusion_hashes=load_exclusion_hashes(exclusion_path)
    )
    observed_payload = (
        "\n".join(canonical_json(row) for row in observed_rows) + "\n"
    ).encode("utf-8")
    rebuilt_payload = (
        "\n".join(canonical_json(row) for row in rebuilt_rows) + "\n"
    ).encode("utf-8")
    if observed_payload != rebuilt_payload:
        raise RuntimeError("candidate bytes do not match deterministic rebuild")
    observed_sha = hashlib.sha256(observed_payload).hexdigest()
    if observed_sha != observed_manifest["data_sha256"]:
        raise RuntimeError("candidate data hash does not match manifest")
    if observed_sha != rebuilt_manifest["data_sha256"]:
        raise RuntimeError("candidate data hash does not match rebuilt manifest")

    coding_total = 0
    coding_pass = 0
    instruction_total = 0
    instruction_positive_pass = 0
    instruction_negative_pass = 0

    for row in observed_rows:
        grader = row["grader_contract"]
        if row["category"] == "coding":
            coding_total += 1
            source = reference_coding_source(
                grader["family"],
                grader["function"],
            )
            run_python_test_contract(source, grader["tests"])
            coding_pass += 1
        elif row["category"] == "instruction_following":
            instruction_total += 1
            contract = grader["constraints"]
            positive = reference_instruction_output(contract)
            negative = invalid_instruction_output(contract)
            if check_instruction_contract(positive, contract):
                instruction_positive_pass += 1
            if not check_instruction_contract(negative, contract):
                instruction_negative_pass += 1

    if coding_pass != coding_total:
        raise RuntimeError("coding reference audit incomplete")
    if instruction_positive_pass != instruction_total:
        raise RuntimeError("instruction positive audit incomplete")
    if instruction_negative_pass != instruction_total:
        raise RuntimeError("instruction negative audit incomplete")

    return {
        "schema": "V10_QWEN35_RETENTION_CANDIDATE_AUDIT_V1",
        "status": "MECHANICAL_AUDIT_PASS_INDEPENDENT_REVIEW_PENDING",
        "candidate_sha256": observed_sha,
        "candidate_manifest_sha256": _sha256(manifest_path),
        "exclusion_hash_file_sha256": _sha256(exclusion_path),
        "rebuild_byte_identical": True,
        "coding_reference_contracts": {
            "total": coding_total,
            "passed": coding_pass,
        },
        "instruction_contracts": {
            "total": instruction_total,
            "positive_reference_passed": instruction_positive_pass,
            "negative_reference_rejected": instruction_negative_pass,
        },
        "independent_family_audit": {
            "policy": "V10_QWEN35_RETENTION_ADMISSION_V3",
            "required": True,
            "cases_per_family": 5,
            "selection": (
                "SHA256(candidate_sha256, family_id, case_id) "
                "lowest 5 per family"
            ),
            "verified_family_receipts": 0,
            "status": "UNBOUND",
        },
        "claim_ceiling": (
            "ALL_CASE_DETERMINISTIC_MECHANICAL_AUDIT_PASS / "
            "FAMILY_LEVEL_EXTERNAL_MODEL_AUDIT_PENDING / "
            "NOT_FINAL_BANK"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion-hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = audit(
            candidate_path=args.candidate,
            manifest_path=args.manifest,
            exclusion_path=args.exclusion_hashes,
        )
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error": str(exc)}, sort_keys=True))
        return 2
    args.output.write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
