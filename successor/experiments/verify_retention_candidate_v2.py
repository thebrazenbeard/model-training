from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10_qwen35_retention_candidate_v2 import (
    CODING_CONTRACT_REVISION_V2,
    CODING_GENERATOR_ID_V2,
)
from successor.experiments.retention_code_mutants_v2 import mutant_sources
from successor.experiments.retention_graders import (
    reference_coding_source,
    run_python_test_contract,
)


def _tests_pass(source: str, tests: list[str]) -> bool:
    try:
        run_python_test_contract(source, tests)
        return True
    except Exception:
        return False


def verify_candidate_v2_rows(
    v1_rows: list[dict],
    v2_rows: list[dict],
) -> dict:
    v1_ids = [row.get("case_id") for row in v1_rows]
    v2_ids = [row.get("case_id") for row in v2_rows]
    case_id_match = v1_ids == v2_ids

    old_by_id = {
        row.get("case_id"): row
        for row in v1_rows
        if isinstance(row.get("case_id"), str)
    }

    noncoding_changed = 0
    coding_rows = 0
    coding_revision_failures: list[str] = []
    reference_failures: list[str] = []
    mutant_survivors: list[dict] = []
    coding_changed = 0

    for row in v2_rows:
        case_id = row.get("case_id")
        if row.get("category") != "coding":
            if old_by_id.get(case_id) != row:
                noncoding_changed += 1
            continue

        coding_rows += 1
        old = old_by_id.get(case_id)
        if old != row:
            coding_changed += 1

        if (
            row.get("source_revision") != CODING_CONTRACT_REVISION_V2
            or row.get("generation_actor_id") != CODING_GENERATOR_ID_V2
            or CODING_CONTRACT_REVISION_V2
            not in str(row.get("source_id", ""))
        ):
            coding_revision_failures.append(str(case_id))

        grader = row.get("grader_contract", {})
        family = grader.get("family")
        function_name = grader.get("function")
        tests = list(grader.get("tests", []))
        if not isinstance(family, str) or not isinstance(function_name, str):
            reference_failures.append(str(case_id))
            continue

        reference = reference_coding_source(family, function_name)
        if not _tests_pass(reference, tests):
            reference_failures.append(str(case_id))

        for mutant_name, source in mutant_sources(
            family,
            function_name,
        ).items():
            if _tests_pass(source, tests):
                mutant_survivors.append({
                    "case_id": case_id,
                    "family": family,
                    "mutant": mutant_name,
                })

    v1_counts = Counter(row.get("category") for row in v1_rows)
    v2_counts = Counter(row.get("category") for row in v2_rows)
    category_counts_match = v1_counts == v2_counts

    reasons: list[str] = []
    if len(v1_rows) != 1500 or len(v2_rows) != 1500:
        reasons.append("case_count_mismatch")
    if not case_id_match:
        reasons.append("case_id_order_mismatch")
    if not category_counts_match:
        reasons.append("category_counts_mismatch")
    if noncoding_changed:
        reasons.append("noncoding_rows_changed")
    if coding_rows != 250:
        reasons.append("coding_row_count_mismatch")
    if coding_changed != 250:
        reasons.append("coding_rows_not_all_revised")
    if coding_revision_failures:
        reasons.append("coding_revision_binding_failure")
    if reference_failures:
        reasons.append("coding_reference_failure")
    if mutant_survivors:
        reasons.append("coding_mutant_survived")

    return {
        "schema": "V10_QWEN35_RETENTION_CANDIDATE_V2_VERIFY_V1",
        "status": (
            "CANDIDATE_V2_MUTATION_ADEQUATE"
            if not reasons
            else "CANDIDATE_V2_HOLD"
        ),
        "reasons": reasons,
        "case_count": len(v2_rows),
        "case_id_order_match": case_id_match,
        "category_counts_match": category_counts_match,
        "coding_rows": coding_rows,
        "coding_changed": coding_changed,
        "noncoding_changed": noncoding_changed,
        "coding_revision_failures": coding_revision_failures,
        "reference_failures": reference_failures,
        "mutant_survivors": mutant_survivors,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = verify_candidate_v2_rows(
        _read_jsonl(args.v1),
        _read_jsonl(args.v2),
    )
    result["v1_sha256"] = _sha256(args.v1)
    result["v2_sha256"] = _sha256(args.v2)
    result["v1_path"] = args.v1.as_posix()
    result["v2_path"] = args.v2.as_posix()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": result["status"],
        "coding_rows": result["coding_rows"],
        "mutant_survivors": len(result["mutant_survivors"]),
        "reference_failures": len(result["reference_failures"]),
        "noncoding_changed": result["noncoding_changed"],
        "output": args.output.as_posix(),
    }, sort_keys=True))
    return 0 if result["status"] == "CANDIDATE_V2_MUTATION_ADEQUATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
