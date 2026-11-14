from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10_qwen35_retention_candidate_v3 import (
    CHUNK_CONTRACT_REVISION_V3,
    CHUNK_GENERATOR_ID_V3,
    FRESH_REQUIRED_PER_FAMILY,
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


def verify_candidate_v3_rows(
    v2_rows: list[dict],
    v3_rows: list[dict],
    *,
    consumed_case_ids: set[str],
) -> dict:
    old = {row.get("case_id"): row for row in v2_rows}
    new = {row.get("case_id"): row for row in v3_rows}
    reasons: list[str] = []

    if len(v2_rows) != 1500 or len(v3_rows) != 1500:
        reasons.append("case_count_mismatch")
    if len(old) != len(v2_rows) or len(new) != len(v3_rows):
        reasons.append("case_id_uniqueness_failure")

    category_match = (
        Counter(row.get("category") for row in v2_rows)
        == Counter(row.get("category") for row in v3_rows)
    )
    family_match = (
        Counter(row.get("family_id") for row in v2_rows)
        == Counter(row.get("family_id") for row in v3_rows)
    )
    if not category_match:
        reasons.append("category_counts_mismatch")
    if not family_match:
        reasons.append("family_counts_mismatch")

    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    if len(removed) != 12 or len(added) != 12:
        reasons.append("replacement_count_mismatch")
    if not set(removed) <= consumed_case_ids:
        reasons.append("removed_unconsumed_case")
    if set(added) & consumed_case_ids:
        reasons.append("added_consumed_case")

    retained_nonchunk_changed: list[str] = []
    chunk_revision_failures: list[str] = []
    coding_rows = 0
    reference_failures: list[str] = []
    mutant_survivors: list[dict] = []

    for row in v3_rows:
        case_id = row.get("case_id")
        family_id = row.get("family_id")
        if case_id in old and family_id != "generated-code:chunk_list":
            if old[case_id] != row:
                retained_nonchunk_changed.append(str(case_id))

        if family_id == "generated-code:chunk_list":
            if (
                row.get("source_revision") != CHUNK_CONTRACT_REVISION_V3
                or row.get("generation_actor_id") != CHUNK_GENERATOR_ID_V3
                or CHUNK_CONTRACT_REVISION_V3
                not in str(row.get("source_id", ""))
            ):
                chunk_revision_failures.append(str(case_id))

        if row.get("category") != "coding":
            continue

        coding_rows += 1
        grader = row.get("grader_contract", {})
        family = grader.get("family")
        function_name = grader.get("function")
        tests = list(grader.get("tests", []))
        if not isinstance(family, str) or not isinstance(function_name, str):
            reference_failures.append(str(case_id))
            continue

        if not _tests_pass(
            reference_coding_source(family, function_name),
            tests,
        ):
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

    if retained_nonchunk_changed:
        reasons.append("retained_nonchunk_rows_changed")
    if chunk_revision_failures:
        reasons.append("chunk_revision_binding_failure")
    if coding_rows != 250:
        reasons.append("coding_row_count_mismatch")
    if reference_failures:
        reasons.append("coding_reference_failure")
    if mutant_survivors:
        reasons.append("coding_mutant_survived")

    fresh_counts = Counter(
        row["family_id"]
        for row in v3_rows
        if row["case_id"] not in consumed_case_ids
    )
    families = Counter(row["family_id"] for row in v3_rows)
    fresh_deficits = {
        family_id: fresh_counts[family_id]
        for family_id in families
        if fresh_counts[family_id] < FRESH_REQUIRED_PER_FAMILY
    }
    if fresh_deficits:
        reasons.append("fresh_family_capacity_failure")

    return {
        "schema": "V10_QWEN35_RETENTION_CANDIDATE_V3_VERIFY_V1",
        "status": (
            "CANDIDATE_V3_MUTATION_AND_FRESHNESS_ADEQUATE"
            if not reasons
            else "CANDIDATE_V3_HOLD"
        ),
        "reasons": reasons,
        "case_count": len(v3_rows),
        "category_counts_match": category_match,
        "family_counts_match": family_match,
        "removed_case_ids": removed,
        "added_case_ids": added,
        "replacement_count": len(removed),
        "removed_all_consumed": set(removed) <= consumed_case_ids,
        "added_all_fresh": not bool(set(added) & consumed_case_ids),
        "retained_nonchunk_changed": retained_nonchunk_changed,
        "chunk_revision_failures": chunk_revision_failures,
        "coding_rows": coding_rows,
        "reference_failures": reference_failures,
        "mutant_survivors": mutant_survivors,
        "fresh_counts_by_family": dict(sorted(fresh_counts.items())),
        "fresh_deficits": fresh_deficits,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument(
        "--predecessor-packet",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    consumed: set[str] = set()
    for path in args.predecessor_packet:
        consumed.update(row["case_id"] for row in _read_jsonl(path))

    result = verify_candidate_v3_rows(
        _read_jsonl(args.v2),
        _read_jsonl(args.v3),
        consumed_case_ids=consumed,
    )
    result["v2_sha256"] = _sha(args.v2)
    result["v3_sha256"] = _sha(args.v3)
    result["predecessor_consumed_case_count"] = len(consumed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": result["status"],
        "replacement_count": result["replacement_count"],
        "coding_rows": result["coding_rows"],
        "reference_failures": len(result["reference_failures"]),
        "mutant_survivors": len(result["mutant_survivors"]),
        "fresh_deficits": result["fresh_deficits"],
        "output": args.output.as_posix(),
    }, sort_keys=True))
    return (
        0
        if result["status"] == "CANDIDATE_V3_MUTATION_AND_FRESHNESS_ADEQUATE"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
