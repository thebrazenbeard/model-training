from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from successor.experiments import build_v10_qwen35_retention_candidate as v1
from successor.experiments import build_v10_qwen35_retention_candidate_v2 as v2


CHUNK_CONTRACT_REVISION_V3 = (
    "V10_RETENTION_OBJECTIVE_CONTRACTS_20261003_V3_CHUNK_WORDING"
)
CHUNK_GENERATOR_ID_V3 = (
    "v10-retention-candidate-generator-v3-chunk-wording"
)
CHUNK_TASK_V3 = (
    "Split a list into consecutive chunks of positive size n; if elements "
    "remain after the full-size chunks, keep those remaining elements as one "
    "shorter final chunk."
)
PARENT_CANDIDATE_V2_SHA256 = (
    "4ed78c578b7c99504f52a2a8cc29d10396ddb2460471704340b915871c1441dc"
)
FRESH_REQUIRED_PER_FAMILY = 5


def _case_id(family: str, variant: int) -> str:
    return f"ret-code-{family}-{variant:03d}"


def _variant_from_row(row: dict) -> int:
    source_id = str(row.get("source_id", ""))
    suffix = source_id.rsplit(":coding:", 1)[-1]
    family, variant_text = suffix.rsplit(":", 1)
    expected = str(row.get("family_id", "")).split(":", 1)[-1]
    if family != expected:
        raise ValueError(
            f"coding source family mismatch:{row.get('case_id')}:{family}:{expected}"
        )
    return int(variant_text)


def _build_chunk_v3_row(variant: int) -> dict:
    spec = deepcopy(v2._coding_spec_v2("chunk_list", variant))
    spec["task"] = CHUNK_TASK_V3
    digest = v1.sha256_json(spec)
    grader = {
        "kind": "deterministic",
        "grader_id": "python_unit_tests_v1",
        "grader_version": "2",
        "answer_key_digest": v1.sha256_json({"tests": spec["tests"]}),
        "tests": list(spec["tests"]),
        "function": spec["function"],
        "family": "chunk_list",
    }
    prompt = (
        f"Implement this Python function exactly: {spec['signature']}\n"
        f"Contract: {CHUNK_TASK_V3}\n"
        "Return only executable Python code."
    )
    row = v2._row_v2(
        case_id=_case_id("chunk_list", variant),
        category="coding",
        family_id="generated-code:chunk_list",
        prompt=prompt,
        source_id=(
            "vera_model_training:"
            f"{CHUNK_CONTRACT_REVISION_V3}:coding:chunk_list:{variant}"
        ),
        source_record=spec,
        grader_contract=grader,
        artifact_digest=digest,
    )
    row["source_revision"] = CHUNK_CONTRACT_REVISION_V3
    row["generation_method"] = "targeted_deterministic_contract_wording_repair_v3"
    row["generation_actor_id"] = CHUNK_GENERATOR_ID_V3
    row["review_receipt"] = v1._pending_review(
        v1.review_subject_digest(row),
        digest,
    )
    row["source_audit_receipt"] = v1._source_audit(
        row["source_hash"],
        digest,
    )
    return row


def _build_fresh_v2_coding_row(family: str, variant: int) -> dict:
    spec = v2._coding_spec_v2(family, variant)
    prompt = (
        f"Implement this Python function exactly: {spec['signature']}\n"
        f"Contract: {spec['task']}\n"
        "Return only executable Python code."
    )
    digest = v1.sha256_json(spec)
    grader = {
        "kind": "deterministic",
        "grader_id": "python_unit_tests_v1",
        "grader_version": "2",
        "answer_key_digest": v1.sha256_json({"tests": spec["tests"]}),
        "tests": list(spec["tests"]),
        "function": spec["function"],
        "family": family,
    }
    return v2._row_v2(
        case_id=_case_id(family, variant),
        category="coding",
        family_id=f"generated-code:{family}",
        prompt=prompt,
        source_id=(
            "vera_model_training:"
            f"{v2.CODING_CONTRACT_REVISION_V2}:coding:{family}:{variant}"
        ),
        source_record=spec,
        grader_contract=grader,
        artifact_digest=digest,
    )


def _replacement_plan(
    v2_rows: list[dict],
    *,
    consumed_case_ids: set[str],
) -> dict[str, dict[str, list[str]]]:
    by_family: dict[str, list[dict]] = {}
    parent_ids = {row["case_id"] for row in v2_rows}

    for row in v2_rows:
        family_id = row.get("family_id")
        if (
            row.get("category") == "coding"
            and isinstance(family_id, str)
            and family_id.startswith("generated-code:")
        ):
            by_family.setdefault(family_id, []).append(row)

    plan: dict[str, dict[str, list[str]]] = {}
    for family in v1.CODING_FAMILIES:
        family_id = f"generated-code:{family}"
        rows = by_family.get(family_id, [])
        fresh_count = sum(
            row["case_id"] not in consumed_case_ids
            for row in rows
        )
        needed = max(0, FRESH_REQUIRED_PER_FAMILY - fresh_count)
        if not needed:
            continue

        removable = sorted(
            row["case_id"]
            for row in rows
            if row["case_id"] in consumed_case_ids
        )
        if len(removable) < needed:
            raise ValueError(
                f"not enough consumed rows to replenish:{family_id}"
            )

        unused = [
            _case_id(family, variant)
            for variant in range(40)
            if (
                _case_id(family, variant) not in parent_ids
                and _case_id(family, variant) not in consumed_case_ids
            )
        ]
        if len(unused) < needed:
            raise ValueError(
                f"not enough unused deterministic variants:{family_id}"
            )

        plan[family_id] = {
            "remove_case_ids": removable[:needed],
            "add_case_ids": sorted(unused)[:needed],
        }

    return plan


def repair_candidate_v3_rows(
    v2_rows: list[dict],
    *,
    consumed_case_ids: set[str],
) -> list[dict]:
    if len(v2_rows) != 1500:
        raise ValueError(f"expected 1500 parent rows, got {len(v2_rows)}")

    parent_ids = [row.get("case_id") for row in v2_rows]
    if (
        any(
            not isinstance(case_id, str) or not case_id
            for case_id in parent_ids
        )
        or len(parent_ids) != len(set(parent_ids))
    ):
        raise ValueError("candidate V2 case IDs invalid")

    plan = _replacement_plan(
        v2_rows,
        consumed_case_ids=consumed_case_ids,
    )
    remove_ids = {
        case_id
        for item in plan.values()
        for case_id in item["remove_case_ids"]
    }

    repaired: list[dict] = []
    for row in v2_rows:
        case_id = row["case_id"]
        if case_id in remove_ids:
            continue
        if row.get("family_id") == "generated-code:chunk_list":
            repaired.append(_build_chunk_v3_row(_variant_from_row(row)))
        else:
            repaired.append(deepcopy(row))

    for family_id, item in sorted(plan.items()):
        family = family_id.split(":", 1)[1]
        for case_id in item["add_case_ids"]:
            variant = int(case_id.rsplit("-", 1)[1])
            row = (
                _build_chunk_v3_row(variant)
                if family == "chunk_list"
                else _build_fresh_v2_coding_row(family, variant)
            )
            if row["case_id"] != case_id:
                raise ValueError(
                    f"generated replacement id mismatch:{case_id}:{row['case_id']}"
                )
            repaired.append(row)

    repaired = sorted(repaired, key=lambda row: row["case_id"])

    if len(repaired) != 1500:
        raise ValueError(
            f"candidate V3 row count changed:{len(repaired)}"
        )

    if (
        Counter(row["category"] for row in repaired)
        != Counter(row["category"] for row in v2_rows)
    ):
        raise ValueError("candidate V3 category counts changed")
    if (
        Counter(row["family_id"] for row in repaired)
        != Counter(row["family_id"] for row in v2_rows)
    ):
        raise ValueError("candidate V3 family counts changed")

    repaired_ids = {row["case_id"] for row in repaired}
    if len(repaired_ids) != 1500:
        raise ValueError("candidate V3 case IDs are not unique")

    fresh_counts = Counter(
        row["family_id"]
        for row in repaired
        if row["case_id"] not in consumed_case_ids
    )
    family_counts = Counter(row["family_id"] for row in repaired)
    deficient = {
        family_id: fresh_counts[family_id]
        for family_id in family_counts
        if fresh_counts[family_id] < FRESH_REQUIRED_PER_FAMILY
    }
    if deficient:
        raise ValueError(
            f"candidate V3 fresh-family deficit:{deficient}"
        )

    added = repaired_ids - set(parent_ids)
    if added & consumed_case_ids:
        raise ValueError(
            "candidate V3 added previously consumed case IDs"
        )

    return repaired


def build_candidate_v3(
    *,
    parent_rows: list[dict],
    consumed_case_ids: set[str],
    exclusion_hashes: set[str],
) -> tuple[list[dict], dict]:
    rows = repair_candidate_v3_rows(
        parent_rows,
        consumed_case_ids=consumed_case_ids,
    )
    preflight = v1.preflight_retention_rows(
        rows,
        exclusion_hashes=exclusion_hashes,
    )
    if preflight["status"] != "STRUCTURE_READY":
        raise RuntimeError(
            "retention candidate V3 preflight failed: "
            + v1.canonical_json(preflight)
        )

    parent_by_id = {row["case_id"]: row for row in parent_rows}
    rows_by_id = {row["case_id"]: row for row in rows}
    removed = sorted(set(parent_by_id) - set(rows_by_id))
    added = sorted(set(rows_by_id) - set(parent_by_id))
    retained_changed = sorted(
        case_id
        for case_id in set(parent_by_id) & set(rows_by_id)
        if parent_by_id[case_id] != rows_by_id[case_id]
    )
    plan = _replacement_plan(
        parent_rows,
        consumed_case_ids=consumed_case_ids,
    )
    fresh_counts = Counter(
        row["family_id"]
        for row in rows
        if row["case_id"] not in consumed_case_ids
    )

    payload = (
        "\n".join(v1.canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")
    manifest = {
        "schema": "V10_QWEN35_RETENTION_CANDIDATE_MANIFEST_V3",
        "bank_id": "V10_QWEN35_RETENTION_CANDIDATE_1500_20261003_V3",
        "status": (
            "CANDIDATE_OBJECTIVE_STRUCTURE_READY_"
            "CHUNK_WORDING_AND_FRESHNESS_REPAIRED_AUDIT_PENDING"
        ),
        "parent_candidate_sha256": PARENT_CANDIDATE_V2_SHA256,
        "case_count": len(rows),
        "category_counts": dict(
            sorted(Counter(row["category"] for row in rows).items())
        ),
        "family_counts": dict(
            sorted(Counter(row["family_id"] for row in rows).items())
        ),
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "chunk_wording_repair": {
            "family_id": "generated-code:chunk_list",
            "row_count": sum(
                row["family_id"] == "generated-code:chunk_list"
                for row in rows
            ),
            "contract_revision": CHUNK_CONTRACT_REVISION_V3,
            "generator_id": CHUNK_GENERATOR_ID_V3,
            "task_text": CHUNK_TASK_V3,
        },
        "freshness_replenishment": {
            "required_fresh_per_family": FRESH_REQUIRED_PER_FAMILY,
            "predecessor_consumed_case_count": len(consumed_case_ids),
            "replacement_plan": plan,
            "removed_case_ids": removed,
            "added_case_ids": added,
            "replacement_count": len(removed),
            "retained_changed_case_ids": retained_changed,
            "fresh_counts_by_family": dict(sorted(fresh_counts.items())),
        },
        "preflight": preflight,
        "successor_audit": {
            "architecture": "V7",
            "required": True,
            "fresh_semantic_packet_required": True,
            "predecessor_consumed_case_ids": len(consumed_case_ids),
            "status": "UNBOUND",
        },
        "identity_evaluation": {
            "local_koboldcpp_required_before_training": True,
            "host_vs_weight_identity_separation_required": True,
            "status": "UNBOUND",
        },
        "claim_ceiling": (
            "OBJECTIVE_RETENTION_CANDIDATE_V3_1500_STRUCTURALLY_READY / "
            "CHUNK_LIST_WORDING_REPAIRED / FRESH_EVIDENCE_CAPACITY_RESTORED / "
            "MECHANICAL_AND_FRESH_SEMANTIC_AUDIT_PENDING / "
            "LOCAL_IDENTITY_EVALUATION_PENDING / NOT_FINAL_BANK / NO_TRAINING"
        ),
    }
    return rows, manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _consumed_from_packets(paths: list[Path]) -> set[str]:
    consumed: set[str] = set()
    for path in paths:
        consumed.update(
            row["case_id"]
            for row in _read_jsonl(path)
        )
    return consumed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion-hashes", type=Path, required=True)
    parser.add_argument(
        "--predecessor-packet",
        type=Path,
        action="append",
        required=True,
    )
    args = parser.parse_args()

    parent_sha = hashlib.sha256(args.parent.read_bytes()).hexdigest()
    if parent_sha != PARENT_CANDIDATE_V2_SHA256:
        raise ValueError(
            "candidate V3 parent hash mismatch:"
            f"{parent_sha}"
        )

    rows, manifest = build_candidate_v3(
        parent_rows=_read_jsonl(args.parent),
        consumed_case_ids=_consumed_from_packets(args.predecessor_packet),
        exclusion_hashes=v1.load_exclusion_hashes(args.exclusion_hashes),
    )
    manifest = v1.write_candidate_files(
        rows,
        manifest,
        args.output,
        args.manifest,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
