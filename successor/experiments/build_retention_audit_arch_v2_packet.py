from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from successor.experiments.retention_audit_arch_v2_validators import (
    load_pinned_location_manifest,
    resolve_source_evidence,
)


RISK_TIEBREAK_NAMESPACE = "ARCH_V2_RISK_TIEBREAK"
SENTINEL_NAMESPACE = "ARCH_V2_SENTINEL"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def packet_bytes(rows: list[dict]) -> bytes:
    return (
        "\n".join(canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")


def _require_candidate_sha(candidate_sha256: str) -> None:
    if (
        not isinstance(candidate_sha256, str)
        or len(candidate_sha256) != 64
        or any(ch not in "0123456789abcdefABCDEF" for ch in candidate_sha256)
    ):
        raise ValueError("candidate_sha256 must be 64 hex characters")


def _hash_parts(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _constraint_count(row: dict) -> int:
    family_id = str(row.get("family_id", ""))
    grader = row.get("grader_contract")
    if not isinstance(grader, dict):
        return 1

    if family_id.startswith("generated-instruction:"):
        constraints = grader.get("constraints")
        return len(constraints) if isinstance(constraints, dict) else 0

    if family_id.startswith("generated-code:"):
        tests = grader.get("tests")
        return len(tests) if isinstance(tests, list) else 0

    if family_id == "location:structured-extraction":
        answer = grader.get("answer_key")
        if not isinstance(answer, str):
            return 0
        try:
            value = json.loads(answer)
        except json.JSONDecodeError:
            return 0
        return len(value) if isinstance(value, dict) else 0

    return 1


def _evidence_policy_weight(row: dict) -> int:
    family_id = str(row.get("family_id", ""))
    if family_id == "evidence-calibration:mode-2":
        return 3
    if family_id.startswith("evidence-calibration:"):
        return 2
    return 0


def _source_entity_count(row: dict) -> int:
    source_id = str(row.get("source_id", ""))
    return 2 if ":pair:" in source_id else 1


def risk_tuple(
    row: dict,
    *,
    candidate_sha256: str,
) -> tuple[int, int, int, int, str]:
    _require_candidate_sha(candidate_sha256)
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("row has invalid case_id")
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{case_id}: prompt must be a string")

    tiebreak = _hash_parts(
        candidate_sha256,
        RISK_TIEBREAK_NAMESPACE,
        case_id,
    )
    return (
        _evidence_policy_weight(row),
        _constraint_count(row),
        _source_entity_count(row),
        len(prompt.encode("utf-8")),
        tiebreak,
    )


def _risk_sort_key(
    row: dict,
    *,
    candidate_sha256: str,
) -> tuple[int, int, int, int, str]:
    weight, constraints, entities, prompt_bytes, tiebreak = risk_tuple(
        row,
        candidate_sha256=candidate_sha256,
    )
    return (
        -weight,
        -constraints,
        -entities,
        -prompt_bytes,
        tiebreak,
    )


def _sentinel_hash(
    row: dict,
    *,
    candidate_sha256: str,
    family_id: str,
) -> str:
    return _hash_parts(
        candidate_sha256,
        SENTINEL_NAMESPACE,
        family_id,
        row["case_id"],
    )


def select_arch_v2_sample(
    rows: list[dict],
    *,
    excluded_case_ids: set[str],
    candidate_sha256: str,
) -> list[dict]:
    _require_candidate_sha(candidate_sha256)

    seen: set[str] = set()
    original_families: set[str] = set()
    fresh_by_family: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        case_id = row.get("case_id")
        family_id = row.get("family_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("row has invalid case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"{case_id}: invalid family_id")
        original_families.add(family_id)
        if case_id in excluded_case_ids:
            continue
        fresh_by_family[family_id].append(row)

    selected: list[dict] = []
    for family_id in sorted(original_families):
        fresh = fresh_by_family.get(family_id, [])
        if len(fresh) < 5:
            raise ValueError(
                f"family {family_id}: requires 5 fresh rows, only "
                f"{len(fresh)} remain"
            )

        ranked = sorted(
            fresh,
            key=lambda row: _risk_sort_key(
                row,
                candidate_sha256=candidate_sha256,
            ),
        )
        risk_rows = ranked[:3]
        risk_ids = {row["case_id"] for row in risk_rows}

        sentinel_pool = [
            row for row in fresh
            if row["case_id"] not in risk_ids
        ]
        sentinels = sorted(
            sentinel_pool,
            key=lambda row: _sentinel_hash(
                row,
                candidate_sha256=candidate_sha256,
                family_id=family_id,
            ),
        )[:2]

        selected.extend(risk_rows)
        selected.extend(sentinels)

    return selected

def build_arch_v2_packet(
    rows: list[dict],
    *,
    excluded_case_ids: set[str],
    candidate_sha256: str,
    predecessor_bindings: list[dict],
    location_manifest: dict | None = None,
) -> tuple[list[dict], dict]:
    selected = select_arch_v2_sample(
        rows,
        excluded_case_ids=excluded_case_ids,
        candidate_sha256=candidate_sha256,
    )

    packet: list[dict] = []
    for row in selected:
        evidence = resolve_source_evidence(
            row,
            location_manifest=location_manifest,
        )
        packet.append({
            "case_id": row["case_id"],
            "category": row["category"],
            "family_id": row["family_id"],
            "prompt": row["prompt"],
            "source_id": row["source_id"],
            "source_revision": row["source_revision"],
            "source_terms": row["source_terms"],
            "source_hash": row["source_hash"],
            "generation_method": row["generation_method"],
            "grader_contract": row["grader_contract"],
            "source_evidence": evidence,
        })

    payload = packet_bytes(packet)
    family_counts = Counter(row["family_id"] for row in packet)
    manifest = {
        "schema": "RETENTION_AUDIT_ARCH_V2_PACKET_MANIFEST_V1",
        "candidate_sha256": candidate_sha256,
        "predecessor_packets": predecessor_bindings,
        "excluded_case_count": len(excluded_case_ids),
        "sample_rows": len(packet),
        "family_count": len(family_counts),
        "per_family": 5,
        "risk_rows_per_family": 3,
        "sentinel_rows_per_family": 2,
        "risk_tiebreak_namespace": RISK_TIEBREAK_NAMESPACE,
        "sentinel_namespace": SENTINEL_NAMESPACE,
        "family_counts": dict(sorted(family_counts.items())),
        "sample_case_ids": [row["case_id"] for row in packet],
        "packet_sha256": hashlib.sha256(payload).hexdigest(),
        "selection": (
            "Exclude all predecessor IDs before ranking. Per family, choose "
            "the three highest rows by the frozen Architecture V2 risk tuple "
            "(first four components descending, tiebreak SHA ascending), then "
            "choose two sentinels by ascending sentinel SHA from the remaining "
            "fresh rows."
        ),
        "disjoint_from_all_predecessors": not (
            {row["case_id"] for row in packet} & excluded_case_ids
        ),
    }
    return packet, manifest


def _read_jsonl_bytes(raw: bytes) -> list[dict]:
    return [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument(
        "--predecessor-packet",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    candidate_raw = args.candidate.read_bytes()
    candidate_sha = hashlib.sha256(candidate_raw).hexdigest()
    if candidate_sha != args.candidate_sha256:
        raise RuntimeError(
            f"candidate hash mismatch:{candidate_sha}!="
            f"{args.candidate_sha256}"
        )
    rows = _read_jsonl_bytes(candidate_raw)

    predecessor_ids: set[str] = set()
    predecessor_bindings: list[dict] = []
    predecessor_case_total = 0
    for path in args.predecessor_packet:
        raw = path.read_bytes()
        packet_rows = _read_jsonl_bytes(raw)
        ids = {row.get("case_id") for row in packet_rows}
        if None in ids or len(ids) != len(packet_rows):
            raise RuntimeError(
                f"predecessor packet case IDs invalid or duplicate:{path}"
            )
        overlap = predecessor_ids & ids
        if overlap:
            raise RuntimeError(
                "predecessor packets overlap:"
                + ",".join(sorted(overlap))
            )
        predecessor_ids.update(ids)
        predecessor_case_total += len(packet_rows)
        predecessor_bindings.append({
            "name": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "case_count": len(packet_rows),
        })

    if len(predecessor_ids) != predecessor_case_total:
        raise RuntimeError("predecessor exclusion cardinality mismatch")

    location_manifest = load_pinned_location_manifest()
    packet, manifest = build_arch_v2_packet(
        rows,
        excluded_case_ids=predecessor_ids,
        candidate_sha256=candidate_sha,
        predecessor_bindings=predecessor_bindings,
        location_manifest=location_manifest,
    )
    payload = packet_bytes(packet)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(json.dumps({
        "packet_sha256": manifest["packet_sha256"],
        "sample_rows": manifest["sample_rows"],
        "family_count": manifest["family_count"],
        "excluded_case_count": manifest["excluded_case_count"],
        "disjoint_from_all_predecessors": (
            manifest["disjoint_from_all_predecessors"]
        ),
        "output": args.output.as_posix(),
        "manifest": args.manifest.as_posix(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
