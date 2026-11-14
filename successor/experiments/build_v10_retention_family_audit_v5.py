from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10_retention_audit_packet import (
    canonical_json,
    load_location_manifest,
    sha256_bytes,
    source_evidence_for_row,
    write_packet_files,
)


SELECTION_NAMESPACE = "RETENTION_FAMILY_AUDIT_V5_DISJOINT_20261002"
EXPECTED_PRIOR_PACKET_SHA256 = (
    "bed5c8133d23be9a5d67f74416fb63ce4f27fe84a080d5802ca21c2bb4ce421d"
)
EXPECTED_PRIOR_CASES = 130
EXPECTED_FAMILIES = 26
EXPECTED_PRIOR_PER_FAMILY = 5


def _rank(
    *,
    candidate_sha256: str,
    family_id: str,
    case_id: str,
) -> str:
    return hashlib.sha256(
        (
            candidate_sha256
            + "\0"
            + SELECTION_NAMESPACE
            + "\0"
            + family_id
            + "\0"
            + case_id
        ).encode("utf-8")
    ).hexdigest()


def select_disjoint_family_audit_sample(
    rows: list[dict],
    *,
    excluded_case_ids: set[str],
    per_family: int,
    candidate_sha256: str,
) -> list[dict]:
    if per_family < 1:
        raise ValueError("per_family must be positive")
    if not isinstance(candidate_sha256, str) or len(candidate_sha256) != 64:
        raise ValueError("candidate_sha256 must be 64 hex characters")

    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_case_ids: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        family_id = row.get("family_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("row has invalid case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"{case_id}: invalid family_id")
        if case_id in excluded_case_ids:
            continue
        grouped[family_id].append(row)

    selected: list[dict] = []
    for family_id in sorted(grouped):
        ranked = sorted(
            grouped[family_id],
            key=lambda row: _rank(
                candidate_sha256=candidate_sha256,
                family_id=family_id,
                case_id=row["case_id"],
            ),
        )
        if len(ranked) < per_family:
            raise ValueError(
                f"family {family_id}: requested {per_family} disjoint rows, "
                f"only {len(ranked)} remain"
            )
        selected.extend(ranked[:per_family])

    original_families = {
        row.get("family_id")
        for row in rows
        if isinstance(row.get("family_id"), str)
    }
    if set(grouped) != original_families:
        missing = sorted(original_families - set(grouped))
        raise ValueError(
            "excluded set removed all rows from family/families: "
            + ",".join(missing)
        )
    return selected


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_v5_packet(
    *,
    candidate_path: Path,
    prior_packet_path: Path,
    prior_packet_manifest_path: Path,
    expected_candidate_sha256: str,
    per_family: int = 5,
) -> tuple[list[dict], dict]:
    candidate_raw = candidate_path.read_bytes()
    candidate_sha = sha256_bytes(candidate_raw)
    if candidate_sha != expected_candidate_sha256:
        raise RuntimeError(
            f"candidate hash mismatch:{candidate_sha}!="
            f"{expected_candidate_sha256}"
        )

    prior_raw = prior_packet_path.read_bytes()
    prior_sha = sha256_bytes(prior_raw)
    prior_manifest = json.loads(
        prior_packet_manifest_path.read_text(encoding="utf-8")
    )
    if prior_sha != EXPECTED_PRIOR_PACKET_SHA256:
        raise RuntimeError(
            f"prior packet hash mismatch:{prior_sha}!="
            f"{EXPECTED_PRIOR_PACKET_SHA256}"
        )
    if prior_manifest.get("packet_sha256") != prior_sha:
        raise RuntimeError("prior packet manifest hash mismatch")
    if prior_manifest.get("candidate_sha256") != candidate_sha:
        raise RuntimeError("prior packet candidate mismatch")

    prior_rows = _read_jsonl(prior_packet_path)
    prior_ids = {row.get("case_id") for row in prior_rows}
    if None in prior_ids or len(prior_ids) != len(prior_rows):
        raise RuntimeError("prior packet case IDs invalid or duplicate")
    if len(prior_ids) != EXPECTED_PRIOR_CASES:
        raise RuntimeError(
            f"prior packet case count:{len(prior_ids)}!="
            f"{EXPECTED_PRIOR_CASES}"
        )
    prior_family_counts = Counter(
        row.get("family_id") for row in prior_rows
    )
    if len(prior_family_counts) != EXPECTED_FAMILIES:
        raise RuntimeError(
            f"prior packet family count:{len(prior_family_counts)}!="
            f"{EXPECTED_FAMILIES}"
        )
    bad_prior_families = {
        family_id: count
        for family_id, count in prior_family_counts.items()
        if count != EXPECTED_PRIOR_PER_FAMILY
    }
    if bad_prior_families:
        raise RuntimeError(
            "prior packet per-family mismatch:"
            + canonical_json(bad_prior_families)
        )

    rows = [
        json.loads(line)
        for line in candidate_raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    selected = select_disjoint_family_audit_sample(
        rows,
        excluded_case_ids=prior_ids,
        per_family=per_family,
        candidate_sha256=candidate_sha,
    )
    selected_ids = {row["case_id"] for row in selected}
    overlap = selected_ids & prior_ids
    if overlap:
        raise RuntimeError(
            "V5 selection overlaps prior packet:"
            + ",".join(sorted(overlap))
        )

    selected_family_counts = Counter(
        row["family_id"] for row in selected
    )
    if len(selected_family_counts) != EXPECTED_FAMILIES:
        raise RuntimeError(
            f"V5 family count:{len(selected_family_counts)}!="
            f"{EXPECTED_FAMILIES}"
        )
    if any(count != per_family for count in selected_family_counts.values()):
        raise RuntimeError("V5 selected family count mismatch")

    location = load_location_manifest()
    countries = {
        str(country["code"]): country
        for country in location["countries"]
    }
    packet = []
    for row in selected:
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
            "source_evidence": source_evidence_for_row(
                row,
                countries,
            ),
        })

    payload = (
        "\n".join(canonical_json(row) for row in packet)
        + "\n"
    ).encode("utf-8")
    manifest = {
        "schema": "V10_RETENTION_FAMILY_AUDIT_PACKET_V5",
        "candidate_sha256": candidate_sha,
        "predecessor_packet_sha256": prior_sha,
        "excluded_case_count": len(prior_ids),
        "sample_rows": len(packet),
        "family_count": len(selected_family_counts),
        "per_family": per_family,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection": (
            "Exclude all predecessor packet case IDs; then rank remaining "
            "rows per family by SHA256(candidate_sha256,NUL,namespace,NUL,"
            "family_id,NUL,case_id) ascending and take lowest per_family."
        ),
        "disjoint_from_predecessor": True,
        "packet_sha256": sha256_bytes(payload),
        "review_role": (
            "fresh methodology-repair benchmark audit; deterministic "
            "source/test truth remains authoritative"
        ),
    }
    return packet, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--prior-packet", type=Path, required=True)
    parser.add_argument("--prior-packet-manifest", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    packet, manifest = build_v5_packet(
        candidate_path=args.candidate,
        prior_packet_path=args.prior_packet,
        prior_packet_manifest_path=args.prior_packet_manifest,
        expected_candidate_sha256=args.candidate_sha256,
        per_family=args.per_family,
    )
    manifest = write_packet_files(
        packet,
        manifest,
        args.output,
        args.manifest,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
