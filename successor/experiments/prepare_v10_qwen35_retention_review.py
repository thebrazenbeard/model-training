from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10_qwen35_retention_candidate import (
    LOCATION_MANIFEST_SHA256,
    _coding_spec,
    _instruction_spec,
    canonical_json,
    load_location_manifest,
    sha256_json,
)
from successor.experiments.v10_qwen35_bank import (
    review_receipt_matches_case,
    review_subject_digest,
    review_subject_payload,
)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate_sha256(rows: list[dict]) -> str:
    payload = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_evidence(row: dict, countries_by_code: dict[str, dict]) -> dict:
    source_id = row["source_id"]
    if ":country:" in source_id:
        code = source_id.rsplit(":country:", 1)[1]
        evidence = countries_by_code[code]
    elif ":pair:" in source_id:
        pair = source_id.rsplit(":pair:", 1)[1]
        left_code, right_code = pair.split(":", 1)
        evidence = {
            "left": countries_by_code[left_code],
            "right": countries_by_code[right_code],
        }
    elif ":coding:" in source_id:
        suffix = source_id.rsplit(":coding:", 1)[1]
        family, variant = suffix.rsplit(":", 1)
        evidence = _coding_spec(family, int(variant))
    elif ":instruction:" in source_id:
        suffix = source_id.rsplit(":instruction:", 1)[1]
        family, variant = suffix.rsplit(":", 1)
        evidence = _instruction_spec(family, int(variant))
    else:
        raise RuntimeError(f"unrecognized retention source id: {source_id}")
    if sha256_json(evidence) != row["source_hash"]:
        raise RuntimeError(f"source evidence hash mismatch: {row['case_id']}")
    return evidence


def build_review_records(rows: list[dict], countries: list[dict]) -> list[dict]:
    countries_by_code = {str(row["code"]): row for row in countries}
    records = []
    for row in sorted(rows, key=lambda item: item["case_id"]):
        if row["review_receipt"]["verdict"] != "PENDING_INDEPENDENT_REVIEW":
            raise RuntimeError(f"case is not pending review: {row['case_id']}")
        subject_digest = review_subject_digest(row)
        if row["review_receipt"]["subject_digest"] != subject_digest:
            raise RuntimeError(f"review subject digest mismatch: {row['case_id']}")
        evidence = _source_evidence(row, countries_by_code)
        records.append({
            "case_id": row["case_id"],
            "category": row["category"],
            "family_id": row["family_id"],
            "review_subject_digest": subject_digest,
            "review_subject": review_subject_payload(row),
            "source_evidence": evidence,
            "source_evidence_sha256": sha256_json(evidence),
            "source_audit_receipt": row["source_audit_receipt"],
            "required_review_receipt": {
                "verdict": "ADMIT_OR_REJECT",
                "reviewer_id": "REQUIRED",
                "reviewer_class": [
                    "HUMAN_REVIEW",
                    "EXTERNAL_MODEL_REVIEW",
                    "EXTERNAL_REVIEW",
                ],
                "provider": "REQUIRED",
                "runtime": "REQUIRED",
                "independent_from_generation": True,
                "artifact_digest": row["review_receipt"]["artifact_digest"],
                "subject_digest": subject_digest,
            },
        })
    return records


def write_review_bundle(
    *,
    rows: list[dict],
    countries: list[dict],
    output_dir: Path,
    batch_size: int = 100,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = build_review_records(rows, countries)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = []
    for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
        batch = records[start:start + batch_size]
        path = output_dir / f"review_batch_{batch_index:03d}.jsonl"
        payload = ("\n".join(canonical_json(record) for record in batch) + "\n").encode("utf-8")
        path.write_bytes(payload)
        batches.append({
            "batch": batch_index,
            "case_count": len(batch),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": path.name,
        })
    manifest = {
        "schema": "V10_QWEN35_RETENTION_REVIEW_BUNDLE_V1",
        "status": "AWAITING_AUTHENTICATED_INDEPENDENT_REVIEW",
        "case_count": len(records),
        "category_counts": dict(sorted(Counter(r["category"] for r in records).items())),
        "candidate_sha256": _candidate_sha256(rows),
        "location_manifest_sha256": LOCATION_MANIFEST_SHA256,
        "batch_size": batch_size,
        "batches": batches,
        "required_receipts": len(records),
        "verified_receipts": 0,
        "plaintext_review_packets_private": True,
        "claim_ceiling": (
            "REVIEW_HANDOFF_PREPARED / NO_INDEPENDENT_RECEIPTS_RETURNED / "
            "NOT_ADMITTED / NOT_FINAL_BANK"
        ),
    }
    manifest_path = output_dir / "review_bundle.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_model_review_receipts(
    *,
    rows: list[dict],
    audit_rows: list[dict],
    audit_receipt: dict,
) -> list[dict]:
    candidate_sha = _candidate_sha256(rows)
    if audit_receipt.get("candidate_sha256") != candidate_sha:
        raise RuntimeError("model audit candidate hash mismatch")
    reviewer = audit_receipt.get("reviewer")
    if not isinstance(reviewer, dict):
        raise RuntimeError("model audit reviewer identity missing")
    for field in ("provider", "model", "model_blob_sha256", "runtime"):
        if not isinstance(reviewer.get(field), str) or not reviewer[field].strip():
            raise RuntimeError(f"model audit reviewer field missing: {field}")
    if len(reviewer["model_blob_sha256"]) != 64:
        raise RuntimeError("model audit reviewer blob digest invalid")

    audit_by_case: dict[str, dict] = {}
    for record in audit_rows:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("model audit row missing case_id")
        if case_id in audit_by_case:
            raise RuntimeError(f"duplicate model audit row: {case_id}")
        audit_by_case[case_id] = record
    expected_ids = {row["case_id"] for row in rows}
    if set(audit_by_case) != expected_ids:
        raise RuntimeError(
            f"model audit case set mismatch: audit={len(audit_by_case)} cases={len(rows)}"
        )

    reviewer_id = f"{reviewer['model']}@sha256:{reviewer['model_blob_sha256'][:16]}"
    runtime = (
        f"{reviewer['runtime']}; temperature={reviewer.get('temperature')}; "
        f"seed={reviewer.get('seed')}; model_blob_sha256={reviewer['model_blob_sha256']}"
    )
    packet_sha = audit_receipt.get("packet_sha256")
    receipts = []
    for row in rows:
        audit = audit_by_case[row["case_id"]]
        review = audit.get("review")
        if audit.get("verdict") != "ADMIT" or not isinstance(review, dict) or review.get("verdict") != "ADMIT":
            raise RuntimeError(f"non-ADMIT model audit verdict: {row['case_id']}")
        receipts.append({
            "case_id": row["case_id"],
            "review_receipt": {
                "verdict": "ADMIT",
                "reviewer_id": reviewer_id,
                "reviewer_class": "EXTERNAL_MODEL_REVIEW",
                "provider": reviewer["provider"],
                "runtime": runtime,
                "independent_from_generation": reviewer_id != row["generation_actor_id"],
                "artifact_digest": row["review_receipt"]["artifact_digest"],
                "subject_digest": review_subject_digest(row),
                "audit_packet_sha256": packet_sha,
                "audit_record_sha256": sha256_json(audit),
                "audit_review_sha256": sha256_json(review),
            },
        })
    return receipts


def apply_review_receipts(
    *,
    rows: list[dict],
    receipts: list[dict],
) -> tuple[list[dict], dict]:
    by_case = {}
    for receipt_record in receipts:
        case_id = receipt_record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("review receipt missing case_id")
        if case_id in by_case:
            raise RuntimeError(f"duplicate review receipt: {case_id}")
        by_case[case_id] = receipt_record

    if set(by_case) != {row["case_id"] for row in rows}:
        raise RuntimeError(
            f"review receipt case set mismatch: receipts={len(by_case)} cases={len(rows)}"
        )

    reviewed = []
    reviewers = Counter()
    for original in rows:
        row = dict(original)
        record = by_case[row["case_id"]]
        receipt = record.get("review_receipt")
        if not isinstance(receipt, dict):
            raise RuntimeError(f"missing review_receipt: {row['case_id']}")
        expected_artifact = original["review_receipt"]["artifact_digest"]
        if receipt.get("artifact_digest") != expected_artifact:
            raise RuntimeError(f"review artifact digest mismatch: {row['case_id']}")
        row["review_receipt"] = receipt
        if not review_receipt_matches_case(row):
            raise RuntimeError(f"invalid independent review receipt: {row['case_id']}")
        reviewers[
            (
                receipt["reviewer_id"],
                receipt["reviewer_class"],
                receipt["provider"],
                receipt["runtime"],
            )
        ] += 1
        reviewed.append(row)

    manifest = {
        "schema": "V10_QWEN35_RETENTION_REVIEW_APPLY_V1",
        "status": "ALL_CASE_RECEIPTS_VALID",
        "case_count": len(reviewed),
        "candidate_sha256_before_review": _candidate_sha256(rows),
        "reviewed_sha256": _candidate_sha256(reviewed),
        "reviewer_bindings": [
            {
                "reviewer_id": key[0],
                "reviewer_class": key[1],
                "provider": key[2],
                "runtime": key[3],
                "case_count": count,
            }
            for key, count in sorted(reviewers.items())
        ],
        "claim_ceiling": (
            "INDEPENDENT_RECEIPTS_STRUCTURALLY_VALID / "
            "SEMANTIC_CONTAMINATION_AND_FINAL_FREEZE_STILL_REQUIRED"
        ),
    }
    return reviewed, manifest


def write_reviewed_files(
    rows: list[dict],
    manifest: dict,
    output_path: Path,
    manifest_path: Path,
) -> dict:
    payload = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    expected = manifest.get("reviewed_sha256")
    if expected is not None and expected != actual:
        raise RuntimeError(f"reviewed hash mismatch: {actual} != {expected}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    written_manifest = dict(manifest)
    written_manifest["reviewed_sha256"] = actual
    written_manifest["reviewed_path"] = output_path.as_posix()
    manifest_path.write_bytes(
        (json.dumps(written_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return written_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--candidate", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=100)

    apply = sub.add_parser("apply")
    apply.add_argument("--candidate", type=Path, required=True)
    apply.add_argument("--candidate-manifest", type=Path, required=True)
    apply.add_argument("--receipts", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--output-manifest", type=Path, required=True)

    args = parser.parse_args()
    rows = _rows(args.candidate)
    candidate_manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.command == "prepare" else json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    if _candidate_sha256(rows) != candidate_manifest["data_sha256"]:
        raise RuntimeError("candidate hash does not match candidate manifest")

    if args.command == "prepare":
        location_manifest = load_location_manifest()
        manifest = write_review_bundle(
            rows=rows,
            countries=location_manifest["countries"],
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0

    receipts = _rows(args.receipts)
    reviewed, manifest = apply_review_receipts(rows=rows, receipts=receipts)
    manifest = write_reviewed_files(
        reviewed,
        manifest,
        args.output,
        args.output_manifest,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
