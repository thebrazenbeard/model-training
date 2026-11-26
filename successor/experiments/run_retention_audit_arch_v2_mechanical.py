from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.retention_audit_arch_v2_qualification import (
    qualify_validators,
)
from successor.experiments.retention_audit_arch_v2_validators import (
    load_pinned_location_manifest,
    validate_candidate_mechanically,
)


def run_mechanical_audit(
    rows: list[dict],
    *,
    expected_count: int,
    location_manifest: dict | None = None,
) -> dict:
    qualification = qualify_validators()
    audit = validate_candidate_mechanically(
        rows,
        location_manifest=location_manifest,
    )

    reasons: list[str] = []
    if len(rows) != expected_count:
        reasons.append("row_count_mismatch")
    if qualification["status"] != "VALIDATORS_QUALIFIED":
        reasons.append("validator_qualification_hold")
    nonvalid = {
        status: count
        for status, count in audit["status_counts"].items()
        if status != "MECHANICAL_VALID"
    }
    if nonvalid:
        reasons.append("nonvalid_mechanical_rows")

    status = (
        "MECHANICAL_VALIDATED"
        if not reasons
        else "MECHANICAL_HOLD"
    )
    return {
        "schema": "RETENTION_AUDIT_ARCH_V2_MECHANICAL_RECEIPT_V1",
        "status": status,
        "expected_count": expected_count,
        "reviewed": audit["reviewed"],
        "row_status_counts": audit["status_counts"],
        "reasons": reasons,
        "qualification": qualification,
        "rows": audit["rows"],
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1500)
    args = parser.parse_args()

    raw = args.candidate.read_bytes()
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    location_manifest = load_pinned_location_manifest()
    receipt = run_mechanical_audit(
        rows,
        expected_count=args.expected_count,
        location_manifest=location_manifest,
    )
    receipt["candidate_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": receipt["status"],
        "reviewed": receipt["reviewed"],
        "row_status_counts": receipt["row_status_counts"],
        "reasons": receipt["reasons"],
        "candidate_sha256": receipt["candidate_sha256"],
        "output": args.output.as_posix(),
    }, sort_keys=True))
    return 0 if receipt["status"] == "MECHANICAL_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
