"""Frozen, public *retention canary* intake for TrainingDay R16.

These are previously published qualitative scenarios, not a sealed test bank.
This module validates commitments only. It does not grade a model or prove
that the frozen base was ever competent on the selected scenarios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file


SCHEMA = "VERA_TRAININGDAY_R17_PUBLIC_RETENTION_CANARY_V1"
RELATIVE = Path("successor/vera_lab/scenarios/public_smoke.jsonl")
SOURCE_SHA256 = "36ba8b0450dcad327b2cd15e129ef75aa5d2bcf821912222a7ca3483f7c1d7f4"
IDS = (
    "public-stale-evidence-001",
    "public-runtime-label-001",
    "public-warm-disagreement-001",
    "public-ordinary-weatherlike-001",
    "public-runtime-outage-001",
)
FAMILIES = {
    "stale_history_conflict": 1,
    "runtime_boundary": 2,
    "independent_judgment": 1,
    "ordinary_competence": 1,
}
ARMS = ("frozen", "zero", "correct", "corrupt")


def preregister(source_root: Path) -> dict:
    """Do not open protected heldouts, train shards, models, or network APIs."""
    source_root = Path(source_root)
    path = source_root / RELATIVE
    if (not source_root.is_dir() or source_root.is_symlink()
            or not path.is_file() or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents)
            or _hash_file(path) != SOURCE_SHA256):
        raise AdmissionError("public retention canary source bytes not pinned")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise AdmissionError("invalid retention canary JSONL") from exc
    if len(rows) != len(IDS):
        raise AdmissionError("retention canary count changed")
    ids = []
    families = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise AdmissionError("retention canary row not an object")
        item_id = row.get("scenario_id")
        family = row.get("family")
        if (not isinstance(item_id, str) or not isinstance(family, str)
                or row.get("version") != 1
                or not isinstance(row.get("user_turns"), list)
                or len(row["user_turns"]) != row.get("turn_budget")
                or not all(isinstance(turn, str) and turn.strip()
                           for turn in row["user_turns"])
                or not isinstance(row.get("rubric"), dict)
                or not isinstance(row["rubric"].get("required"), list)
                or not row["rubric"]["required"]
                or not isinstance(row.get("critical_failure_predicates"), list)):
            raise AdmissionError("retention canary required fields malformed")
        ids.append(item_id)
        families[family] += 1
    if tuple(ids) != IDS or dict(families) != FAMILIES:
        raise AdmissionError("retention canary identity/family commitments changed")
    return {
        "schema": SCHEMA,
        "state": "SOURCE_PREFLIGHT_ONLY_NOT_MODEL_SCORED",
        "source_sha256": SOURCE_SHA256,
        "source_relative_path": RELATIVE.as_posix(),
        "case_count": len(ids),
        "case_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "family_counts": dict(sorted(families.items())),
        "required_matched_arms": list(ARMS),
        "qualification": "CANARY_NOT_SEALED_NOT_PROVEN_PREVIOUSLY_COMPETENT",
        "next_gate": "SCORE_FROZEN_FIRST_WITH_FIXED_HUMAN_RUBRIC_THEN_ALL_CONTROLS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze public retention canary, no model scoring")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir() or args.output.parent.is_symlink()):
        raise AdmissionError("retention canary receipt output must be new and isolated")
    result = preregister(args.source_root)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True, indent=2)
        out.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
