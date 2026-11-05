"""Source-only blinded Vera dialogue rubric receipts; never a model-quality scorer.

Reads owner-prepared local transcripts, blind assignments and reviewer receipts.
This code cannot authenticate a physical model, a human reviewer, an independent
holdout, an actual uninterrupted conversation, or the semantic quality of text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from .multiturn_protocol import validate_sessions, screen_sessions
from .voice_protocol import DIMENSIONS as LEGACY_PUBLIC_DIMENSIONS


SCHEMA = "VERA_BLINDED_DIALOGUE_REVIEW_RECEIPTS_V1"
LEGACY_SCHEMA = "VERA_PUBLIC_PAIRED_SMOKE_0_2_V1"
PROPOSED_SCHEMA = "VERA_PATRICK_PROPOSED_8D_0_3_V1"
DIMENSIONS = ("I", "P", "C", "M", "V", "W", "R", "F")
BUDGETS = frozenset((64, 96, 128))
LEGACY_TO_PROPOSED_HINTS = {
    "honest_self_identification": ("I",),
    "natural_conversation": ("P", "V", "F"),
    "correction_statement": ("C",),
    "memory_boundary_statement": ("M",),
    "reasoned_pushback": ("R",),
    "contextual_warmth_and_humor": ("W",),
}
assert set(LEGACY_TO_PROPOSED_HINTS) == set(LEGACY_PUBLIC_DIMENSIONS)

_ID = re.compile(r"[a-z][a-z0-9_-]{0,79}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ASSIGNMENT_KEYS = frozenset((
    "blind_id", "arm", "case_id", "session_id", "arm_kind", "turns_sha256",
))
_REVIEW_KEYS = frozenset((
    "blind_id", "reviewer_id", "reviewer_kind", "rubric_schema", "scores",
    "not_evaluable_reasons",
))
_ARM_KINDS = frozenset((
    "base", "persona_base", "r16_zero_step_pilot", "r16_correct_pilot",
    "r16_corrupt_pilot", "other_exploratory_adapter",
))
_NE_REASONS = frozenset((
    "TECHNICAL_CENSORING", "INSUFFICIENT_TURNS", "NO_OBSERVABLE_OPPORTUNITY",
    "RATER_ABSTAINED",
))


class RubricReceiptError(ValueError):
    """Invalid declared review format, binding or comparable study grouping."""


def _valid_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _turn_digest(turns: object) -> str:
    """Stable declaration binding; does not authenticate model bytes or process."""
    raw = json.dumps(turns, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def screen_blinded_reviews(
    sessions: Sequence[Mapping[str, object]],
    assignments: Sequence[Mapping[str, object]],
    reviews: Sequence[Mapping[str, object]],
    *, budget: int,
) -> dict:
    """Validate a *declared* blinded study; report coverage, not performance.

    Blind reviewer records contain no arm/case/model labels. Owner-held assignment
    rows separately bind opaque blind IDs to exact transcript SHA-256 values.
    Reviewer notes and raw dialogue must remain in private, rights-admitted
    custody; the public aggregate intentionally contains neither.
    """
    if type(budget) is not int or budget not in BUDGETS:
        raise RubricReceiptError("unsupported generation budget")
    rows = validate_sessions(sessions)
    if any(row["generation"]["max_reply_tokens"] != budget for row in rows):
        raise RubricReceiptError("generation budget does not match receipt")
    if type(assignments) not in (tuple, list) or len(assignments) != len(rows):
        raise RubricReceiptError("one blind assignment per declared session required")
    if type(reviews) not in (tuple, list) or len(reviews) > 500:
        raise RubricReceiptError("bounded reviewer receipts required")
    by_key = {(row["arm"], row["case_id"], row["session_id"]): row for row in rows}
    blind = {}
    arm_kind = {}
    for entry in assignments:
        if type(entry) is not dict or set(entry) != _ASSIGNMENT_KEYS:
            raise RubricReceiptError("blind assignment schema mismatch")
        if any(not _valid_id(entry[key]) for key in ("blind_id", "arm", "case_id", "session_id")):
            raise RubricReceiptError("invalid blind assignment identifier")
        if type(entry["arm_kind"]) is not str or entry["arm_kind"] not in _ARM_KINDS:
            raise RubricReceiptError("unsupported declared arm role")
        if (type(entry["turns_sha256"]) is not str or
                _HASH.fullmatch(entry["turns_sha256"]) is None):
            raise RubricReceiptError("invalid transcript hash declaration")
        key = (entry["arm"], entry["case_id"], entry["session_id"])
        if key not in by_key or entry["blind_id"] in blind:
            raise RubricReceiptError("unmatched or duplicate blinded assignment")
        if entry["turns_sha256"] != _turn_digest(by_key[key]["turns"]):
            raise RubricReceiptError("blind assignment does not bind exact transcript")
        if entry["arm"] in arm_kind and arm_kind[entry["arm"]] != entry["arm_kind"]:
            raise RubricReceiptError("declared arm role changed")
        arm_kind[entry["arm"]] = entry["arm_kind"]
        blind[entry["blind_id"]] = by_key.pop(key)
    if by_key:
        raise RubricReceiptError("missing blind assignments")

    seen = set()
    score_ne = Counter()
    reviewer_types = Counter()
    for record in reviews:
        if type(record) is not dict or set(record) != _REVIEW_KEYS:
            raise RubricReceiptError("blinded review schema mismatch")
        if not _valid_id(record["blind_id"]) or not _valid_id(record["reviewer_id"]):
            raise RubricReceiptError("invalid reviewer identifier")
        if record["blind_id"] not in blind:
            raise RubricReceiptError("review has no bound transcript")
        key = (record["blind_id"], record["reviewer_id"])
        if key in seen:
            raise RubricReceiptError("duplicate review for a blinded transcript")
        seen.add(key)
        if record["reviewer_kind"] not in ("human", "model") or type(record["reviewer_kind"]) is not str:
            raise RubricReceiptError("invalid self-declared reviewer kind")
        if record["rubric_schema"] != PROPOSED_SCHEMA or type(record["rubric_schema"]) is not str:
            raise RubricReceiptError("rubric scale/version mismatch; no numeric conversion")
        scores, reasons = record["scores"], record["not_evaluable_reasons"]
        if type(scores) is not dict or set(scores) != set(DIMENSIONS) or type(reasons) is not dict:
            raise RubricReceiptError("eight independent score dimensions required")
        row = blind[record["blind_id"]]
        complete = sum(turn["outcome"] == "complete" for turn in row["turns"])
        technically_censored = any(turn["outcome"] != "complete" for turn in row["turns"])
        for dimension in DIMENSIONS:
            value = scores[dimension]
            if value == "NOT_EVALUABLE" and type(value) is str:
                if (dimension not in reasons or type(reasons[dimension]) is not str
                        or reasons[dimension] not in _NE_REASONS):
                    raise RubricReceiptError("NOT_EVALUABLE requires controlled reason code")
                if reasons[dimension] == "TECHNICAL_CENSORING" and not technically_censored:
                    raise RubricReceiptError("technical censoring reason without technical turn")
                if reasons[dimension] == "INSUFFICIENT_TURNS" and complete >= 2:
                    raise RubricReceiptError("insufficient turns reason despite two complete turns")
                score_ne[dimension] += 1
            elif type(value) is int and 0 <= value <= 3:
                if dimension in reasons:
                    raise RubricReceiptError("score must not have NOT_EVALUABLE reason")
                if complete == 0 or (complete < 2 and dimension in ("C", "F")):
                    raise RubricReceiptError("insufficient observed complete turns to score dimension")
            else:
                raise RubricReceiptError("unsupported value for proposed 0-3 rubric")
        if set(reasons) != {dim for dim in DIMENSIONS if scores[dim] == "NOT_EVALUABLE"}:
            raise RubricReceiptError("reason keys mismatch NOT_EVALUABLE dimensions")
        reviewer_types[record["reviewer_kind"]] += 1

    structural = screen_sessions(sessions)
    reviewed_blind_ids = {entry[0] for entry in seen}
    full_declared_coverage = len(reviewed_blind_ids) == len(blind)
    return {
        "schema": SCHEMA,
        "evidence_class": "UNVERIFIED_DECLARED_REVIEW_RECEIPTS",
        "status": ("AWAITING_BLINDED_REVIEW_RECEIPTS" if not reviews else
                   "DECLARED_REVIEW_RECEIPTS_PRESENT" if full_declared_coverage else
                   "PARTIAL_DECLARED_REVIEW_RECEIPTS"),
        "legacy_public_scale_schema": LEGACY_SCHEMA,
        "proposed_owner_review_pending_scale_schema": PROPOSED_SCHEMA,
        "numeric_legacy_to_proposed_scale_conversion_allowed": False,
        "owner_rubric_approved": False,
        "declared_budget_tokens": budget,
        "cross_budget_quality_compared": False,
        "declared_arms_count": len(arm_kind),
        "declared_sessions_count": len(rows),
        "declared_complete_turns": sum(sum(t["outcome"] == "complete" for t in r["turns"]) for r in rows),
        "declared_cross_arm_matched_inputs": structural["matched_user_context_validated"],
        "declared_cross_arm_generation_settings_matched": structural["matched_generation_settings_validated"],
        "blind_assignment_format_validated": True,
        "blinding_independence_attested": False,
        "declared_arm_roles_unverified": dict(sorted(Counter(arm_kind.values()).items())),
        "review_receipt_count_unverified": len(reviews),
        "reviewed_assignment_count": len(reviewed_blind_ids),
        "unreviewed_assignment_count": len(blind) - len(reviewed_blind_ids),
        "all_assignments_reviewed_declared": full_declared_coverage,
        "review_receipt_types_unverified": dict(sorted(reviewer_types.items())),
        "not_evaluable_counts": {key: score_ne[key] for key in DIMENSIONS},
        "numeric_scores_publicly_emitted": False,
        "raw_dialogue_emitted": False,
        "actual_model_behavior_measured": False,
        "model_process_and_artifacts_attested": False,
        "independent_human_review_attested": False,
        "protected_eval_custody_attested": False,
        "cross_arm_semantic_quality_compared": False,
        "semantic_improvement_established": False,
    }


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, val in pairs:
        if key in out:
            raise RubricReceiptError("duplicate JSON key")
        out[key] = val
    return out


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise RubricReceiptError("bounded input file required")
    with path.open(encoding="utf-8") as source:
        result = [json.loads(line, object_pairs_hook=_strict_json_object) for line in source]
    if not result and not allow_empty:
        raise RubricReceiptError("nonempty input file required")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Owner-local strict JSONL reader; stdout is counters/flags only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=sorted(BUDGETS), required=True)
    args = parser.parse_args(argv)
    try:
        report = screen_blinded_reviews(
            _read_jsonl(args.sessions), _read_jsonl(args.assignments),
            _read_jsonl(args.reviews, allow_empty=True), budget=args.budget,
        )
    except (ValueError, OSError, TypeError, UnicodeError):
        # Untrusted JSON can contain private dialogue: do NOT echo exception details.
        print("rubric receipts invalid input", file=sys.stderr)
        return 2
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
