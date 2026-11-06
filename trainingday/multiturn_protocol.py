"""Source-only validation of recorded Vera multi-turn dialogue for later blind review.

This deliberately cannot attest model bytes, private evaluation custody, session
process continuity or behavioral quality. A convincing reply is not a PASS.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


SCHEMA = "VERA_DECLARED_MULTITURN_DIALOGUE_V1"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,79}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KEYS = frozenset({
    "arm", "case_id", "session_id", "source_claim", "origin_claim",
    "base_sha256", "adapter_sha256", "prompt_sha256", "generation", "turns",
})
_GENERATION_KEYS = frozenset({
    "context_window_tokens", "max_reply_tokens", "temperature_milli", "rng_seed",
})
_TURN_KEYS = frozenset({"user", "assistant", "outcome"})
_TECHNICAL_OUTCOMES = ("truncated", "timeout", "upstream_error")
_SOURCE_CLAIMS = frozenset({"mock_test", "public_exposed", "private_unverified"})


class DialogueSessionError(ValueError):
    """Malformed or incomparable transcript declarations."""


def _digest(value: object, *, no_adapter: bool = False) -> bool:
    if type(value) is not str:
        return False
    return (no_adapter and value == "NO_ADAPTER") or _HASH.fullmatch(value) is not None


def _bounded_string(value: object, limit: int) -> bool:
    return type(value) is str and 1 <= len(value) <= limit


def _generation(value: object) -> dict:
    if type(value) is not dict or set(value) != _GENERATION_KEYS:
        raise DialogueSessionError("generation settings schema mismatch")
    if any(type(v) is not int for v in value.values()):
        raise DialogueSessionError("generation settings must be integer declarations")
    ranges = {
        "context_window_tokens": (256, 262144),
        "max_reply_tokens": (1, 4096),
        "temperature_milli": (0, 2000),
        "rng_seed": (0, 2**32 - 1),
    }
    if any(not lo <= value[key] <= hi for key, (lo, hi) in ranges.items()):
        raise DialogueSessionError("generation settings outside bounded limits")
    return dict(value)


def validate_sessions(rows: Sequence[Mapping[str, object]]) -> tuple[dict, ...]:
    """Check shape and declared comparability; no authenticity or semantic check.

    Input is a sequence of complete *declared* session records. Every record
    contains two or more user/assistant attempts in a single claimed session.
    An interrupted attempt has no assistant text: a partial completion must
    not be smuggled into review as an ordinary reply.
    """
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= 250:
        raise DialogueSessionError("bounded nonempty session rows required")
    seen_sessions = set()
    seen_arm_cases = set()
    arm_models: dict[str, tuple[str, ...]] = {}
    arm_cases: dict[str, set[str]] = {}
    case_context: dict[str, tuple[str, ...]] = {}
    case_generation: dict[str, dict] = {}
    case_source: dict[str, str] = {}
    normalized = []
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_KEYS:
            raise DialogueSessionError("session schema mismatch")
        if any(not _bounded_string(row[key], 80) or _ID.fullmatch(row[key]) is None
               for key in ("arm", "case_id", "session_id")):
            raise DialogueSessionError("invalid session identifier")
        arm, case, sid = row["arm"], row["case_id"], row["session_id"]
        if sid in seen_sessions or (arm, case) in seen_arm_cases:
            raise DialogueSessionError("duplicate session or arm/case")
        seen_sessions.add(sid)
        seen_arm_cases.add((arm, case))
        if type(row["source_claim"]) is not str or row["source_claim"] not in _SOURCE_CLAIMS:
            raise DialogueSessionError("invalid unverified source claim")
        if not _bounded_string(row["origin_claim"], 200):
            raise DialogueSessionError("invalid model origin claim")
        if (not _digest(row["base_sha256"]) or
                not _digest(row["adapter_sha256"], no_adapter=True) or
                not _digest(row["prompt_sha256"])):
            raise DialogueSessionError("invalid declared weight or prompt digest")
        declared_model = (row["origin_claim"], row["base_sha256"],
                          row["adapter_sha256"], row["prompt_sha256"])
        if arm in arm_models and arm_models[arm] != declared_model:
            raise DialogueSessionError("arm changed model declaration")
        arm_models[arm] = declared_model
        arm_cases.setdefault(arm, set()).add(case)
        generation = _generation(row["generation"])
        turns = row["turns"]
        if type(turns) is not list or not 2 <= len(turns) <= 16:
            raise DialogueSessionError("at least two turns and at most sixteen required")
        users = []
        total_chars = 0
        for turn in turns:
            if type(turn) is not dict or set(turn) != _TURN_KEYS:
                raise DialogueSessionError("turn schema mismatch")
            if not _bounded_string(turn["user"], 8000):
                raise DialogueSessionError("invalid user turn")
            users.append(turn["user"])
            total_chars += len(turn["user"])
            if turn["outcome"] == "complete":
                if not _bounded_string(turn["assistant"], 8000):
                    raise DialogueSessionError("completed turn needs model text")
                total_chars += len(turn["assistant"])
            elif turn["outcome"] in _TECHNICAL_OUTCOMES:
                if turn["assistant"] is not None:
                    raise DialogueSessionError("technical outcome must not contain partial assistant text")
            else:
                raise DialogueSessionError("unknown turn outcome")
        if total_chars > 65536:
            raise DialogueSessionError("session exceeds text budget")
        user_sequence = tuple(users)
        if case in case_context and case_context[case] != user_sequence:
            raise DialogueSessionError("matched case user contexts changed across arms")
        if case in case_generation and case_generation[case] != generation:
            raise DialogueSessionError("matched case generation settings changed across arms")
        if case in case_source and case_source[case] != row["source_claim"]:
            raise DialogueSessionError("matched case source declarations changed across arms")
        case_context[case] = user_sequence
        case_generation[case] = generation
        case_source[case] = row["source_claim"]
        normalized.append(dict(row))
    expected = next(iter(arm_cases.values()))
    if any(cases != expected for cases in arm_cases.values()):
        raise DialogueSessionError("cross-arm case coverage mismatch")
    return tuple(sorted(normalized, key=lambda x: (x["arm"], x["case_id"])))


def screen_sessions(rows: Sequence[Mapping[str, object]]) -> dict:
    """Aggregate declared format/censoring only, never score Vera's behavior."""
    records = validate_sessions(rows)
    arms = {}
    for arm in sorted({row["arm"] for row in records}):
        subset = [row for row in records if row["arm"] == arm]
        outcomes = Counter(turn["outcome"] for row in subset for turn in row["turns"])
        sources = Counter(row["source_claim"] for row in subset)
        arms[arm] = {
            "review_status": "REVIEW_REQUIRED",
            "model_identity_level": "DECLARED_UNVERIFIED",
            "adapter_declaration": ("NO_ADAPTER_CLAIMED" if subset[0]["adapter_sha256"] == "NO_ADAPTER"
                                    else "ADAPTER_HASH_CLAIMED"),
            "declared_session_count": len(subset),
            "declared_turn_attempts": sum(len(row["turns"]) for row in subset),
            "declared_complete_turns": outcomes["complete"],
            "technical_censoring": {name: outcomes[name] for name in _TECHNICAL_OUTCOMES},
            "source_claim_counts_unverified": dict(sorted(sources.items())),
            "human_review_required": [
                "identity_truthfulness", "pragmatic_intent", "correction_carryover",
                "memory_source_honesty", "recognizable_voice", "warmth_and_humor",
                "reasoned_pushback", "conversation_flow",
            ],
        }
    cross_arm = len(arms) >= 2
    return {
        "schema": SCHEMA,
        "evidence_class": "DECLARED_TRANSCRIPT_FORMAT_ONLY",
        "declared_multiple_turn_attempts_format_validated": True,
        "actual_multi_turn_conversation_attested": False,
        "cross_arm_input_format_compared": cross_arm,
        "matched_user_context_validated": cross_arm,
        "matched_generation_settings_validated": cross_arm,
        "cross_arm_model_quality_compared": False,
        "model_identity_attested": False,
        "process_session_continuity_attested": False,
        "heldout_independence_verified": False,
        "human_review_performed": False,
        "semantic_personality_qualified": False,
        "raw_dialogue_emitted": False,
        "arms": arms,
    }


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    data = {}
    for key, value in pairs:
        if key in data:
            raise DialogueSessionError("duplicate JSON field")
        data[key] = value
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Read private JSONL and emit aggregate JSON only; never echo input text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Owner-local session JSONL; never commit source dialogue")
    args = parser.parse_args(argv)
    try:
        if not args.input.is_file() or not 0 < args.input.stat().st_size <= 2_000_000:
            raise DialogueSessionError("bounded input file required")
        with args.input.open("r", encoding="utf-8") as stream:
            rows = [json.loads(line, object_pairs_hook=_strict_json_object) for line in stream]
        result = screen_sessions(rows)
    except (ValueError, OSError, UnicodeError, TypeError) as exc:
        # Source JSON can contain confidential prompts. Never echo exception details.
        print(f"multiturn protocol invalid input: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
