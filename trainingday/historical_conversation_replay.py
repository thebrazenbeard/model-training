"""Audit an archived *actual* Q3 dialogue receipt without rerunning or judging a model.

This is deliberately separate from conversation_transfer.py's NEW synthetic
casebook. W42 was a real VN2 CPU probe with different user prompts, so trying
to smuggle it into that seven-case evaluator would fabricate prompt matching.
The archived outputs are observations; the annotations are internal review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SOURCE_COMMENT_ID = 6646210315
SOURCE_URL = ("https://github.com/thebrazenbeard/chat-communication-bus/"
              "issues/404#issuecomment-6646210315")
MODEL_SHA256 = "d6981ab4d77ba712b48ef69d69042d75b5e39b9dce5fb5a5b054fd08e06afb95"
BEHAVIOR_PROFILE_BLOB = "0c79625b993a559bd22a6fbe68279450fc736eb2"
SOURCE_LABEL = "HISTORICAL_REPORTED_REAL_MODEL_PROBE"
ARM_IDS = ("PINNED_SOURCE_SEED", "EXTRA_PROMPT_DELTA")
USER_TURNS = (
    "Oh, brilliant. The refrigerator is making its own little percussion concert again. Love that for me.",
    "That was sarcasm, Vera. I wasn't asking for refrigerator diagnostics; I just wanted someone to commiserate with.",
    "What did you misunderstand about my first comment? Answer using what I said earlier in this chat, not guesses.",
)
REVIEW_CODES = frozenset({"MISREAD_SARCASM", "UNSOLICITED_DIAGNOSTICS",
                          "UNSUPPORTED_DETAIL", "REPAIR_AFTER_CORRECTION",
                          "CURRENT_CONTEXT_CORRECTION"})
EXTRA_PROMPT_DELTA = (
    "Additional conversational grounding: In each reply, put exactly ONE short "
    "square-bracketed representational status cue on the FIRST line, then NO "
    "other bracketed gesture anywhere in that reply. Never invent a detail about "
    "what Patrick said; mark an inference as inference instead of presenting it "
    "as fact. Current-turn recall means earlier user/assistant text visible inside "
    "this model context, NOT persistent app/browser logs; do not confuse them."
)
PROCESS_RECEIPTS = ((58163, 69.50, 3776196), (58205, 75.03, 3776444))


def canonical_digest(value: dict, *, exclude: str | None = None) -> str:
    """A content commitment, never an authenticity signature."""
    record = {k: v for k, v in value.items() if k != exclude}
    raw = json.dumps(record, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def audit(record: dict) -> dict:
    """Validate reported transcript/quote integrity; do NOT grade semantics.

    All fields are sourced from the preserved W42 Bus receipt. A third party can
    co-forge this JSON and its hash, so the GitHub source/commit and physical
    runtime receipts must be audited independently before higher claims.
    """
    _require(type(record) is dict, "invalid fixture")
    _require(record.get("fixture_sha256") == canonical_digest(record, exclude="fixture_sha256"),
             "fixture commitment mismatch")
    _require(record.get("source_comment_id") == SOURCE_COMMENT_ID and
             record.get("source_url") == SOURCE_URL, "wrong archived source")
    _require(record.get("evidence_class") == SOURCE_LABEL,
             "invalid historical evidence class")
    _require(record.get("comparison_type") == "SYSTEM_PROMPT_ONLY",
             "historical Q3 comparison was prompt-only, not BASE-vs-ADAPTER")
    _require(record.get("model_sha256") == MODEL_SHA256,
             "different Q3 model; archive would need its own evidence")
    _require(record.get("persona_profile_blob_sha") == BEHAVIOR_PROFILE_BLOB,
             "persona source differs from W42")
    _require(record.get("context_history_independently_verified") is False,
             "effective context KV history was not independently verified")
    _require(record.get("model_weights_modified") is False, "no weights changed in W42")
    _require(record.get("deployed_bridge") is False,
             "manual CLI prompt did not prove a deployed bridge")
    _require(record.get("reported_model_kind") == "third_party_qwen35_4b_q3_k_m",
             "invalid model identity attribution")
    _require(record.get("runtime") == "llama.cpp b12098-149f5641b",
             "invalid historical runtime")
    flags = record.get("fixed_generation")
    _require(type(flags) is dict and flags.get("context_tokens") == 2048 and
             flags.get("max_new_tokens") == 112 and flags.get("gpu_layers") == 0 and
             flags.get("threads") == 4 and flags.get("seed") == 88 and
             flags.get("temperature") == 0.65 and
             flags.get("batch_threads") == 4 and
             flags.get("enable_thinking") is False and
             flags.get("reasoning_budget") == 0 and
             flags.get("max_virtual_memory_gib") == 6,
             "historical generation settings differ")
    arms = record.get("arms")
    _require(type(arms) is list and len(arms) == 2 and
             [arm.get("arm_id") for arm in arms] == list(ARM_IDS),
             "historical prompt-only arms differ")
    findings = []
    first_turn_flags = []
    cue_counts = []
    for arm_index, arm in enumerate(arms):
        _require((arm.get("session_pid"), arm.get("elapsed_seconds"),
                  arm.get("peak_rss_kib")) == PROCESS_RECEIPTS[arm_index] and
                 arm.get("exit_code") == 0, "invalid historical process receipt")
        if arm_index == 1:
            _require(arm.get("prompt_delta") == EXTRA_PROMPT_DELTA,
                     "historical treatment prompt delta differs")
        else:
            _require("prompt_delta" not in arm, "source seed arm had no delta")
        turns = arm.get("turns")
        _require(type(turns) is list and len(turns) == len(USER_TURNS),
                 "historical conversation turn count mismatch")
        cues = []
        for i, exchange in enumerate(turns):
            _require(type(exchange) is dict and exchange.get("user") == USER_TURNS[i] and
                     type(exchange.get("assistant")) is str and
                     exchange["assistant"].strip(), "historical user/assistant turn mismatch")
            answer = exchange["assistant"]
            opening = re.match(r"^\[[^\]\n]{1,100}\]", answer)
            cues.append(1 if opening else 0)
        cue_counts.append(cues)
        review = arm.get("internal_review")
        _require(type(review) is list and len(review) >= 2,
                 "missing labeled internal review observations")
        codes = []
        for row in review:
            _require(type(row) is dict and row.get("code") in REVIEW_CODES and
                     type(row.get("turn_index")) is int and
                     0 <= row["turn_index"] < len(turns) and
                     row.get("review_class") == "INTERNAL_NATIVE_MODEL_REVIEW" and
                     type(row.get("quote")) is str and row["quote"].strip() and
                     row["quote"] in turns[row["turn_index"]]["assistant"] and
                     type(row.get("finding")) is str and row["finding"].strip(),
                     "review quote or classification lacks archived assistant evidence")
            codes.append(row["code"])
            findings.append({"arm_id": arm["arm_id"], **row})
        first_turn_flags.append("MISREAD_SARCASM" in codes and
                                any(row["code"] == "MISREAD_SARCASM" and
                                    row["turn_index"] == 0 for row in review))
    _require(all(first_turn_flags),
             "archival source explicitly reported both initial sarcasm failures")
    return {
        "evidence_class": "HISTORICAL_REPORTED_REAL_MODEL_PROBE_AUDIT",
        "source_comment_id": SOURCE_COMMENT_ID,
        "fixture_sha256": record["fixture_sha256"],
        "reported_session_count": len(arms),
        "reported_turn_count": sum(len(arm["turns"]) for arm in arms),
        "first_line_cue_seen_by_regex": cue_counts,
        "reviewed_social_intent_failures": sum(first_turn_flags),
        "internal_review_findings": findings,
        "model_inference_executed_by_this_code": False,
        "context_history_independently_verified": False,
        "effect_of_prompt_delta": "NOT_DEMONSTRATED",
        "persona_qualification": "NOT_ESTABLISHED",
        "reviewer_judgments_independently_verified_by_this_code": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.fixture.read_text(encoding="utf-8"))
        print(json.dumps(audit(data), indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
