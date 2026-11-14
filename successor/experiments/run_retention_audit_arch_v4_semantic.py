from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Callable

from successor.experiments.retention_audit_arch_v2_reviewer import (
    ALLOWED_OBSERVATIONS,
    review_prompt,
)
from successor.experiments.retention_audit_arch_v4_reviewer import (
    meaningful_witness_v4,
    witness_text_v4,
)


_WITNESS_REQUIRED = {
    "AMBIGUITY_WITNESS",
    "CONTRACT_COUNTEREXAMPLE",
}


def parse_witness_response_v4(
    text: str,
    *,
    expected_case_ids: list[str],
) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid V4 semantic JSON:{exc}") from exc

    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("V4 semantic response must contain only reviews")
    reviews = value["reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(expected_case_ids):
        raise ValueError("V4 semantic review count mismatch")

    expected = list(expected_case_ids)
    seen: set[str] = set()
    parsed: list[dict] = []
    required = {
        "case_id",
        "observation",
        "derived_answer",
        "witness",
        "reason",
        "confidence",
    }

    for review in reviews:
        if not isinstance(review, dict) or set(review) != required:
            raise ValueError("V4 semantic review schema mismatch")
        case_id = review["case_id"]
        if case_id not in expected or case_id in seen:
            raise ValueError("V4 semantic case_id mismatch or duplicate")
        seen.add(case_id)

        observation = review["observation"]
        if observation not in ALLOWED_OBSERVATIONS:
            raise ValueError("V4 semantic observation is not allowed")
        if not isinstance(review["reason"], str) or not review["reason"].strip():
            raise ValueError("V4 semantic reason must be non-empty")
        if review["confidence"] not in {"low", "medium", "high"}:
            raise ValueError("V4 semantic confidence invalid")

        witness = review["witness"]
        witness_required = observation in _WITNESS_REQUIRED
        witness_valid = (
            meaningful_witness_v4(witness)
            if witness_required
            else True
        )
        row = dict(review)
        row["witness_text"] = witness_text_v4(witness)
        row["witness_contract_valid"] = witness_valid
        parsed.append(row)

    if seen != set(expected):
        raise ValueError("V4 semantic missing expected case_id")
    order = {case_id: index for index, case_id in enumerate(expected)}
    parsed.sort(key=lambda item: order[item["case_id"]])
    return parsed


def run_semantic_audit_v4(
    cases: list[dict],
    *,
    call_reviewer: Callable[[str, list[dict]], str],
    expected_count: int,
    max_attempts: int = 3,
    batch_size: int = 1,
) -> dict:
    if max_attempts < 1:
        raise ValueError("V4 max_attempts must be positive")
    if batch_size < 1:
        raise ValueError("V4 batch_size must be positive")
    if len(cases) != expected_count:
        return {
            "schema": "RETENTION_AUDIT_ARCH_V4_SEMANTIC_RECEIPT_V1",
            "status": "SEMANTIC_AUDIT_HOLD",
            "reviewed": 0,
            "expected_count": expected_count,
            "observation_counts": {},
            "reasons": ["row_count_mismatch"],
            "rows": [],
        }

    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("V4 semantic case_id invalid")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("V4 semantic case IDs must be unique")

    rows: list[dict] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        batch_ids = [case["case_id"] for case in batch]
        prompt = review_prompt(batch)
        failures: list[dict] = []
        parsed: list[dict] | None = None
        attempts = 0

        for attempts in range(1, max_attempts + 1):
            response_text = ""
            try:
                response_text = call_reviewer(prompt, batch)
                if (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                ):
                    raise ValueError("empty V4 semantic response")
                parsed = parse_witness_response_v4(
                    response_text,
                    expected_case_ids=batch_ids,
                )
                break
            except (ValueError, json.JSONDecodeError) as exc:
                raw = (
                    response_text.encode("utf-8")
                    if isinstance(response_text, str)
                    else b""
                )
                failures.append({
                    "attempt": attempts,
                    "error": f"{type(exc).__name__}:{exc}",
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                })

        if parsed is None:
            return {
                "schema": "RETENTION_AUDIT_ARCH_V4_SEMANTIC_RECEIPT_V1",
                "status": "SEMANTIC_AUDIT_HOLD",
                "reviewed": len(rows),
                "expected_count": expected_count,
                "observation_counts": dict(
                    sorted(
                        Counter(
                            row["observation"] for row in rows
                        ).items()
                    )
                ),
                "reasons": [f"transport_failure:{batch_ids[0]}"],
                "rows": rows,
                "transport_failures": failures,
                "failed_batch_case_ids": batch_ids,
            }

        by_id = {case["case_id"]: case for case in batch}
        for review in parsed:
            case_id = review["case_id"]
            case = by_id[case_id]
            rows.append({
                "case_id": case_id,
                "family_id": case.get("family_id"),
                "observation": review["observation"],
                "derived_answer": review["derived_answer"],
                "witness": review["witness"],
                "witness_text": review["witness_text"],
                "witness_contract_valid": review["witness_contract_valid"],
                "reason": review["reason"],
                "confidence": review["confidence"],
                "attempts": attempts,
                "prior_attempt_failures": failures,
            })

    counts = Counter(row["observation"] for row in rows)
    witness_contract_violations = [
        row["case_id"]
        for row in rows
        if not row["witness_contract_valid"]
    ]
    return {
        "schema": "RETENTION_AUDIT_ARCH_V4_SEMANTIC_RECEIPT_V1",
        "status": "SEMANTIC_AUDIT_COMPLETE",
        "reviewed": len(rows),
        "expected_count": expected_count,
        "batch_size": batch_size,
        "observation_counts": dict(sorted(counts.items())),
        "witness_contract_violations": witness_contract_violations,
        "reasons": [],
        "rows": rows,
    }
