from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Callable

from successor.experiments.retention_audit_arch_v2_reviewer import review_prompt
from successor.experiments.retention_audit_arch_v5_reviewer import (
    ollama_structured_text_v5,
    semantic_response_schema_v5,
    sha256_json,
)
from successor.experiments.run_retention_audit_arch_v4_semantic import (
    parse_witness_response_v4,
)


def review_prompt_v6(cases: list[dict]) -> str:
    base = review_prompt(cases)
    marker = "SUBJECTS="
    if marker not in base:
        raise ValueError("V6 semantic prompt anchor missing")
    rule = (
        "For DERIVED_ANSWER, derived_answer must contain only the bare answer "
        "requested by the case prompt, in the prompt's requested output form, "
        "with no explanation or surrounding prose. For all other observations, "
        "derived_answer may be null. "
    )
    return base.replace(marker, rule + marker, 1)


def run_semantic_audit_v6(
    cases: list[dict],
    *,
    call_structured: Callable[[str, dict], str] = ollama_structured_text_v5,
    expected_count: int,
    max_attempts: int = 3,
    batch_size: int = 1,
) -> dict:
    if max_attempts < 1:
        raise ValueError("V6 max_attempts must be positive")
    if batch_size < 1:
        raise ValueError("V6 batch_size must be positive")
    if len(cases) != expected_count:
        return {
            "schema": "RETENTION_AUDIT_ARCH_V6_SEMANTIC_RECEIPT_V1",
            "status": "SEMANTIC_AUDIT_HOLD",
            "reviewed": 0,
            "expected_count": expected_count,
            "observation_counts": {},
            "reasons": ["row_count_mismatch"],
            "rows": [],
        }

    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("V6 semantic case_id invalid")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("V6 semantic case IDs must be unique")

    rows: list[dict] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        batch_ids = [case["case_id"] for case in batch]
        prompt = review_prompt_v6(batch)
        schema = semantic_response_schema_v5(batch_ids)
        schema_sha256 = sha256_json(schema)
        failures: list[dict] = []
        parsed: list[dict] | None = None
        attempts = 0

        for attempts in range(1, max_attempts + 1):
            response_text = ""
            try:
                response_text = call_structured(prompt, schema)
                if (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                ):
                    raise ValueError("empty V6 semantic response")
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
                "schema": "RETENTION_AUDIT_ARCH_V6_SEMANTIC_RECEIPT_V1",
                "status": "SEMANTIC_AUDIT_HOLD",
                "reviewed": len(rows),
                "expected_count": expected_count,
                "observation_counts": dict(
                    sorted(Counter(row["observation"] for row in rows).items())
                ),
                "reasons": [f"transport_failure:{batch_ids[0]}"],
                "rows": rows,
                "response_schema_sha256": schema_sha256,
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
                "response_schema_sha256": schema_sha256,
                "prior_attempt_failures": failures,
            })

    counts = Counter(row["observation"] for row in rows)
    violations = [
        row["case_id"] for row in rows
        if not row["witness_contract_valid"]
    ]
    return {
        "schema": "RETENTION_AUDIT_ARCH_V6_SEMANTIC_RECEIPT_V1",
        "status": "SEMANTIC_AUDIT_COMPLETE",
        "reviewed": len(rows),
        "expected_count": expected_count,
        "batch_size": batch_size,
        "observation_counts": dict(sorted(counts.items())),
        "witness_contract_violations": violations,
        "reasons": [],
        "rows": rows,
    }
