from __future__ import annotations

import json

from successor.experiments.run_retention_audit_arch_v5_semantic import (
    run_semantic_audit_v5,
)


def _case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "category": "test",
        "family_id": "family-a",
        "prompt": f"Prompt {case_id}",
        "source_id": f"source:{case_id}",
        "source_revision": "TEST",
        "source_terms": "TEST",
        "source_hash": "0" * 64,
        "generation_method": "test",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "test",
            "grader_version": "1",
        },
        "source_evidence": {"fact": case_id},
    }


def test_v5_semantic_runner_uses_structured_schema_and_preserves_invalid_witness() -> None:
    cases = [_case("case-1"), _case("case-2")]
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        calls.append((prompt, schema))
        return json.dumps({
            "reviews": [
                {
                    "case_id": "case-1",
                    "observation": "AMBIGUITY_WITNESS",
                    "derived_answer": None,
                    "witness": None,
                    "reason": "claims ambiguity without witness",
                    "confidence": "high",
                },
                {
                    "case_id": "case-2",
                    "observation": "NO_SEMANTIC_DEFECT_FOUND",
                    "derived_answer": None,
                    "witness": None,
                    "reason": "clear",
                    "confidence": "high",
                },
            ]
        })

    result = run_semantic_audit_v5(
        cases,
        call_structured=call_structured,
        expected_count=2,
        batch_size=2,
        max_attempts=3,
    )

    assert result["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["reviewed"] == 2
    assert len(calls) == 1
    assert calls[0][1]["additionalProperties"] is False
    assert result["rows"][0]["witness_contract_valid"] is False
    assert result["rows"][0]["attempts"] == 1
    assert len(result["rows"][0]["response_schema_sha256"]) == 64


def test_v5_semantic_runner_retries_only_unscorable_structure() -> None:
    cases = [_case("case-1")]
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        calls.append((prompt, schema))
        return '{"reviews": []}'

    result = run_semantic_audit_v5(
        cases,
        call_structured=call_structured,
        expected_count=1,
        batch_size=1,
        max_attempts=3,
    )

    assert result["status"] == "SEMANTIC_AUDIT_HOLD"
    assert result["reviewed"] == 0
    assert len(calls) == 3
    assert result["reasons"] == ["transport_failure:case-1"]
    assert all(
        len(item["response_sha256"]) == 64
        for item in result["transport_failures"]
    )
