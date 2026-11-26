from __future__ import annotations

import json

from successor.experiments.run_retention_audit_arch_v4_semantic import (
    parse_witness_response_v4,
    run_semantic_audit_v4,
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


def test_v4_semantic_parser_accepts_structured_witness() -> None:
    raw = json.dumps({
        "reviews": [
            {
                "case_id": "case-1",
                "observation": "AMBIGUITY_WITNESS",
                "derived_answer": None,
                "witness": {
                    "reading_a": "A",
                    "reading_b": "B",
                },
                "reason": "two readings",
                "confidence": "high",
            },
            {
                "case_id": "case-2",
                "observation": "CONTRACT_COUNTEREXAMPLE",
                "derived_answer": None,
                "witness": ["output A", "output B"],
                "reason": "two outputs satisfy prompt",
                "confidence": "medium",
            },
        ]
    })

    parsed = parse_witness_response_v4(
        raw,
        expected_case_ids=["case-1", "case-2"],
    )

    assert parsed[0]["witness_contract_valid"] is True
    assert parsed[0]["witness_text"] == (
        '{"reading_a":"A","reading_b":"B"}'
    )
    assert parsed[1]["witness_contract_valid"] is True
    assert parsed[1]["witness_text"] == '["output A","output B"]'


def test_v4_semantic_invalid_required_witness_is_scoreable_not_transport() -> None:
    raw = json.dumps({
        "reviews": [
            {
                "case_id": "case-1",
                "observation": "AMBIGUITY_WITNESS",
                "derived_answer": None,
                "witness": None,
                "reason": "claims ambiguity but gives no witness",
                "confidence": "high",
            }
        ]
    })

    parsed = parse_witness_response_v4(
        raw,
        expected_case_ids=["case-1"],
    )

    assert parsed[0]["witness_contract_valid"] is False
    assert parsed[0]["witness_text"] == "null"


def test_v4_semantic_runner_does_not_retry_scoreable_witness_violation() -> None:
    cases = [_case("case-1"), _case("case-2")]
    calls = []

    def reviewer(prompt: str, batch: list[dict]) -> str:
        calls.append(prompt)
        return json.dumps({
            "reviews": [
                {
                    "case_id": "case-1",
                    "observation": "AMBIGUITY_WITNESS",
                    "derived_answer": None,
                    "witness": None,
                    "reason": "missing concrete witness",
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

    result = run_semantic_audit_v4(
        cases,
        call_reviewer=reviewer,
        expected_count=2,
        batch_size=2,
        max_attempts=3,
    )

    assert result["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["reviewed"] == 2
    assert len(calls) == 1
    assert result["rows"][0]["witness_contract_valid"] is False
    assert result["rows"][0]["attempts"] == 1


def test_v4_semantic_runner_retries_unscorable_case_set_failure() -> None:
    cases = [_case("case-1")]
    calls = []

    def reviewer(prompt: str, batch: list[dict]) -> str:
        calls.append(prompt)
        return json.dumps({
            "reviews": [
                {
                    "case_id": "wrong-case",
                    "observation": "NO_SEMANTIC_DEFECT_FOUND",
                    "derived_answer": None,
                    "witness": None,
                    "reason": "wrong id",
                    "confidence": "high",
                }
            ]
        })

    result = run_semantic_audit_v4(
        cases,
        call_reviewer=reviewer,
        expected_count=1,
        batch_size=1,
        max_attempts=3,
    )

    assert result["status"] == "SEMANTIC_AUDIT_HOLD"
    assert result["reviewed"] == 0
    assert len(calls) == 3
    assert result["reasons"] == ["transport_failure:case-1"]
