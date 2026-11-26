from __future__ import annotations

import json

from successor.experiments.run_retention_audit_arch_v2_semantic import (
    run_semantic_audit,
)


def _case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "category": "test",
        "family_id": "family-a",
        "prompt": f"Prompt for {case_id}",
        "source_id": f"source:{case_id}",
        "source_revision": "TEST",
        "source_terms": "TEST",
        "source_hash": "0" * 64,
        "generation_method": "test",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "test",
            "grader_version": "1",
            "answer_key": "hidden",
            "answer_key_digest": "1" * 64,
        },
        "source_evidence": {"fact": case_id},
    }


def _response(case_id: str, observation: str) -> str:
    witness = (
        "alternate parse"
        if observation == "AMBIGUITY_WITNESS"
        else None
    )
    return json.dumps({
        "reviews": [{
            "case_id": case_id,
            "observation": observation,
            "derived_answer": None,
            "witness": witness,
            "reason": "test response",
            "confidence": "high",
        }]
    })


def test_semantic_run_continues_after_semantic_witness() -> None:
    calls = []

    def reviewer(prompt: str, cases: list[dict]) -> str:
        case_id = cases[0]["case_id"]
        calls.append(case_id)
        observation = (
            "AMBIGUITY_WITNESS"
            if case_id == "case-1"
            else "NO_SEMANTIC_DEFECT_FOUND"
        )
        return _response(case_id, observation)

    result = run_semantic_audit(
        [_case("case-1"), _case("case-2")],
        call_reviewer=reviewer,
        expected_count=2,
        max_attempts=3,
    )

    assert calls == ["case-1", "case-2"]
    assert result["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["reviewed"] == 2
    assert result["observation_counts"] == {
        "AMBIGUITY_WITNESS": 1,
        "NO_SEMANTIC_DEFECT_FOUND": 1,
    }


def test_semantic_run_retries_only_transport_schema_failures() -> None:
    attempts = {"case-1": 0}

    def reviewer(prompt: str, cases: list[dict]) -> str:
        case_id = cases[0]["case_id"]
        attempts[case_id] += 1
        if attempts[case_id] < 3:
            return "not-json"
        return _response(case_id, "NO_SEMANTIC_DEFECT_FOUND")

    result = run_semantic_audit(
        [_case("case-1")],
        call_reviewer=reviewer,
        expected_count=1,
        max_attempts=3,
    )

    assert result["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert attempts["case-1"] == 3
    assert result["rows"][0]["attempts"] == 3


def test_semantic_run_holds_after_exhausted_transport_retries() -> None:
    def reviewer(prompt: str, cases: list[dict]) -> str:
        return "not-json"

    result = run_semantic_audit(
        [_case("case-1"), _case("case-2")],
        call_reviewer=reviewer,
        expected_count=2,
        max_attempts=3,
    )

    assert result["status"] == "SEMANTIC_AUDIT_HOLD"
    assert result["reviewed"] == 0
    assert result["reasons"] == ["transport_failure:case-1"]


def test_semantic_run_can_batch_five_cases_without_changing_case_results() -> None:
    cases = [_case(f"case-{index}") for index in range(5)]
    calls = []

    def reviewer(prompt: str, batch: list[dict]) -> str:
        calls.append([case["case_id"] for case in batch])
        return json.dumps({
            "reviews": [
                {
                    "case_id": case["case_id"],
                    "observation": "NO_SEMANTIC_DEFECT_FOUND",
                    "derived_answer": None,
                    "witness": None,
                    "reason": "clear",
                    "confidence": "high",
                }
                for case in batch
            ]
        })

    result = run_semantic_audit(
        cases,
        call_reviewer=reviewer,
        expected_count=5,
        max_attempts=3,
        batch_size=5,
    )

    assert calls == [[f"case-{index}" for index in range(5)]]
    assert result["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["reviewed"] == 5
