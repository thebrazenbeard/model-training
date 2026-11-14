from __future__ import annotations

import json

import pytest

import successor.experiments.run_v10_retention_independent_audit as audit_module
from successor.experiments.run_v10_retention_independent_audit import (
    parse_model_blob_path,
    parse_review,
    review_prompt,
    summarize_reviews,
)


def test_review_prompt_contains_subject_not_prior_verdict() -> None:
    case = {
        "case_id": "ret-x",
        "category": "coding",
        "prompt": "Implement f.",
        "grader_contract": {"kind": "deterministic", "tests": ["assert f()==1"]},
        "source_evidence": None,
    }
    prompt = review_prompt(case)
    assert "ret-x" in prompt
    assert "assert f()==1" in prompt
    assert "PENDING_INDEPENDENT_REVIEW" not in prompt


def test_parse_review_requires_closed_schema() -> None:
    value = parse_review(
        json.dumps({
            "verdict": "ADMIT",
            "prompt_well_posed": True,
            "grader_matches_prompt": True,
            "source_evidence_sufficient": True,
            "issue_code": "NONE",
            "reason": "No material defect.",
            "confidence": "high",
        })
    )
    assert value["verdict"] == "ADMIT"
    with pytest.raises(ValueError):
        parse_review('{"verdict":"MAYBE"}')


def test_summary_refuses_any_reject() -> None:
    reviews = [
        {"verdict": "ADMIT", "category": "a"},
        {"verdict": "REJECT", "category": "b"},
    ]
    result = summarize_reviews(reviews)
    assert result["status"] == "HOLD"
    assert result["admit"] == 1
    assert result["reject"] == 1


def test_parse_model_blob_path_from_modelfile() -> None:
    text = "FROM C:\\models\\blobs\\sha256-deadbeef\nPARAMETER temperature 0.1\n"
    assert str(parse_model_blob_path(text)).endswith("sha256-deadbeef")


def test_call_reviewer_retries_empty_response(monkeypatch) -> None:
    valid = json.dumps({
        "verdict": "ADMIT",
        "prompt_well_posed": True,
        "grader_matches_prompt": True,
        "source_evidence_sufficient": True,
        "issue_code": "NONE",
        "reason": "No material defect.",
        "confidence": "high",
    })
    responses = iter([
        {"response": "", "done_reason": "stop"},
        {"response": valid, "done_reason": "stop"},
    ])
    monkeypatch.setattr(
        audit_module,
        "_ollama_request",
        lambda prompt: next(responses),
    )
    review, runtime = audit_module.call_reviewer("same prompt")
    assert review["verdict"] == "ADMIT"
    assert runtime["attempts"] == 2
