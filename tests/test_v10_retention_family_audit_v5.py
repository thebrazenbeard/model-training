from __future__ import annotations

import json

import pytest

import successor.experiments.run_v10_retention_family_audit_v5 as audit_v5


def _case(case_id: str, family_id: str = "fam", answer: str = "TRUE") -> dict:
    return {
        "case_id": case_id,
        "category": "truthfulness_factual_calibration",
        "family_id": family_id,
        "prompt": f"Prompt {case_id}",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "exact_choice_v1",
            "grader_version": "1",
            "answer_key_digest": "a" * 64,
            "answer_key": answer,
        },
        "source_evidence": {"code": "AA", "name": "Example"},
    }


def _judgment(
    case_id: str,
    *,
    prompt_ok: bool = True,
    grader_ok: bool = True,
    evidence_ok: bool = True,
) -> dict:
    ok = prompt_ok and grader_ok and evidence_ok
    return {
        "case_id": case_id,
        "prompt_well_posed": prompt_ok,
        "grader_matches_prompt": grader_ok,
        "evidence_policy_supports_expected_answer": evidence_ok,
        "issue_code": "NONE" if ok else "DEFECT",
        "reason": "No material defect." if ok else "Material defect.",
        "confidence": "high",
    }


def test_v5_prompt_defines_not_enough_info_semantics() -> None:
    cases = [_case(f"c{i}", answer="NOT_ENOUGH_INFO") for i in range(5)]
    prompt = audit_v5.family_review_prompt_v5(cases)
    assert "Do not require external evidence of the real-world claim" in prompt
    assert "NOT_ENOUGH_INFO" in prompt
    assert "evidence_policy_supports_expected_answer" in prompt
    assert "Do not return a verdict field" in prompt


def test_v5_parse_derives_admit_from_three_true_checks() -> None:
    ids = [f"c{i}" for i in range(5)]
    parsed = audit_v5.parse_family_judgments_v5(
        json.dumps({"reviews": [_judgment(case_id) for case_id in ids]}),
        expected_case_ids=ids,
    )
    assert all(row["verdict"] == "ADMIT" for row in parsed)


def test_v5_parse_derives_reject_from_evidence_policy_false() -> None:
    ids = [f"c{i}" for i in range(5)]
    rows = [_judgment(case_id) for case_id in ids]
    rows[0] = _judgment(ids[0], evidence_ok=False)
    parsed = audit_v5.parse_family_judgments_v5(
        json.dumps({"reviews": rows}),
        expected_case_ids=ids,
    )
    assert parsed[0]["verdict"] == "REJECT"


def test_v5_rejects_model_supplied_verdict_field() -> None:
    ids = [f"c{i}" for i in range(5)]
    rows = [_judgment(case_id) for case_id in ids]
    rows[0]["verdict"] = "ADMIT"
    with pytest.raises(ValueError, match="closed schema"):
        audit_v5.parse_family_judgments_v5(
            json.dumps({"reviews": rows}),
            expected_case_ids=ids,
        )


def test_v5_call_retries_only_invalid_transport(monkeypatch) -> None:
    cases = [_case(f"c{i}") for i in range(5)]
    valid = json.dumps({
        "reviews": [_judgment(case["case_id"]) for case in cases]
    })
    responses = iter([
        {"response": "", "done_reason": "stop"},
        {"response": valid, "done_reason": "stop"},
    ])
    monkeypatch.setattr(
        audit_v5,
        "_ollama_request",
        lambda prompt: next(responses),
    )
    parsed, runtime = audit_v5.call_family_reviewer_v5(cases)
    assert len(parsed) == 5
    assert runtime["attempts"] == 2


def test_v5_valid_negative_does_not_retry(monkeypatch) -> None:
    cases = [_case(f"c{i}") for i in range(5)]
    rows = [_judgment(case["case_id"]) for case in cases]
    rows[2] = _judgment(cases[2]["case_id"], grader_ok=False)
    calls = 0

    def request(prompt: str) -> dict:
        nonlocal calls
        calls += 1
        return {"response": json.dumps({"reviews": rows}), "done_reason": "stop"}

    monkeypatch.setattr(audit_v5, "_ollama_request", request)
    parsed, runtime = audit_v5.call_family_reviewer_v5(cases)
    assert calls == 1
    assert runtime["attempts"] == 1
    assert parsed[2]["verdict"] == "REJECT"
