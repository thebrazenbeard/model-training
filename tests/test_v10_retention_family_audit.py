from __future__ import annotations

import json

import pytest

import successor.experiments.run_v10_retention_family_audit as family_audit


def _case(case_id: str, family_id: str = "fam") -> dict:
    return {
        "case_id": case_id,
        "category": "coding",
        "family_id": family_id,
        "prompt": f"Prompt {case_id}",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "g",
            "grader_version": "1",
            "answer_key_digest": "a" * 64,
        },
        "source_evidence": {"family": family_id, "variant": 1},
    }


def _review(case_id: str, verdict: str = "ADMIT") -> dict:
    ok = verdict == "ADMIT"
    return {
        "case_id": case_id,
        "verdict": verdict,
        "prompt_well_posed": ok,
        "grader_matches_prompt": ok,
        "source_evidence_sufficient": ok,
        "issue_code": "NONE" if ok else "DEFECT",
        "reason": "No material defect." if ok else "Material defect.",
        "confidence": "high",
    }


def test_family_prompt_contains_all_case_ids_without_prior_verdicts() -> None:
    cases = [_case(f"c{i}") for i in range(5)]
    prompt = family_audit.family_review_prompt(cases)
    for case in cases:
        assert case["case_id"] in prompt
    assert "PENDING_INDEPENDENT_REVIEW" not in prompt


def test_parse_family_reviews_requires_exact_case_set_and_closed_schema() -> None:
    ids = [f"c{i}" for i in range(5)]
    text = json.dumps({"reviews": [_review(case_id) for case_id in ids]})
    parsed = family_audit.parse_family_reviews(text, expected_case_ids=ids)
    assert [row["case_id"] for row in parsed] == ids

    missing = json.dumps({"reviews": [_review(case_id) for case_id in ids[:-1]]})
    with pytest.raises(ValueError, match="case set"):
        family_audit.parse_family_reviews(missing, expected_case_ids=ids)

    extra_field = _review(ids[0])
    extra_field["unexpected"] = True
    bad = json.dumps({
        "reviews": [extra_field] + [_review(case_id) for case_id in ids[1:]]
    })
    with pytest.raises(ValueError, match="closed schema"):
        family_audit.parse_family_reviews(bad, expected_case_ids=ids)


def test_call_family_reviewer_retries_invalid_transport(monkeypatch) -> None:
    cases = [_case(f"c{i}") for i in range(5)]
    valid = json.dumps({
        "reviews": [_review(case["case_id"]) for case in cases]
    })
    responses = iter([
        {"response": "", "done_reason": "stop"},
        {"response": valid, "done_reason": "stop"},
    ])
    monkeypatch.setattr(
        family_audit,
        "_ollama_request",
        lambda prompt: next(responses),
    )
    reviews, runtime = family_audit.call_family_reviewer(cases)
    assert len(reviews) == 5
    assert runtime["attempts"] == 2


def test_group_packet_requires_exact_family_size() -> None:
    rows = [_case(f"a{i}", family_id="a") for i in range(5)]
    rows += [_case(f"b{i}", family_id="b") for i in range(4)]
    with pytest.raises(ValueError, match="family b"):
        family_audit.group_packet_by_family(rows, expected_per_family=5)


def test_summarize_family_audit_holds_on_any_reject() -> None:
    rows = []
    for family in ("a", "b"):
        for i in range(5):
            item = {
                "case_id": f"{family}{i}",
                "family_id": family,
                "category": "coding",
                "verdict": "ADMIT",
                "review": _review(f"{family}{i}"),
            }
            rows.append(item)
    rows[-1]["verdict"] = "REJECT"
    rows[-1]["review"] = _review(rows[-1]["case_id"], "REJECT")
    result = family_audit.summarize_family_audit(rows, expected_per_family=5)
    assert result["status"] == "HOLD"
    assert result["families_reviewed"] == 2
    assert result["reject"] == 1
