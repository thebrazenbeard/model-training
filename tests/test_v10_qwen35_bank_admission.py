from __future__ import annotations

import hashlib


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_case(case_id, lane, prompt, family_id, **extra):
    row = {
        "case_id": case_id,
        "lane": lane,
        "family_id": family_id,
        "prompt": prompt,
        "source_id": "fixture-source",
        "source_revision": "fixture-v1",
        "source_terms": "fixture-only",
        "source_hash": digest(case_id),
        "generation_method": "test_fixture",
        "generation_actor_id": "fixture-generator",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "fixture-grader",
            "grader_version": "1",
            "answer_key_digest": digest("answer:" + case_id),
        },
        "review_receipt": {
            "verdict": "ADMIT",
            "reviewer_id": "fixture-independent-reviewer",
            "reviewer_class": "EXTERNAL_MODEL_REVIEW",
            "provider": "fixture-provider",
            "runtime": "fixture-runtime",
            "independent_from_generation": True,
            "artifact_digest": digest("review:" + case_id),
            "subject_digest": digest("subject:" + case_id),
        },
    }
    row.update(extra)
    from successor.experiments.v10_qwen35_bank import review_subject_digest
    row["review_receipt"]["subject_digest"] = review_subject_digest(row)
    return row


def full_fixture_bank():
    rows = []
    for dim in range(1, 21):
        d = f"H{dim:02d}"
        for i in range(500):
            rows.append(make_case(
                f"b-{d}-{i}", "behavioral", f"behavioral {d} prompt {i}",
                f"{d}-family-{i % 50}", dimension=d,
            ))
        for i in range(100):
            rows.append(make_case(
                f"a-{d}-{i}", "adversarial", f"adversarial {d} prompt {i}",
                f"{d}-adv-family-{i}", dimension=d,
            ))
    allocation = {
        "knowledge_factuality": 300,
        "reasoning_math": 300,
        "coding": 250,
        "instruction_following": 250,
        "extraction_structured": 200,
        "truthfulness_factual_calibration": 200,
    }
    for category, count in allocation.items():
        for i in range(count):
            rows.append(make_case(
                f"r-{category}-{i}", "retention", f"retention {category} prompt {i}",
                f"{category}-family-{i}", category=category,
            ))
    return rows


def test_complete_bank_is_ready_to_freeze():
    from successor.experiments.v10_qwen35_bank import preflight_bank
    result = preflight_bank(full_fixture_bank(), exclusion_hashes=set())
    assert result["status"] == "READY_TO_FREEZE"
    assert result["case_count"] == 13_500
    assert result["reasons"] == []


def test_excluded_normalized_prompt_is_rejected():
    from successor.experiments.v10_qwen35_bank import preflight_bank, prompt_fingerprint
    rows = full_fixture_bank()
    excluded = {prompt_fingerprint("  BEHAVIORAL   H01 prompt 0  ")}
    result = preflight_bank(rows, exclusion_hashes=excluded)
    assert result["status"] == "HOLD"
    assert any(reason.startswith("excluded_prompt:") for reason in result["reasons"])


def test_nonindependent_review_is_rejected():
    from successor.experiments.v10_qwen35_bank import preflight_bank
    rows = full_fixture_bank()
    rows[0]["review_receipt"]["reviewer_id"] = rows[0]["generation_actor_id"]
    result = preflight_bank(rows, exclusion_hashes=set())
    assert result["status"] == "HOLD"
    assert any(reason.startswith("invalid_review_receipt:") for reason in result["reasons"])


def test_review_requires_provider_runtime_and_subject_digest():
    from successor.experiments.v10_qwen35_bank import preflight_bank
    rows = full_fixture_bank()
    del rows[0]["review_receipt"]["runtime"]
    result = preflight_bank(rows, exclusion_hashes=set())
    assert result["status"] == "HOLD"
    assert any(reason.startswith("invalid_review_receipt:") for reason in result["reasons"])


def test_semantic_case_rejects_model_only_review() -> None:
    from successor.experiments.v10_qwen35_bank import preflight_bank
    rows = full_fixture_bank()
    rows[0]["grader_contract"] = {
        "kind": "semantic_review",
        "grader_id": "semantic-rubric",
        "grader_version": "1",
        "rubric_digest": digest("semantic-rubric"),
    }
    rows[0]["review_receipt"]["reviewer_class"] = "EXTERNAL_MODEL_REVIEW"
    from successor.experiments.v10_qwen35_bank import review_subject_digest
    rows[0]["review_receipt"]["subject_digest"] = review_subject_digest(rows[0])
    result = preflight_bank(rows, exclusion_hashes=set())
    assert result["status"] == "HOLD"
    assert any(reason.startswith("invalid_review_receipt:") for reason in result["reasons"])


def test_semantic_case_accepts_independent_human_review() -> None:
    from successor.experiments.v10_qwen35_bank import preflight_bank
    rows = full_fixture_bank()
    rows[0]["grader_contract"] = {
        "kind": "semantic_review",
        "grader_id": "semantic-rubric",
        "grader_version": "1",
        "rubric_digest": digest("semantic-rubric"),
    }
    rows[0]["review_receipt"]["reviewer_class"] = "HUMAN_REVIEW"
    from successor.experiments.v10_qwen35_bank import review_subject_digest
    rows[0]["review_receipt"]["subject_digest"] = review_subject_digest(rows[0])
    result = preflight_bank(rows, exclusion_hashes=set())
    assert result["status"] == "READY_TO_FREEZE"
