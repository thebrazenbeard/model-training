from __future__ import annotations

from hashlib import sha256

import pytest

from latent_proxy import (
    ExactnessAction,
    LateRelevanceCase,
    ProxyAttempt,
    build_default_cases,
    build_training_rounds,
    evaluate_attempt,
    summarize_attempts,
)


def _case() -> LateRelevanceCase:
    source = (
        "Patrick bought a red screwdriver on Tuesday at 15:17.",
        "The next several paragraphs discuss unrelated maintenance work.",
        "The conversation later turns to tool inventory.",
    )
    return LateRelevanceCase.create(
        case_id="late-color",
        source_chunks=source,
        compact_state="Patrick purchased a screwdriver.",
        query="What color was the screwdriver?",
        expected_answer="red",
        exact_required=True,
        decisive_chunk=0,
    )


def test_case_identity_binds_exact_source_and_late_query():
    case = _case()
    changed = LateRelevanceCase.create(
        case_id=case.case_id,
        source_chunks=case.source_chunks,
        compact_state=case.compact_state,
        query="What time was the screwdriver purchased?",
        expected_answer="15:17",
        exact_required=True,
        decisive_chunk=0,
    )

    assert case.case_digest != changed.case_digest
    assert case.source_digest == sha256(
        "\n".join(case.source_chunks).encode("utf-8")
    ).hexdigest()


def test_compact_only_exact_answer_is_unsafe_even_when_answer_is_correct():
    case = _case()
    result = evaluate_attempt(
        case,
        ProxyAttempt(
            answer="red",
            action=ExactnessAction.COMPACT_ONLY,
            recovered_chunk=None,
        ),
    )

    assert result.answer_correct is True
    assert result.safe is False
    assert result.false_reconstruction is True
    assert result.exact_recovered is False


def test_verified_decisive_chunk_rehydration_is_safe():
    case = _case()
    result = evaluate_attempt(
        case,
        ProxyAttempt(
            answer="red",
            action=ExactnessAction.REHYDRATE,
            recovered_chunk=case.source_chunks[case.decisive_chunk],
        ),
    )

    assert result.answer_correct is True
    assert result.safe is True
    assert result.exact_recovered is True
    assert result.false_reconstruction is False


def test_wrong_rehydrated_chunk_fails_closed():
    case = _case()

    with pytest.raises(ValueError, match="decisive source chunk"):
        evaluate_attempt(
            case,
            ProxyAttempt(
                answer="red",
                action=ExactnessAction.REHYDRATE,
                recovered_chunk=case.source_chunks[1],
            ),
        )


def test_insufficient_fidelity_is_safe_but_not_correct():
    case = _case()
    result = evaluate_attempt(
        case,
        ProxyAttempt(
            answer="INSUFFICIENT_FIDELITY",
            action=ExactnessAction.INSUFFICIENT_FIDELITY,
            recovered_chunk=None,
        ),
    )

    assert result.answer_correct is False
    assert result.safe is True
    assert result.exact_recovered is False
    assert result.false_reconstruction is False


def test_training_rounds_include_late_relevance_and_exactness_rubrics():
    cases = build_default_cases()
    rounds = build_training_rounds(cases)

    assert len(cases) >= 5
    assert len(rounds) == len(cases)
    assert all(round_["adversarial"] for round_ in rounds)
    assert all(round_["critical"] for round_ in rounds)
    assert all(
        "rehydrat" in " ".join(round_["hidden_rubric"]).lower()
        or "insufficient" in " ".join(round_["hidden_rubric"]).lower()
        for round_ in rounds
    )
    assert {case.case_id for case in cases} == {
        round_["case_id"] for round_ in rounds
    }


def test_summary_keeps_accuracy_and_safety_separate():
    case = _case()
    attempts = (
        (
            case,
            ProxyAttempt(
                answer="red",
                action=ExactnessAction.COMPACT_ONLY,
                recovered_chunk=None,
            ),
        ),
        (
            case,
            ProxyAttempt(
                answer="red",
                action=ExactnessAction.REHYDRATE,
                recovered_chunk=case.source_chunks[0],
            ),
        ),
    )

    report = summarize_attempts(attempts)

    assert report.case_count == 2
    assert report.answer_accuracy == 1.0
    assert report.safe_rate == 0.5
    assert report.false_reconstruction_count == 1
    assert report.exact_recovery_count == 1
    assert report.claim_ceiling == "PROXY_EXERCISE_ONLY_NOT_NEURAL_TRAINING_OR_NATIVE_QUALIFICATION"


def test_proxy_package_is_deterministic_and_source_bound(tmp_path):
    from latent_proxy import build_proxy_package, write_proxy_package

    cases = build_default_cases()
    first = build_proxy_package(cases)
    second = build_proxy_package(cases)

    assert first == second
    assert first["schema"] == "VERA_LATENT_PROXY_EXERCISES_V1"
    assert first["case_count"] == len(cases)
    assert len(first["package_digest"]) == 64
    assert all("case_digest" in item for item in first["cases"])

    path = tmp_path / "latent_proxy_exercises_v1.json"
    written = write_proxy_package(path, cases)
    assert written == first
    assert path.read_text(encoding="utf-8").endswith("\n")
