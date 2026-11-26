from __future__ import annotations

import pytest

from successor.experiments.analyze_v10r2_checkpoint_comparison import (
    AnalysisHold,
    classify_record_family,
    summarize_checkpoint_comparison,
)


def _result() -> dict:
    return {
        "candidates": [
            {
                "name": "step40",
                "token_weighted_completion_nll": 1.50,
                "cases": [
                    {"record_id": "v4.1-privacy_boundary-0001", "loss": 1.0},
                    {
                        "record_id": (
                            "hf-smoltalk2-OpenHermes_2.5_no_think-abc123"
                        ),
                        "loss": 2.0,
                    },
                    {"record_id": "v4.1-privacy_boundary-0002", "loss": 1.5},
                ],
            },
            {
                "name": "step48",
                "token_weighted_completion_nll": 1.40,
                "cases": [
                    {"record_id": "v4.1-privacy_boundary-0001", "loss": 0.8},
                    {
                        "record_id": (
                            "hf-smoltalk2-OpenHermes_2.5_no_think-abc123"
                        ),
                        "loss": 2.2,
                    },
                    {"record_id": "v4.1-privacy_boundary-0002", "loss": 1.5},
                ],
            },
        ]
    }


def test_classify_record_family_preserves_v4_and_hf_source() -> None:
    assert (
        classify_record_family("v4.1-privacy_boundary-0001")
        == "v4.1:privacy_boundary"
    )
    assert (
        classify_record_family(
            "hf-smoltalk2-OpenHermes_2.5_no_think-abc123"
        )
        == "hf:OpenHermes_2.5_no_think"
    )


def test_summarize_checkpoint_comparison_exposes_family_regressions() -> None:
    got = summarize_checkpoint_comparison(_result(), "step40", "step48")

    assert got["overall"]["right_case_wins"] == 1
    assert got["overall"]["left_case_wins"] == 1
    assert got["overall"]["ties"] == 1
    assert got["overall"]["right_minus_left_nll"] == pytest.approx(-0.10)

    privacy = got["families"]["v4.1:privacy_boundary"]
    assert privacy["count"] == 2
    assert privacy["right_case_wins"] == 1
    assert privacy["left_case_wins"] == 0
    assert privacy["ties"] == 1
    assert privacy["right_minus_left_mean_case_loss"] == pytest.approx(-0.10)

    hermes = got["families"]["hf:OpenHermes_2.5_no_think"]
    assert hermes["count"] == 1
    assert hermes["left_case_wins"] == 1
    assert hermes["right_case_wins"] == 0
    assert hermes["right_minus_left_mean_case_loss"] == pytest.approx(0.20)


def test_summarize_rejects_mismatched_case_sets() -> None:
    value = _result()
    value["candidates"][1]["cases"].pop()

    with pytest.raises(AnalysisHold, match="candidate case sets differ"):
        summarize_checkpoint_comparison(value, "step40", "step48")
