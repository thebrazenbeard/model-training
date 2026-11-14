from successor.vera_lab.scenario import LabTranscript, LabTurn, sha256_json
from successor.vera_lab.scoring import (
    CalibrationRecord,
    JudgeResult,
    judge_is_calibrated,
    score_with_judge,
)


def transcript():
    return LabTranscript(
        scenario_digest="scenario-sha",
        candidate_digest="candidate-sha",
        seed=7,
        turns=(LabTurn(0, "assistant", "answer", "state", (), {}),),
        completion_status="COMPLETE",
    )


def test_uncalibrated_judge_cannot_gate():
    records = [CalibrationRecord(expected="PASS", observed="FAIL")]
    assert judge_is_calibrated(records, min_agreement=0.90) is False


def test_too_few_calibration_examples_never_qualify():
    records = [CalibrationRecord(expected="PASS", observed="PASS") for _ in range(9)]
    assert judge_is_calibrated(records, min_agreement=0.90) is False


def test_ten_examples_at_threshold_qualify():
    records = [CalibrationRecord(expected="PASS", observed="PASS") for _ in range(9)]
    records.append(CalibrationRecord(expected="FAIL", observed="WARN"))
    assert judge_is_calibrated(records, min_agreement=0.90) is True


class FakeJudge:
    judge_id = "fake-judge"
    judge_version = "1.0"
    calibration_records = tuple(
        CalibrationRecord(expected="PASS", observed="PASS") for _ in range(10)
    )

    def evaluate(self, transcript, rubric):
        return {
            "verdict": "WARN",
            "dimension_scores": {"identity": 0.8, "ordinary_competence": 1.0},
            "evidence_turn_indices": [0],
            "critical_failure_ids": [],
        }


def test_score_records_provenance_and_calibration():
    rubric = {"dimensions": ["identity", "ordinary_competence"]}
    result = score_with_judge(transcript(), rubric, FakeJudge())
    assert isinstance(result, JudgeResult)
    assert result.verdict == "WARN"
    assert result.judge_id == "fake-judge"
    assert result.judge_version == "1.0"
    assert result.rubric_digest == sha256_json(rubric)
    assert result.calibrated is True
    assert len(result.calibration_digest) == 64
    assert result.evidence_turn_indices == (0,)


class BadJudge(FakeJudge):
    def evaluate(self, transcript, rubric):
        return {
            "verdict": "MAYBE",
            "dimension_scores": {},
            "evidence_turn_indices": [],
            "critical_failure_ids": [],
        }


def test_invalid_judge_verdict_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        score_with_judge(transcript(), {}, BadJudge())


def test_calibration_digest_is_deterministic():
    first = score_with_judge(transcript(), {}, FakeJudge())
    second = score_with_judge(transcript(), {}, FakeJudge())
    assert first.calibration_digest == second.calibration_digest
    assert first.to_dict() == second.to_dict()
