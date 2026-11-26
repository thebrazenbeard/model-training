from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from .scenario import LabTranscript, sha256_json


VERDICTS = {"PASS", "WARN", "FAIL"}


@dataclass(frozen=True)
class CalibrationRecord:
    expected: str
    observed: str

    def __post_init__(self) -> None:
        if self.expected not in VERDICTS:
            raise ValueError("invalid expected calibration verdict")
        if self.observed not in VERDICTS:
            raise ValueError("invalid observed calibration verdict")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def judge_is_calibrated(
    records: Sequence[CalibrationRecord], min_agreement: float = 0.90
) -> bool:
    if len(records) < 10:
        return False
    if not 0.0 <= min_agreement <= 1.0:
        raise ValueError("min_agreement must be between 0 and 1")
    matches = sum(record.expected == record.observed for record in records)
    return matches / len(records) >= min_agreement


@dataclass(frozen=True)
class JudgeResult:
    verdict: str
    dimension_scores: dict[str, float]
    evidence_turn_indices: tuple[int, ...]
    critical_failure_ids: tuple[str, ...]
    judge_id: str
    judge_version: str
    rubric_digest: str
    calibration_digest: str
    calibrated: bool

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError("invalid judge verdict")
        if not self.judge_id or not self.judge_version:
            raise ValueError("judge id/version must be nonempty")
        if not self.rubric_digest or not self.calibration_digest:
            raise ValueError("rubric/calibration digests must be nonempty")
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in self.evidence_turn_indices):
            raise ValueError("evidence turn indices must be nonnegative integers")
        if any(not isinstance(item, str) or not item for item in self.critical_failure_ids):
            raise ValueError("critical failure ids must be nonempty strings")
        for name, score in self.dimension_scores.items():
            if not isinstance(name, str) or not name:
                raise ValueError("dimension names must be nonempty strings")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0.0 <= float(score) <= 1.0:
                raise ValueError("dimension scores must be numeric in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LabJudge(Protocol):
    judge_id: str
    judge_version: str
    calibration_records: Sequence[CalibrationRecord]

    def evaluate(self, transcript: LabTranscript, rubric: dict[str, Any]) -> dict[str, Any]:
        ...


def _calibration_digest(records: Sequence[CalibrationRecord]) -> str:
    return sha256_json([record.to_dict() for record in records])


def score_with_judge(
    transcript: LabTranscript, rubric: dict[str, Any], judge: LabJudge
) -> JudgeResult:
    if not isinstance(rubric, dict):
        raise ValueError("rubric must be an object")
    judge_id = getattr(judge, "judge_id", "")
    judge_version = getattr(judge, "judge_version", "")
    records = tuple(getattr(judge, "calibration_records", ()))
    if not isinstance(judge_id, str) or not judge_id or not isinstance(judge_version, str) or not judge_version:
        raise ValueError("judge must expose nonempty judge_id and judge_version")
    if not all(isinstance(record, CalibrationRecord) for record in records):
        raise ValueError("calibration_records must contain CalibrationRecord instances")

    raw = judge.evaluate(transcript, rubric)
    if not isinstance(raw, dict):
        raise ValueError("judge evaluation must be an object")
    verdict = raw.get("verdict")
    dimensions = raw.get("dimension_scores")
    evidence = raw.get("evidence_turn_indices")
    failures = raw.get("critical_failure_ids")
    if verdict not in VERDICTS:
        raise ValueError("invalid judge verdict")
    if not isinstance(dimensions, dict):
        raise ValueError("dimension_scores must be an object")
    if not isinstance(evidence, list) or not isinstance(failures, list):
        raise ValueError("judge evidence/failures must be lists")
    if any(index >= len(transcript.turns) for index in evidence if isinstance(index, int) and not isinstance(index, bool)):
        raise ValueError("evidence turn index outside transcript")

    return JudgeResult(
        verdict=verdict,
        dimension_scores={key: float(value) for key, value in dimensions.items()},
        evidence_turn_indices=tuple(evidence),
        critical_failure_ids=tuple(failures),
        judge_id=judge_id,
        judge_version=judge_version,
        rubric_digest=sha256_json(rubric),
        calibration_digest=_calibration_digest(records),
        calibrated=judge_is_calibrated(records),
    )
