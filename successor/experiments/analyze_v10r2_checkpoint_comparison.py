from __future__ import annotations

import math
import statistics


class AnalysisHold(RuntimeError):
    """Fail-closed refusal for malformed checkpoint comparison results."""


def classify_record_family(record_id: str) -> str:
    if not isinstance(record_id, str) or not record_id:
        raise AnalysisHold("record id missing")
    if record_id.startswith("v4.1-"):
        body = record_id[len("v4.1-") :]
        family = body.rsplit("-", 1)[0]
        if not family:
            raise AnalysisHold("v4.1 record family missing")
        return f"v4.1:{family}"
    if record_id.startswith("hf-smoltalk2-"):
        body = record_id[len("hf-smoltalk2-") :]
        family = body.rsplit("-", 1)[0]
        if not family:
            raise AnalysisHold("hf record family missing")
        return f"hf:{family}"
    family = record_id.rsplit("-", 1)[0]
    return f"other:{family}"


def _candidate(result: dict, name: str) -> dict:
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise AnalysisHold("candidates missing")
    matched = [item for item in candidates if item.get("name") == name]
    if len(matched) != 1:
        raise AnalysisHold(f"candidate not uniquely found:{name}")
    return matched[0]


def _case_map(candidate: dict) -> dict[str, dict]:
    cases = candidate.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AnalysisHold("candidate cases missing")
    mapped: dict[str, dict] = {}
    for row in cases:
        if not isinstance(row, dict):
            raise AnalysisHold("candidate case invalid")
        record_id = row.get("record_id")
        loss = row.get("loss")
        if not isinstance(record_id, str) or not record_id:
            raise AnalysisHold("candidate record id missing")
        if record_id in mapped:
            raise AnalysisHold(f"duplicate candidate record id:{record_id}")
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise AnalysisHold(f"candidate loss invalid:{record_id}")
        mapped[record_id] = row
    return mapped


def _summary(deltas: list[float]) -> dict:
    return {
        "count": len(deltas),
        "right_minus_left_mean_case_loss": statistics.fmean(deltas),
        "right_minus_left_median_case_loss": statistics.median(deltas),
        "right_case_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "left_case_wins": sum(delta > 0 for delta in deltas),
    }


def summarize_checkpoint_comparison(
    result: dict,
    left_name: str,
    right_name: str,
) -> dict:
    left = _candidate(result, left_name)
    right = _candidate(result, right_name)
    left_map = _case_map(left)
    right_map = _case_map(right)
    if set(left_map) != set(right_map):
        raise AnalysisHold("candidate case sets differ")

    family_deltas: dict[str, list[float]] = {}
    overall_deltas: list[float] = []
    for record_id in sorted(left_map):
        delta = float(right_map[record_id]["loss"]) - float(
            left_map[record_id]["loss"]
        )
        overall_deltas.append(delta)
        family = classify_record_family(record_id)
        family_deltas.setdefault(family, []).append(delta)

    left_nll = left.get("token_weighted_completion_nll")
    right_nll = right.get("token_weighted_completion_nll")
    if not isinstance(left_nll, (int, float)) or not math.isfinite(
        float(left_nll)
    ):
        raise AnalysisHold("left candidate NLL invalid")
    if not isinstance(right_nll, (int, float)) or not math.isfinite(
        float(right_nll)
    ):
        raise AnalysisHold("right candidate NLL invalid")

    overall = _summary(overall_deltas)
    overall.update(
        {
            "left": left_name,
            "right": right_name,
            "left_token_weighted_nll": float(left_nll),
            "right_token_weighted_nll": float(right_nll),
            "right_minus_left_nll": float(right_nll) - float(left_nll),
        }
    )

    return {
        "schema": "V10R2_CHECKPOINT_FAMILY_COMPARISON_V1",
        "status": "COMPLETE",
        "overall": overall,
        "families": {
            family: _summary(deltas)
            for family, deltas in sorted(family_deltas.items())
        },
        "claim_ceiling": (
            "DEVELOPMENT_DIAGNOSTIC_BREAKDOWN_ONLY_NOT_FINAL_BANK_"
            "NOT_EXTERNAL_QUALIFICATION"
        ),
    }
