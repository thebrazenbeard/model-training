"""Source-only preregistration math for a future R18 MeZO/QLoRA rerun.

Forecasts are declared, untrusted measurements. They cannot authorize a GPU
lease, demonstrate optimizer updates, or rehabilitate the invalid R18 run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.r18_benchmark import canonical_sha256, validate_plan


ARMS = ("mezo", "qlora")
CONTROLS = ("frozen", "zero", "correct", "corrupt")
COUNTERS = (
    "optimizer_steps", "example_presentations", "input_tokens_presented",
    "supervised_target_tokens", "forward_sample_tokens", "backward_sample_tokens",
    "padded_forward_tokens", "padded_backward_tokens",
)
_PROFILE_FIELDS = {
    "arm", "plan_sha256", "selected_id_sha256", "train_content_sha256",
    "seed", "source", "data_partition", "eval_access",
    "elapsed_seconds", *COUNTERS,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def target_token_mean(microbatch_losses: Sequence[float],
                      supervised_target_counts: Sequence[int]) -> float:
    """The common R19 objective, not an unweighted mean of microbatch means."""
    _require(len(microbatch_losses) == len(supervised_target_counts)
             and bool(microbatch_losses), "loss/count shape mismatch")
    _require(all(type(n) is int and 0 < n <= 2**53 for n in supervised_target_counts),
             "invalid supervised target count")
    _require(all(type(v) in (int, float) and abs(v) <= 1e9
                 and math.isfinite(v)
                 for v in microbatch_losses), "invalid microbatch loss")
    total = sum(supervised_target_counts)
    return sum(loss * count for loss, count in zip(
        microbatch_losses, supervised_target_counts)) / total


def preregister_work_forecast(plan: Mapping, profiles: Sequence[Mapping]) -> dict:
    """Bind two untrusted train-only throughput profiles to the same R18 plan.

    Calibrations are descriptive. In particular, a matched *wall time* budget
    is NOT equivalent to equal examples or equal forward/backward token work.
    """
    plan_sha = validate_plan(plan)
    _require(isinstance(profiles, (list, tuple)) and len(profiles) == 2,
             "both method profiles required")
    keyed = {}
    for profile in profiles:
        _require(isinstance(profile, Mapping) and set(profile) == _PROFILE_FIELDS,
                 "unrecognized/missing profile fields")
        arm = profile["arm"]
        _require(arm in ARMS and arm not in keyed, "duplicate/unknown method")
        _require(profile["plan_sha256"] == plan_sha
                 and profile["selected_id_sha256"] == plan["selected_id_sha256"]
                 and profile["train_content_sha256"] == plan["train_content_sha256"]
                 and profile["seed"] == plan["seed"], "calibration is not plan-bound")
        _require(profile["source"] == "DECLARED_UNTRUSTED_CALIBRATION"
                 and profile["data_partition"] == "TRAIN_ONLY"
                 and profile["eval_access"] is False,
                 "private evaluation or unrecognized calibration")
        elapsed = profile["elapsed_seconds"]
        _require(type(elapsed) in (int, float)
                 and 0 < elapsed <= plan["probe_budget_seconds"]
                 and math.isfinite(elapsed),
                 "invalid/overrun calibration window")
        _require(all(type(profile[c]) is int and 0 <= profile[c] <= 2**53
                     for c in COUNTERS), "invalid declared work counters")
        _require(profile["optimizer_steps"] > 0
                 and profile["example_presentations"] > 0
                 and 0 < profile["supervised_target_tokens"] <= profile["input_tokens_presented"]
                 and profile["forward_sample_tokens"] >= profile["input_tokens_presented"]
                 and profile["padded_forward_tokens"] >= profile["forward_sample_tokens"]
                 and profile["padded_backward_tokens"] >= profile["backward_sample_tokens"],
                 "impossible declared useful/padded token work")
        _require((arm == "mezo" and profile["backward_sample_tokens"] == 0
                  and profile["padded_backward_tokens"] == 0
                  and profile["forward_sample_tokens"] >=
                  2 * profile["input_tokens_presented"])
                 or (arm == "qlora" and profile["backward_sample_tokens"] > 0),
                 "algorithm accounting is not MeZO two-sided or QLoRA backward")
        keyed[arm] = profile

    budget = plan["training_budget_seconds"]
    forecast = {}
    for arm in ARMS:
        profile = keyed[arm]
        scale = budget / profile["elapsed_seconds"]
        _require(math.isfinite(scale) and scale > 0,
                 "nonfinite projected work scale")
        projected_work = {}
        for name in COUNTERS:
            estimate = profile[name] * scale
            _require(math.isfinite(estimate) and estimate >= 0,
                     "nonfinite projected work counter")
            projected_work[name] = round(estimate, 3)
        forecast[arm] = {
            "calibration_elapsed_seconds": profile["elapsed_seconds"],
            "estimated_at_equal_optimizer_wall_time": projected_work,
        }
    # SHA binds the *declarations*, not physical clocks, model identity or data.
    return {
        "schema": "VERA_TRAININGDAY_R22_EQUAL_COST_FORECAST_V1",
        "status": "UNTRUSTED_FORECAST_NOT_AN_EXPERIMENT",
        "plan_sha256": plan_sha,
        "declared_profile_sha256": canonical_sha256([keyed[a] for a in ARMS]),
        "base_revision": plan["base_revision"],
        "train_content_sha256": plan["train_content_sha256"],
        "selected_id_sha256": plan["selected_id_sha256"],
        "seed": plan["seed"],
        "equal_optimizer_wall_budget_seconds": {arm: budget for arm in ARMS},
        "controls": list(CONTROLS),
        "objective": "SUPERVISED_TARGET_TOKEN_WEIGHTED_MEAN_BOTH_ARMS",
        "forecast": forecast,
        "equal_sample_token_exposure_asserted": False,
        "same_admitted_train_selection_declared": True,
        "physical_host_lease_authenticated": False,
        "authorizes_gpu_start": False,
        "model_improvement_proven": False,
        "promotion_allowed": False,
        "historical_r18_overrun_comparison_valid": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Untrusted R22 work forecast; never launches GPU")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = preregister_work_forecast(
            json.loads(args.plan.read_text(encoding="utf-8")),
            json.loads(args.profiles.read_text(encoding="utf-8")),
        )
    except (AdmissionError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "HOLD_INVALID_DECLARED_FORECAST",
                          "reason": str(exc), "authorizes_gpu_start": False},
                         sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    # A complete forecast is NOT an optimizer-launch authorization.
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
