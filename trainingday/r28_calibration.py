"""Bounded R28 public-synthetic uncertainty/calibration gate for the R27 pilot.

All case labels, confidences, arm assignments and run/split identities are
*caller-declared*. None are authenticated model inference, independent truth
or evidence for a real Qwen3.5-4B weight improvement. Never open an OOD bank.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from statistics import NormalDist

from trainingday.r27_pilot_preregistration import (
    PilotError, PilotProtocol, PreregReceipt,
    preregister_pilot, score_synthetic_pilot,
)


_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{3,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ROLES = frozenset(("FROZEN", "ZERO_UPDATE", "CORRECT_LABEL", "CORRUPTED_LABEL"))


class CalibrationError(ValueError):
    """Bad public declaration or mismatch with a previously frozen plan."""


@dataclass(frozen=True)
class CalibrationPlan:
    run_id: str
    split_tag: str
    pilot_manifest_sha256: str
    bin_count: int
    interval_alpha: float
    max_calibration_regression: float


@dataclass(frozen=True)
class CalibrationReceipt:
    manifest_sha256: str
    canonical_json: str
    oracle_authenticated: bool = False
    blind_allocation_authenticated: bool = False
    live_inference_authenticated: bool = False
    release_authorized: bool = False


@dataclass(frozen=True)
class CalibrationSample:
    case_id: str
    prompt_sha256: str
    confidence: float
    correct: bool
    format_ok: bool
    refused: bool


@dataclass(frozen=True)
class CalibrationBin:
    index: int
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class ArmCalibration:
    role: str
    case_count: int
    accuracy: float
    accuracy_interval: tuple[float, float]
    ece: float
    brier_score: float
    overconfident_error_rate: float
    refusal_rate: float
    format_failure_rate: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class SyntheticCalibrationReceipt:
    status: str
    plan_sha256: str
    pilot_manifest_sha256: str
    frozen: ArmCalibration
    zero_update: ArmCalibration
    correct_label: ArmCalibration
    corrupted_label: ArmCalibration
    correct_minus_frozen_gain: float
    correct_minus_frozen_interval: tuple[float, float]
    synthetic_control_gates_passed: bool
    calibration_nonregression_passed: bool
    zero_update_parity_passed: bool
    interval_assumptions_authenticated: bool = False
    oracle_authenticated: bool = False
    blinded_roles_authenticated: bool = False
    physical_model_outputs_verified: bool = False
    neural_improvement_claim_authorized: bool = False
    release_authorized: bool = False


def _identifier(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def preregister_calibration(
    pilot: PilotProtocol, pilot_receipt: PreregReceipt, plan: CalibrationPlan,
) -> CalibrationReceipt:
    """Freeze bins and policy *before* receiving any synthetic outcomes."""
    if type(plan) is not CalibrationPlan:
        raise CalibrationError("MALFORMED_CALIBRATION_PLAN")
    try:
        expected_pilot = preregister_pilot(pilot)
    except PilotError as exc:
        raise CalibrationError("INVALID_PARENT_PILOT") from exc
    if (type(pilot_receipt) is not PreregReceipt or
            not _hash(pilot_receipt.manifest_sha256) or
            type(pilot_receipt.canonical_json) is not str or
            type(pilot_receipt.case_count) is not int or
            not all(type(v) is bool for v in (
                pilot_receipt.declared_training_overlap_absent,
                pilot_receipt.training_inventory_complete_attested,
                pilot_receipt.rights_authenticated,
                pilot_receipt.blind_allocation_authenticated,
                pilot_receipt.protected_eval_access_authorized,
                pilot_receipt.release_authorized,
            )) or pilot_receipt != expected_pilot):
        raise CalibrationError("PILOT_COMMITMENT_MISMATCH")
    if (not _identifier(plan.run_id) or not _identifier(plan.split_tag) or
            not _hash(plan.pilot_manifest_sha256) or
            plan.pilot_manifest_sha256 != pilot_receipt.manifest_sha256):
        raise CalibrationError("RUN_SPLIT_OR_PILOT_UNBOUND")
    if type(plan.bin_count) is not int or not 2 <= plan.bin_count <= 20:
        raise CalibrationError("INVALID_FIXED_BIN_COUNT")
    if (type(plan.interval_alpha) is not float or
            not math.isfinite(plan.interval_alpha) or
            not 0.001 <= plan.interval_alpha <= 0.2):
        raise CalibrationError("INVALID_INTERVAL_ALPHA")
    if (type(plan.max_calibration_regression) is not float or
            not math.isfinite(plan.max_calibration_regression) or
            not 0.0 <= plan.max_calibration_regression <= 0.05):
        raise CalibrationError("INVALID_NONREGRESSION_TOLERANCE")
    canonical = json.dumps({
        "schema": "trainingday-r28-public-calibration-v1",
        "plan": asdict(plan),
        "fixed_roles": sorted(_ROLES),
        "case_manifest": pilot_receipt.manifest_sha256,
        "bin_rule": "equal-width-left-closed-right-open-last-inclusive",
        "overconfidence_threshold_inclusive": 0.9,
        "overconfident_error_refusal_format_regression_max": 0.0,
        "zero_update_parity": "exact_per_case_correctness_confidence_format_refusal",
        "interval_method": "Wilson accuracy; Hoeffding paired gain; IID not attested",
        "no_neural_claim": True,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return CalibrationReceipt(hashlib.sha256(canonical.encode("ascii")).hexdigest(), canonical)


def _wilson(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Wilson interval, conditional on independent Bernoulli cases."""
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    middle = (p + z * z / (2.0 * n)) / denom
    radius = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, middle - radius), min(1.0, middle + radius)


def _paired_hoeffding(gain: float, n: int, alpha: float) -> tuple[float, float]:
    """Conservative range-two paired mean bound; IID remains unverified."""
    radius = math.sqrt(2.0 * math.log(2.0 / alpha) / n)
    return max(-1.0, gain - radius), min(1.0, gain + radius)


def _summarize(role: str, samples: dict[str, CalibrationSample],
               bins_count: int, alpha: float) -> ArmCalibration:
    n = len(samples)
    occupied = [0] * bins_count
    prob_sum = [0.0] * bins_count
    correct_sum = [0] * bins_count
    brier = 0.0
    overconfident_errors = 0
    refusals = 0
    format_failures = 0
    for sample in samples.values():
        idx = min(bins_count - 1, int(sample.confidence * bins_count))
        occupied[idx] += 1
        prob_sum[idx] += sample.confidence
        correct_sum[idx] += int(sample.correct)
        brier += (sample.confidence - float(sample.correct)) ** 2
        overconfident_errors += int(sample.confidence >= 0.9 and not sample.correct)
        refusals += int(sample.refused)
        format_failures += int(not sample.format_ok)
    bins = tuple(CalibrationBin(i, occupied[i],
        prob_sum[i] / occupied[i] if occupied[i] else 0.0,
        correct_sum[i] / occupied[i] if occupied[i] else 0.0,
    ) for i in range(bins_count))
    ece = sum(b.count / n * abs(b.accuracy - b.mean_confidence) for b in bins)
    total_correct = sum(correct_sum)
    return ArmCalibration(role, n, total_correct / n, _wilson(total_correct, n, alpha),
        ece, brier / n, overconfident_errors / n, refusals / n,
        format_failures / n, bins)


def score_synthetic_calibration(
    pilot: PilotProtocol,
    pilot_receipt: PreregReceipt,
    plan: CalibrationPlan,
    receipt: CalibrationReceipt,
    role_by_arm: dict[str, str],
    public_samples_by_arm: dict[str, dict[str, CalibrationSample]],
) -> SyntheticCalibrationReceipt:
    """Gate caller-declared PUBLIC toy scores; all real-world claims remain false."""
    expected = preregister_calibration(pilot, pilot_receipt, plan)
    if (type(receipt) is not CalibrationReceipt or
            not _hash(receipt.manifest_sha256) or
            type(receipt.canonical_json) is not str or
            not all(type(v) is bool for v in (
                receipt.oracle_authenticated,
                receipt.blind_allocation_authenticated,
                receipt.live_inference_authenticated,
                receipt.release_authorized,
            )) or receipt != expected):
        raise CalibrationError("CALIBRATION_COMMITMENT_MISMATCH")
    arms = pilot.opaque_arm_ids
    if (type(role_by_arm) is not dict or len(role_by_arm) != 4 or
            not all(type(k) is str and type(v) is str for k, v in role_by_arm.items()) or
            set(role_by_arm) != set(arms) or
            set(role_by_arm.values()) != _ROLES):
        raise CalibrationError("ROLE_ALLOCATION_MALFORMED")
    if (type(public_samples_by_arm) is not dict or
            len(public_samples_by_arm) != 4 or
            not all(type(k) is str for k in public_samples_by_arm) or
            set(public_samples_by_arm) != set(arms)):
        raise CalibrationError("MISSING_OR_EXTRA_ARMS")
    cases = {c.case_id: c for c in pilot.cases}
    for arm in arms:
        samples = public_samples_by_arm[arm]
        if (type(samples) is not dict or len(samples) != len(cases) or
                not all(type(k) is str for k in samples)):
            raise CalibrationError("SAMPLE_MAP_SHAPE_INVALID")
        if set(samples) != set(cases):
            raise CalibrationError("PUBLIC_SPLIT_CASE_MISMATCH")
        for case_id, sample in samples.items():
            if (type(case_id) is not str or type(sample) is not CalibrationSample or
                    type(sample.case_id) is not str or
                    type(sample.prompt_sha256) is not str or
                    sample.case_id != case_id or
                    sample.prompt_sha256 != cases[case_id].prompt_sha256 or
                    type(sample.confidence) is not float or
                    not math.isfinite(sample.confidence) or
                    not 0.0 <= sample.confidence <= 1.0 or
                    type(sample.correct) is not bool or
                    type(sample.format_ok) is not bool or
                    type(sample.refused) is not bool or
                    (sample.correct and (sample.refused or not sample.format_ok))):
                raise CalibrationError("SAMPLE_PROMPT_OR_OUTCOME_INVALID")
    by_role = {role: public_samples_by_arm[arm] for arm, role in role_by_arm.items()}
    # A zero-update arm has no legitimate synthetic freedom to change its
    # per-case outputs relative to frozen. Accuracy-only comparison permits
    # arbitrarily divergent confidence/format/refusal behavior to slip by.
    # Strict equality is a preregistered public-toy condition; physical
    # backend floating tolerance, if justified, needs a different protocol.
    zero_update_parity = all(
        by_role["ZERO_UPDATE"][k] == by_role["FROZEN"][k] for k in cases
    )
    summaries = {role: _summarize(role, by_role[role], plan.bin_count, plan.interval_alpha)
                 for role in _ROLES}
    declared_bool = {arm: {k: row.correct for k, row in scores.items()}
                     for arm, scores in public_samples_by_arm.items()}
    try:
        pilot_score = score_synthetic_pilot(pilot, pilot_receipt, role_by_arm, declared_bool)
    except PilotError as exc:
        raise CalibrationError("PARENT_SYNTHETIC_CONTROLS_INVALID") from exc
    base = summaries["FROZEN"]
    correct = summaries["CORRECT_LABEL"]
    tol = plan.max_calibration_regression
    # Scores are source-declared; no independent oracle or run custody exists.
    nonregression = (
        correct.ece <= base.ece + tol + 1e-12 and
        correct.brier_score <= base.brier_score + tol + 1e-12 and
        correct.overconfident_error_rate <= base.overconfident_error_rate + 1e-12 and
        correct.refusal_rate <= base.refusal_rate + 1e-12 and
        correct.format_failure_rate <= base.format_failure_rate + 1e-12
    )
    gain = correct.accuracy - base.accuracy
    status = (
        "NO_CLAIM_ZERO_UPDATE_DIVERGENCE" if not zero_update_parity else
        "NO_CLAIM_CONTROL_FAILURE" if not pilot_score.control_checks_passed else
        "NO_CLAIM_CALIBRATION_REGRESSION" if not nonregression else
        "SYNTHETIC_CALIBRATION_THRESHOLD_MET_NO_CLAIM"
    )
    return SyntheticCalibrationReceipt(status, expected.manifest_sha256,
        pilot_receipt.manifest_sha256, summaries["FROZEN"],
        summaries["ZERO_UPDATE"], summaries["CORRECT_LABEL"],
        summaries["CORRUPTED_LABEL"], gain,
        _paired_hoeffding(gain, len(cases), plan.interval_alpha),
        pilot_score.control_checks_passed and zero_update_parity,
        nonregression, zero_update_parity)
