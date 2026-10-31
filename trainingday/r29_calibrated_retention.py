"""R29 PUBLIC TOY A→B→C calibrated retention: a declaration-only HOLD gate.

No model, evaluation-bank, trainer, API or adapter is accessed. Callers supply
every outcome, hash and control; this module authenticates none of them.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from trainingday.admission import AdmissionError
from trainingday.r25_multigeneration_retention import (
    GenerationEvidence, PublicBankManifest, RetentionPolicy,
    assess_generation_retention, case_order_sha256,
)
from trainingday.r27_pilot_preregistration import PilotProtocol, PreregReceipt
from trainingday.r28_calibration import (
    CalibrationError, CalibrationPlan, CalibrationReceipt, CalibrationSample,
    preregister_calibration, score_synthetic_calibration,
)


_ROLES = frozenset(("FROZEN", "ZERO_UPDATE", "CORRECT_LABEL", "CORRUPTED_LABEL"))
_GENS = ("A", "B", "C")
_SCHEMA = "trainingday-r29-public-toy-calibrated-retention-v1"


class CalibratedRetentionError(ValueError):
    """Public toy commitment or source/role/cohort mismatch."""


@dataclass(frozen=True)
class CalibratedRetentionPlan:
    source_sha256: str
    case_order_sha256: str
    pilot_manifest_sha256: str
    calibration_manifest_sha256: str
    allowed_abstain_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibratedRetentionReceipt:
    manifest_sha256: str
    canonical_json: str
    role_allocation_sha256: str
    independent_oracle_authenticated: bool = False
    physical_model_outputs_verified: bool = False
    release_authorized: bool = False


def _roles(pilot: PilotProtocol, role_by_arm: object) -> dict[str, str]:
    if (type(role_by_arm) is not dict or len(role_by_arm) != 4 or
            not all(type(k) is str and type(v) is str
                    for k, v in role_by_arm.items()) or
            set(role_by_arm) != set(pilot.opaque_arm_ids) or
            set(role_by_arm.values()) != _ROLES):
        raise CalibratedRetentionError("R29_ROLE_SOURCE_MISMATCH")
    return role_by_arm


def _role_sha256(role_by_arm: dict[str, str]) -> str:
    canonical = json.dumps(role_by_arm, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def preregister_calibrated_retention(
    manifest: PublicBankManifest,
    pilot: PilotProtocol,
    pilot_receipt: PreregReceipt,
    calibration_plan: CalibrationPlan,
    calibration_receipt: CalibrationReceipt,
    plan: CalibratedRetentionPlan,
    role_by_arm: dict[str, str],
) -> CalibratedRetentionReceipt:
    """Freeze public cohort, abstention allowlist and opaque control roles."""
    if type(plan) is not CalibratedRetentionPlan:
        raise CalibratedRetentionError("R29_PLAN_TYPE_MISMATCH")
    if type(manifest) is not PublicBankManifest or type(pilot) is not PilotProtocol:
        raise CalibratedRetentionError("R29_PARENT_TYPE_MISMATCH")
    try:
        order = case_order_sha256(manifest.case_ids)
        expected_calibration = preregister_calibration(
            pilot, pilot_receipt, calibration_plan)
    except (AdmissionError, CalibrationError, AttributeError, TypeError) as exc:
        raise CalibratedRetentionError("R29_PARENT_COMMITMENT_INVALID") from exc
    if (type(calibration_receipt) is not CalibrationReceipt or
            calibration_receipt != expected_calibration or
            manifest.base_sha256 != pilot.frozen_base_sha256 or
            manifest.case_ids != tuple(c.case_id for c in pilot.cases) or
            type(plan.source_sha256) is not str or
            plan.source_sha256 != manifest.source_sha256 or
            type(plan.case_order_sha256) is not str or
            plan.case_order_sha256 != order or
            type(plan.pilot_manifest_sha256) is not str or
            plan.pilot_manifest_sha256 != pilot_receipt.manifest_sha256 or
            type(plan.calibration_manifest_sha256) is not str or
            plan.calibration_manifest_sha256 != calibration_receipt.manifest_sha256):
        raise CalibratedRetentionError("R29_PUBLIC_SOURCE_OR_COHORT_MISMATCH")
    if (type(plan.allowed_abstain_case_ids) is not tuple or
            len(plan.allowed_abstain_case_ids) > len(manifest.case_ids) or
            any(type(k) is not str for k in plan.allowed_abstain_case_ids) or
            len(set(plan.allowed_abstain_case_ids)) != len(plan.allowed_abstain_case_ids) or
            not set(plan.allowed_abstain_case_ids).issubset(manifest.case_ids)):
        raise CalibratedRetentionError("R29_ABSTENTION_PREREG_INVALID")
    role_map = _roles(pilot, role_by_arm)
    role_digest = _role_sha256(role_map)
    canonical = json.dumps({
        "schema": _SCHEMA,
        "plan": asdict(plan),
        "role_allocation_sha256": role_digest,
        "generations": _GENS,
        "max_case_regressions": 0,
        "max_new_abstentions": 0,
        "format_regressions": 0,
        "control_drift_allowed": False,
        "calibration_regression_tolerance": calibration_plan.max_calibration_regression,
        "physical_inference_or_reviewer_claim": False,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return CalibratedRetentionReceipt(digest, canonical, role_digest)


def score_synthetic_calibrated_retention(
    manifest: PublicBankManifest,
    generation_records: tuple[GenerationEvidence, ...],
    pilot: PilotProtocol,
    pilot_receipt: PreregReceipt,
    calibration_plan: CalibrationPlan,
    calibration_receipt: CalibrationReceipt,
    plan: CalibratedRetentionPlan,
    plan_receipt: CalibratedRetentionReceipt,
    role_by_arm: dict[str, str],
    samples_by_generation: dict[str, dict[str, dict[str, CalibrationSample]]],
) -> dict:
    """Pair R25 lineage and R28 per-case/arm calibration across A/B/C.

    Checks are deterministic consistency checks only. A clean synthetic case
    still returns HOLD: no authenticated labels, tensors, custody, or reload.
    """
    # Freeze every caller-owned source before invoking a downstream assessor.
    # Both R25 and R28 can be wrapped/monkeypatched by a caller; passing their
    # mutable references through would permit reported statistics to refer to
    # one state while our later comparisons read a different state (TOCTOU).
    if (type(manifest) is not PublicBankManifest or
            type(pilot) is not PilotProtocol or
            type(pilot_receipt) is not PreregReceipt or
            type(calibration_plan) is not CalibrationPlan or
            type(calibration_receipt) is not CalibrationReceipt or
            type(manifest.case_ids) is not tuple or
            not 3 <= len(manifest.case_ids) <= 512 or
            type(pilot.cases) is not tuple or
            not 3 <= len(pilot.cases) <= 512 or
            type(pilot.opaque_arm_ids) is not tuple or
            len(pilot.opaque_arm_ids) != 4 or
            type(generation_records) is not tuple or
            len(generation_records) != 3 or
            any(type(row) is not GenerationEvidence or
                type(row.case_passes) is not tuple or
                len(row.case_passes) > 512 for row in generation_records)):
        raise CalibratedRetentionError("R29_SOURCE_SNAPSHOT_INVALID")
    frozen_manifest = copy.deepcopy(manifest)
    frozen_pilot = copy.deepcopy(pilot)
    frozen_pilot_receipt = copy.deepcopy(pilot_receipt)
    frozen_calibration_plan = copy.deepcopy(calibration_plan)
    frozen_calibration_receipt = copy.deepcopy(calibration_receipt)
    frozen_plan = copy.deepcopy(plan)
    frozen_records = copy.deepcopy(generation_records)
    frozen_roles = dict(_roles(frozen_pilot, role_by_arm))
    if (type(samples_by_generation) is not dict or
            len(samples_by_generation) != 3 or
            set(samples_by_generation) != set(_GENS) or
            any(type(k) is not str for k in samples_by_generation)):
        raise CalibratedRetentionError("R29_GENERATION_SAMPLE_SHAPE_INVALID")
    frozen_samples = {}
    for generation in _GENS:
        arm_maps = samples_by_generation[generation]
        if (type(arm_maps) is not dict or len(arm_maps) != 4 or
                set(arm_maps) != set(frozen_pilot.opaque_arm_ids) or
                any(type(k) is not str for k in arm_maps)):
            raise CalibratedRetentionError("R29_ARM_SNAPSHOT_INVALID")
        frozen_samples[generation] = {}
        for arm in frozen_pilot.opaque_arm_ids:
            rows = arm_maps[arm]
            if (type(rows) is not dict or
                    len(rows) != len(frozen_manifest.case_ids) or
                    any(type(k) is not str or type(v) is not CalibrationSample
                        for k, v in rows.items())):
                raise CalibratedRetentionError("R29_CASE_SNAPSHOT_INVALID")
            # Frozen CalibrationSample has scalar immutable fields; cloning the
            # exact class breaks aliases to every caller-owned nested dict.
            frozen_samples[generation][arm] = {
                k: CalibrationSample(v.case_id, v.prompt_sha256, v.confidence,
                                     v.correct, v.format_ok, v.refused)
                for k, v in rows.items()
            }
    expected = preregister_calibrated_retention(
        copy.deepcopy(frozen_manifest), copy.deepcopy(frozen_pilot),
        copy.deepcopy(frozen_pilot_receipt), copy.deepcopy(frozen_calibration_plan),
        copy.deepcopy(frozen_calibration_receipt), copy.deepcopy(frozen_plan),
        dict(frozen_roles))
    if (type(plan_receipt) is not CalibratedRetentionReceipt or
            plan_receipt != expected or
            type(plan_receipt.manifest_sha256) is not str or
            type(plan_receipt.canonical_json) is not str or
            type(plan_receipt.role_allocation_sha256) is not str or
            not all(type(x) is bool for x in (
                plan_receipt.independent_oracle_authenticated,
                plan_receipt.physical_model_outputs_verified,
                plan_receipt.release_authorized))):
        raise CalibratedRetentionError("R29_PREREG_RECEIPT_MISMATCH")
    try:
        # R25 always returns HOLD; its hashes bind the declared boolean vectors,
        # not external outputs or an independently authenticated scoring oracle.
        retention = assess_generation_retention(
            copy.deepcopy(frozen_manifest), copy.deepcopy(frozen_records),
            RetentionPolicy(0, 0.0))
    except (AdmissionError, TypeError, AttributeError) as exc:
        raise CalibratedRetentionError("R29_RETENTION_LINEAGE_INVALID") from exc
    arm_for_role = {role: arm for arm, role in frozen_roles.items()}
    case_ids = frozen_manifest.case_ids
    correct_maps = {}
    all_maps = {}
    by_gen = {}
    reasons: list[str] = []
    for generation, record in zip(_GENS, frozen_records):
        maps = frozen_samples[generation]
        try:
            calibration = score_synthetic_calibration(
                copy.deepcopy(frozen_pilot), copy.deepcopy(frozen_pilot_receipt),
                copy.deepcopy(frozen_calibration_plan),
                copy.deepcopy(frozen_calibration_receipt),
                dict(frozen_roles), copy.deepcopy(maps))
        except (CalibrationError, TypeError, AttributeError) as exc:
            raise CalibratedRetentionError("R29_CALIBRATION_SAMPLE_INVALID") from exc
        correct = maps[arm_for_role["CORRECT_LABEL"]]
        if tuple(correct[k].correct for k in case_ids) != record.case_passes:
            raise CalibratedRetentionError("R29_SCORE_VECTOR_VS_CALIBRATION_MISMATCH")
        correct_maps[generation] = correct
        all_maps[generation] = {role: maps[arm] for role, arm in arm_for_role.items()}
        by_gen[generation] = {
            "ece": calibration.correct_label.ece,
            "brier_score": calibration.correct_label.brier_score,
            "accuracy": calibration.correct_label.accuracy,
            "overconfident_error_rate": calibration.correct_label.overconfident_error_rate,
            "refusal_rate": calibration.correct_label.refusal_rate,
            "format_failure_rate": calibration.correct_label.format_failure_rate,
            "zero_update_parity_passed": calibration.zero_update_parity_passed,
            "control_gates_passed": calibration.synthetic_control_gates_passed,
            "calibration_nonregression_vs_frozen": calibration.calibration_nonregression_passed,
        }
        if not calibration.zero_update_parity_passed:
            reasons.append("CONTROL_DRIFT")
        if not calibration.synthetic_control_gates_passed:
            reasons.append("SYNTHETIC_CONTROL_GATE_FAILURE")
        if not calibration.calibration_nonregression_passed:
            reasons.append("CALIBRATION_REGRESSION_VS_FROZEN")
        if any(correct[k].refused and k not in frozen_plan.allowed_abstain_case_ids
               for k in case_ids):
            reasons.append("UNDECLARED_ABSTENTION")
    if "DECLARED_CATASTROPHIC_FORGETTING_RISK" in retention["hold_reasons"]:
        reasons.append("DECLARED_CATASTROPHIC_FORGETTING_RISK")
    if any(not record.zero_control_present or not record.corrupt_control_present
           for record in frozen_records):
        reasons.append("CONTROL_RECEIPT_MISSING")
    for role in ("FROZEN", "ZERO_UPDATE", "CORRUPTED_LABEL"):
        if any(all_maps[g][role] != all_maps["A"][role] for g in ("B", "C")):
            reasons.append("CONTROL_DRIFT")
    transitions = []
    tolerance = frozen_calibration_plan.max_calibration_regression
    for before, after in (("A", "B"), ("B", "C"), ("A", "C")):
        previous, current = correct_maps[before], correct_maps[after]
        regressions = sum(previous[k].correct and not current[k].correct for k in case_ids)
        new_abstentions = sum(not previous[k].refused and current[k].refused
                              for k in case_ids)
        format_regressions = sum(previous[k].format_ok and not current[k].format_ok
                                 for k in case_ids)
        transitions.append({
            "transition": f"{before}->{after}",
            "case_regressions": regressions,
            "case_recoveries": sum(not previous[k].correct and current[k].correct
                                   for k in case_ids),
            "new_abstentions": new_abstentions,
            "format_regressions": format_regressions,
        })
        if regressions:
            reasons.append("DECLARED_CATASTROPHIC_FORGETTING_RISK")
        if new_abstentions:
            reasons.append("ABSTENTION_REGRESSION")
        if format_regressions:
            reasons.append("FORMAT_REGRESSION")
        p, c = by_gen[before], by_gen[after]
        if (c["ece"] > p["ece"] + tolerance + 1e-12 or
                c["brier_score"] > p["brier_score"] + tolerance + 1e-12 or
                c["overconfident_error_rate"] > p["overconfident_error_rate"] + 1e-12 or
                c["refusal_rate"] > p["refusal_rate"] + 1e-12 or
                c["format_failure_rate"] > p["format_failure_rate"] + 1e-12):
            reasons.append("INTERGENERATION_CALIBRATION_REGRESSION")
    if not reasons:
        reasons.append("DECLARED_ONLY_NO_INDEPENDENT_EVIDENCE")
    return {
        "schema": _SCHEMA,
        "status": "HOLD",
        "hold_reasons": list(dict.fromkeys(reasons)),
        "preregistered_manifest_sha256": expected.manifest_sha256,
        "public_source_sha256": frozen_manifest.source_sha256,
        "case_order_sha256": case_order_sha256(case_ids),
        "case_count": len(case_ids),
        "generation_calibration": by_gen,
        "transitions": transitions,
        "evidence_class": "CALLER_DECLARED_PUBLIC_TOY_ONLY",
        "independent_oracle_authenticated": False,
        "physical_model_outputs_verified": False,
        "fresh_reload_verified": False,
        "private_eval_unsealed": False,
        "neural_improvement_proven": False,
        "release_authorized": False,
    }
