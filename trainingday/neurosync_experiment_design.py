"""Source-only preregistration contract for Neuro-Sync ↔ TRAININGDAY.

This module parses and checks a proposed seven-arm donor/receiver experiment.
It performs no model inference, host probing, training, artifact mutation,
rights verification, or physical-launch authorization. A valid plan is NOT
evidence of executed results, independent custody, or learned capabilities.

Provenance at design time:
  neuro-sync main: 89fdef1dd83536cf29875d3de7c2acb07e6f2cfa
  model-training main: 2b83430dd36076c5f65624709968e372b170ea36

Keep this weight-transfer comparison separate from Neuro-Sync's external-state
N/V/P/D/X/F/H routing pilot; matching bus versus file is a distinct question.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA = "bt2.neurosync.transfer-study.v1"
ARMS = (
    "untouched", "context_only", "full_copy", "delta_transfer",
    "distillation", "coupling", "shuffled",
)
ARM_INTERVENTIONS = {
    # Codes are protocol obligations to be independently implemented/verified,
    # never evidence that an implementation or a weight change currently exists.
    "untouched": "NO_INTERVENTION",
    "context_only": "TEMPORARY_CONTEXT_ONLY",
    "full_copy": "DONOR_ADAPTER_DIRECT_COPY",
    "delta_transfer": "EXACT_SHARED_BASE_DELTA",
    "distillation": "APPROVED_OUTPUTS_RECIPIENT_NATIVE_TRAIN",
    "coupling": "PREDECLARED_TRAINABLE_BRIDGE",
    "shuffled": "SHUFFLED_DONOR_OUTPUT_ALIGNMENT",
}
WEIGHT_MODIFICATION_PLANNED = {
    "untouched": False,
    "context_only": False,
    "full_copy": True,
    "delta_transfer": True,
    "distillation": True,
    "coupling": True,
    "shuffled": True,
}
# Guards are proposed conditions, never reports of observed isolation. In
# particular the untouched arm must not be a conveniently starved baseline.
ARM_EXPOSURE_GUARDS = {
    "untouched": "MATCHED_IO_NO_DONOR_AND_NO_NEW_CONTEXT",
    "context_only": "MATCHED_IO_TEMP_CONTEXT_PURGED_BEFORE_COLD_EVAL",
    "full_copy": "APPROVED_DONOR_COPY_ONLY",
    "delta_transfer": "SAME_BASE_TENSOR_ALIGNMENT_REQUIRED",
    "distillation": "RIGHTS_APPROVED_DONOR_OUTPUTS_ONLY",
    "coupling": "BRIDGE_NOT_EXECUTABLE_UNTIL_SEPARATELY_QUALIFIED",
    "shuffled": "DONOR_OUTPUTS_PRESEEDED_WRONG_ALIGNMENT",
}
# One candidate's advantage over a sham baseline is not sufficient causal
# attribution; precommit comparisons to context, copy and negative controls.
REQUIRED_CONTRASTS = (
    ("context_only", "untouched"),
    ("full_copy", "untouched"),
    ("delta_transfer", "untouched"),
    ("delta_transfer", "context_only"),
    ("delta_transfer", "full_copy"),
    ("delta_transfer", "shuffled"),
    ("distillation", "shuffled"),
    ("coupling", "shuffled"),
)
STRATA = (
    "self_identification", "conversation_continuity", "memory_humility",
    "voice_consistency", "general_capability",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,79}\Z")
_MAX_JSON_BYTES = 65536


class PlanHold(ValueError):
    """Invalid preregistration: do not start or interpret an experiment."""


class IndependentResultAuthorityUnavailable(PlanHold):
    """Source lacks host-executed, independent result-verification authority."""


@dataclass(frozen=True)
class ValidatedSourcePlan:
    study_id: str
    canonical_sha256: str
    arm_ids: tuple[str, ...]
    heldout_item_count: int
    disposition: str = "SOURCE_PLAN_VALID__EXECUTION_UNVERIFIED"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, entry in pairs:
        if key in value:
            raise PlanHold(f"duplicate JSON object key: {key}")
        value[key] = entry
    return value


def _reject_json_constant(value: str) -> None:
    raise PlanHold(f"non-finite JSON value forbidden: {value}")


def _object(value: Any, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PlanHold(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise PlanHold(f"{label} fields mismatch: missing={sorted(keys-actual)} extra={sorted(actual-keys)}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanHold(f"{label} must be a nonempty string")
    return value


def _match(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    value = _string(value, label)
    if pattern.fullmatch(value) is None:
        raise PlanHold(f"{label} has invalid form")
    return value


def _positive_int(value: Any, label: str, limit: int = 1_000_000_000) -> int:
    if type(value) is not int or not 0 < value <= limit:
        raise PlanHold(f"{label} must be a positive bounded integer")
    return value


def _required_true(value: Any, label: str) -> None:
    if value is not True:
        raise PlanHold(f"{label} must be explicitly true in the experimental plan")


def validate_study_plan(document: str | bytes | dict[str, Any]) -> ValidatedSourcePlan:
    """Validate a plan's *shape*, not the truth of its claims or future results.

    The sealed heldout manifest, its answers, the arm-to-blind-label mapping,
    training records, and validator secrets must remain external to this plan.
    Even hashes and policy declarations alone do not establish that separation.
    """
    if isinstance(document, bytes):
        if len(document) > _MAX_JSON_BYTES:
            raise PlanHold("plan exceeds size limit")
        try:
            document = document.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PlanHold("plan is not UTF-8") from exc
    if isinstance(document, str):
        try:
            encoded_length = len(document.encode("utf-8"))
        except UnicodeError as exc:
            raise PlanHold("plan must contain valid UTF-8 Unicode") from exc
        if encoded_length > _MAX_JSON_BYTES:
            raise PlanHold("plan exceeds size limit")
        try:
            document = json.loads(
                document,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise PlanHold("invalid JSON") from exc

    root = _object(document, "study", {
        "schema", "study_id", "evidence_class", "source_pins", "models",
        "dataset", "scoring", "controls", "arms", "required_contrasts",
    })
    if root["schema"] != SCHEMA:
        raise PlanHold("unsupported study schema")
    study_id = _match(root["study_id"], "study_id", _IDENT)
    if root["evidence_class"] != "SOURCE_ONLY_PREREGISTRATION":
        raise PlanHold("plan may not claim an executed or authorized evidence class")

    pins = _object(root["source_pins"], "source_pins", {
        "neuro_sync_head", "model_training_head",
    })
    for key, sha in pins.items():
        _match(sha, "source_pins." + key, _HEX40)

    models = _object(root["models"], "models", {
        "donor_id", "recipient_id", "shared_base_sha256", "tokenizer_sha256",
        "donor_checkpoint_sha256", "recipient_checkpoint_sha256",
    })
    for key in ("donor_id", "recipient_id"):
        _match(models[key], "models." + key, _IDENT)
    if models["donor_id"] == models["recipient_id"]:
        raise PlanHold("donor and receiver must have different explicit identities")
    for key in ("shared_base_sha256", "tokenizer_sha256", "donor_checkpoint_sha256", "recipient_checkpoint_sha256"):
        _match(models[key], "models." + key, _HEX64)
    if models["donor_checkpoint_sha256"] == models["recipient_checkpoint_sha256"]:
        raise PlanHold("donor and receiver checkpoints must be distinct")

    data = _object(root["dataset"], "dataset", {
        "approved_train_manifest_sha256", "sealed_heldout_manifest_sha256",
        "data_rights_reference_sha256", "train_items", "heldout_items",
        "heldout_custody",
    })
    for key in ("approved_train_manifest_sha256", "sealed_heldout_manifest_sha256", "data_rights_reference_sha256"):
        _match(data[key], "dataset." + key, _HEX64)
    if len({data["approved_train_manifest_sha256"], data["sealed_heldout_manifest_sha256"], data["data_rights_reference_sha256"]}) != 3:
        raise PlanHold("train, heldout, and rights commitments must differ")
    _positive_int(data["train_items"], "dataset.train_items")
    heldout_count = _positive_int(data["heldout_items"], "dataset.heldout_items")
    if data["heldout_custody"] != "EXTERNAL_INDEPENDENT_CUSTODIAN_PENDING":
        raise PlanHold("custody must be declared pending; this source cannot verify it")

    scoring = _object(root["scoring"], "scoring", {
        "external_scorer_id", "scorer_artifact_sha256", "blind_assignment_commit_sha256",
        "strata", "grading_policy",
    })
    _match(scoring["external_scorer_id"], "scoring.external_scorer_id", _IDENT)
    for key in ("scorer_artifact_sha256", "blind_assignment_commit_sha256"):
        _match(scoring[key], "scoring." + key, _HEX64)
    if scoring["blind_assignment_commit_sha256"] == data["sealed_heldout_manifest_sha256"]:
        raise PlanHold("blinded-assignment and heldout commitments cannot be identical")
    if scoring["grading_policy"] != "EXTERNAL_BLINDED_HELDOUT_ONLY":
        raise PlanHold("only independent blinded heldout grading may qualify outcome claims")
    strata = _object(scoring["strata"], "scoring.strata", set(STRATA))
    if sum(_positive_int(strata[key], "scoring.strata." + key) for key in STRATA) != heldout_count:
        raise PlanHold("heldout strata counts must sum to heldout_items")

    controls = _object(root["controls"], "controls", {
        "study_seed", "max_intervention_tokens_per_arm", "max_eval_tokens_per_arm",
        "matched_starting_base", "matched_eval_budget", "frozen_task_set",
        "fresh_receiver_process_per_arm", "cold_restart_before_heldout",
        "donor_disconnected_during_heldout", "external_state_disabled_during_heldout",
        "no_heldout_answers_in_model_inputs", "independent_blinded_scorer",
        "no_automatic_checkpoint_commit",
    })
    for key in ("study_seed", "max_intervention_tokens_per_arm", "max_eval_tokens_per_arm"):
        _positive_int(controls[key], "controls." + key)
    for key in set(controls) - {"study_seed", "max_intervention_tokens_per_arm", "max_eval_tokens_per_arm"}:
        _required_true(controls[key], "controls." + key)

    contrasts = root["required_contrasts"]
    if not isinstance(contrasts, list) or len(contrasts) != len(REQUIRED_CONTRASTS):
        raise PlanHold("all precommitted causal contrasts must be present")
    try:
        normalized_contrasts = tuple(tuple(pair) for pair in contrasts)
    except TypeError as exc:
        raise PlanHold("invalid causal contrasts") from exc
    if normalized_contrasts != REQUIRED_CONTRASTS:
        raise PlanHold("causal comparisons must include baseline, context, copy and shuffled controls")

    arms = root["arms"]
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        raise PlanHold("exactly seven predefined arms are required")
    seen: set[str] = set()
    for index, arm in enumerate(arms):
        item = _object(arm, f"arms[{index}]", {
            "arm_id", "disposition", "intervention", "weight_modification_planned",
            "same_base_required", "evaluation_plan", "exposure_guard",
        })
        arm_id = _string(item["arm_id"], f"arms[{index}].arm_id")
        if arm_id not in ARMS or arm_id in seen:
            raise PlanHold(f"unknown or duplicated study arm {arm_id}")
        seen.add(arm_id)
        if item["disposition"] != "PROPOSED_UNVERIFIED":
            raise PlanHold("arms may not declare execution or qualification in a source plan")
        if item["intervention"] != ARM_INTERVENTIONS[arm_id]:
            raise PlanHold("arm protocol does not match its registered intervention")
        if item["weight_modification_planned"] is not WEIGHT_MODIFICATION_PLANNED[arm_id]:
            raise PlanHold("arm weight-modification expectation contradicts protocol")
        if item["exposure_guard"] != ARM_EXPOSURE_GUARDS[arm_id]:
            raise PlanHold("arm must declare its precommitted baseline/exposure guard")
        _required_true(item["same_base_required"], f"arms[{index}].same_base_required")
        if item["evaluation_plan"] != "BLINDED_COLD_RECEIVER_ONLY":
            raise PlanHold("each arm must use the blinded cold-receiver evaluation")
    if seen != set(ARMS):
        raise PlanHold("missing required arms")

    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return ValidatedSourcePlan(
        study_id=study_id,
        canonical_sha256=hashlib.sha256(encoded).hexdigest(),
        arm_ids=tuple(ARMS),
        heldout_item_count=heldout_count,
    )


def require_independent_result_qualification(*, plan_sha256: str, purported_receipt: Any) -> None:
    """Unconditionally HOLD promotion of a self-reported result to evidence.

    There is intentionally no source-only success path: the eventual separate
    host/evaluator verifier must check the immutable plan hash, each arm's
    actual before/after artifacts, intervention exposure, cold neural restart,
    independent blind mapping and heldout custody against real host receipts.
    A hash string or positive-looking JSON never supplies that authority.
    """
    raise IndependentResultAuthorityUnavailable(
        "source-only result qualification HARD HOLD: independently authenticated "
        "host/evaluator verification is unavailable"
    )


def example_source_plan() -> dict[str, Any]:
    """Entirely synthetic commitments; intended for contract tests only."""
    return {
        "schema": SCHEMA,
        "study_id": "ns-trainingday-fixture-v1",
        "evidence_class": "SOURCE_ONLY_PREREGISTRATION",
        "source_pins": {
            "neuro_sync_head": "89fdef1dd83536cf29875d3de7c2acb07e6f2cfa",
            "model_training_head": "2b83430dd36076c5f65624709968e372b170ea36",
        },
        "models": {
            "donor_id": "synthetic-A", "recipient_id": "synthetic-B",
            "shared_base_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "donor_checkpoint_sha256": "c" * 64,
            "recipient_checkpoint_sha256": "d" * 64,
        },
        "dataset": {
            "approved_train_manifest_sha256": "e" * 64,
            "sealed_heldout_manifest_sha256": "f" * 64,
            "data_rights_reference_sha256": "0" * 64,
            "train_items": 120, "heldout_items": 50,
            "heldout_custody": "EXTERNAL_INDEPENDENT_CUSTODIAN_PENDING",
        },
        "scoring": {
            "external_scorer_id": "independent-scorer-pending",
            "scorer_artifact_sha256": "1" * 64,
            "blind_assignment_commit_sha256": "2" * 64,
            "strata": dict.fromkeys(STRATA, 10),
            "grading_policy": "EXTERNAL_BLINDED_HELDOUT_ONLY",
        },
        "controls": {
            "study_seed": 317, "max_intervention_tokens_per_arm": 12000,
            "max_eval_tokens_per_arm": 5000,
            "matched_starting_base": True,
            "matched_eval_budget": True,
            "frozen_task_set": True,
            "fresh_receiver_process_per_arm": True,
            "cold_restart_before_heldout": True,
            "donor_disconnected_during_heldout": True,
            "external_state_disabled_during_heldout": True,
            "no_heldout_answers_in_model_inputs": True,
            "independent_blinded_scorer": True,
            "no_automatic_checkpoint_commit": True,
        },
        "required_contrasts": [list(pair) for pair in REQUIRED_CONTRASTS],
        "arms": [
            {"arm_id": arm, "disposition": "PROPOSED_UNVERIFIED",
             "intervention": ARM_INTERVENTIONS[arm],
             "weight_modification_planned": WEIGHT_MODIFICATION_PLANNED[arm],
             "exposure_guard": ARM_EXPOSURE_GUARDS[arm],
             "same_base_required": True, "evaluation_plan": "BLINDED_COLD_RECEIVER_ONLY"}
            for arm in ARMS
        ],
    }
