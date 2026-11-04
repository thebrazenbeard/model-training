"""Fail-closed source protocol for a paired MeZO/QLoRA TRAININGDAY study.

This module prepares a reproducible study schedule and validates untrusted
numeric outcomes. It does not import torch, read model files, unseal DEV data,
start a process, grant an evaluator role, or authorize a training run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence


SCHEMA = "TRAININGDAY_MEZO_QLORA_MATCHED_STUDY_SOURCE_V1"
BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
METHODS = ("MEZO_LORA", "QLORA_ADAMW")
LABEL_MODES = ("correct", "corrupted", "zero")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"case-[0-9a-f]{16}\Z")


class MatchedStudyError(ValueError):
    """A proposed comparison would admit a confound or unverifiable shape."""


def _sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def build_study_plan(*, official_weight_sha256: str,
                     tokenizer_sha256: str, train_sha256: str,
                     private_dev_commitment_sha256: str,
                     lora_target_spec_sha256: str,
                     seeds: Sequence[int] = (113, 271, 997),
                     wall_clock_budget_seconds: int = 600) -> dict:
    """Specify matched *adapter-subspace* trials; deny every physical grant.

    MeZO full-parameter training and QLoRA LoRA-only training do not have a
    matched trainable parameter space. A fair optimizer comparison first uses
    the same target LoRA parameters. Full-parameter MeZO is a *separate*
    systems/capability comparison and must not be reported as this experiment.
    """
    hashes = (official_weight_sha256, tokenizer_sha256, train_sha256,
              private_dev_commitment_sha256, lora_target_spec_sha256)
    if any(not _sha(x) for x in hashes):
        raise MatchedStudyError("all source and evaluator digests must be SHA-256")
    if train_sha256 == private_dev_commitment_sha256:
        raise MatchedStudyError("TRAIN and DEV commitments cannot be identical")
    if (type(seeds) not in (tuple, list) or not 3 <= len(seeds) <= 32
            or any(type(seed) is not int or seed < 0 or seed > 2**31 - 1
                   for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise MatchedStudyError("3-32 distinct bounded integer seeds required")
    if (type(wall_clock_budget_seconds) is not int
            or not 60 <= wall_clock_budget_seconds <= 86_400):
        raise MatchedStudyError("bounded equal wall-clock budget required")

    arms = [
        {"arm_id": f"{method}:{mode}:{seed}", "method": method,
         "label_mode": mode, "seed": seed,
         "trainable_subspace": "IDENTICAL_PINNED_LORA_TARGETS",
         "wall_clock_budget_seconds": wall_clock_budget_seconds}
        for seed in seeds for method in METHODS for mode in LABEL_MODES
    ]
    record = {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_HARD_HOLD",
        "base_id": BASE_ID,
        "base_revision": BASE_REVISION,
        "official_weight_sha256": official_weight_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "train_sha256": train_sha256,
        "private_dev_commitment_sha256": private_dev_commitment_sha256,
        "lora_target_spec_sha256": lora_target_spec_sha256,
        "seeds": list(seeds),
        "methods": list(METHODS),
        "labels": list(LABEL_MODES),
        "arms": arms,
        "base_zero_update_reference_required": True,
        "equal_wall_clock_budget_seconds": wall_clock_budget_seconds,
        "require_equal_example_exposure_record": True,
        "require_forward_and_backward_count_record": True,
        "require_per_arm_checkpoint_reload": True,
        "require_trained_arm_interruption_resume": True,
        "require_independent_blinded_eval": True,
        "allow_full_parameter_mezo_as_matched_arm": False,
        "source_rights_admitted": False,
        "protected_evaluator_custody_verified": False,
        "host_owned_lease_verified": False,
        "authorizes_training": False,
        "authorizes_eval_unseal": False,
    }
    record["plan_sha256"] = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    return record


def validate_untrusted_receipts(plan: Mapping[str, object],
                                receipts: Sequence[Mapping[str, object]]) -> dict:
    """Check coverage and budget accounting, never authenticate a host/run.

    Receipts must contain only scalar public telemetry. Evaluation prompts,
    labels, scores, identity tokens and private DEV rows do not belong here.
    """
    if (type(plan) is not dict or plan.get("schema") != SCHEMA
            or not _sha(plan.get("plan_sha256"))):
        raise MatchedStudyError("invalid study plan")
    copy = dict(plan)
    claimed_sha = copy.pop("plan_sha256")
    if hashlib.sha256(_canonical_bytes(copy)).hexdigest() != claimed_sha:
        raise MatchedStudyError("study plan modified after commitment")
    if (type(plan.get("arms")) is not list or not plan["arms"]
            or any(type(arm) is not dict or type(arm.get("arm_id")) is not str
                   for arm in plan["arms"])):
        raise MatchedStudyError("study arm schedule missing or malformed")
    scheduled = {arm["arm_id"]: arm for arm in plan["arms"]}
    if len(scheduled) != len(plan["arms"]):
        raise MatchedStudyError("duplicate scheduled arm")
    if (type(receipts) not in (list, tuple)
            or len(receipts) != len(scheduled)):
        raise MatchedStudyError("incomplete study arm coverage")
    expected_keys = {"arm_id", "plan_sha256", "elapsed_ms",
                     "optimizer_steps", "examples_seen", "tokens_seen",
                     "forward_calls", "backward_calls", "peak_vram_bytes",
                     "peak_host_ram_bytes", "checkpoint_reloaded",
                     "interruption_resumed"}
    seen = set()
    for receipt in receipts:
        if type(receipt) is not dict or set(receipt) != expected_keys:
            raise MatchedStudyError("unexpected public receipt field or omitted telemetry")
        arm_id = receipt["arm_id"]
        if (type(arm_id) is not str or arm_id not in scheduled
                or arm_id in seen or receipt["plan_sha256"] != claimed_sha):
            raise MatchedStudyError("unknown, duplicate, or replayed arm")
        seen.add(arm_id)
        for field in ("elapsed_ms", "optimizer_steps", "examples_seen",
                      "tokens_seen", "forward_calls", "backward_calls",
                      "peak_vram_bytes", "peak_host_ram_bytes"):
            if type(receipt[field]) is not int or receipt[field] < 0:
                raise MatchedStudyError("invalid nonnegative integer telemetry")
        if not 0 < receipt["elapsed_ms"] <= plan["equal_wall_clock_budget_seconds"] * 1_000:
            raise MatchedStudyError("wall-clock budget exceeded or not measured")
        if (type(receipt["checkpoint_reloaded"]) is not bool
                or type(receipt["interruption_resumed"]) is not bool):
            raise MatchedStudyError("unverifiable reload/resume telemetry shape")
        if not receipt["checkpoint_reloaded"]:
            raise MatchedStudyError("fresh-process checkpoint reload not demonstrated")
        arm = scheduled[arm_id]
        if arm["label_mode"] == "zero":
            if receipt["optimizer_steps"] != 0 or receipt["backward_calls"] != 0:
                raise MatchedStudyError("zero-update control performed updates")
        else:
            if (receipt["optimizer_steps"] < 1 or receipt["examples_seen"] < 1
                    or receipt["tokens_seen"] < 1
                    or receipt["forward_calls"] < 1):
                raise MatchedStudyError("training arm lacks work accounting")
            if not receipt["interruption_resumed"]:
                raise MatchedStudyError("training-arm recovery not demonstrated")
        if arm["method"] == "MEZO_LORA" and receipt["backward_calls"]:
            raise MatchedStudyError("MeZO arm used gradient backpropagation")
        if (arm["method"] == "MEZO_LORA" and arm["label_mode"] != "zero"
                and receipt["forward_calls"] < 2 * receipt["optimizer_steps"]):
            raise MatchedStudyError("MeZO lacks two-sided forward-call accounting")
        if arm["method"] == "QLORA_ADAMW" and arm["label_mode"] != "zero" \
                and receipt["backward_calls"] < receipt["optimizer_steps"]:
            raise MatchedStudyError("QLoRA arm lacks backward-pass accounting")
    return {
        "schema": SCHEMA,
        "status": "SHAPE_VALIDATED_UNAUTHENTICATED_RECEIPTS",
        "plan_sha256": claimed_sha,
        "arms_checked": len(seen),
        "covers_all_declared_arms": seen == set(scheduled),
        "physical_execution_independently_verified": False,
        "data_rights_verified": False,
        "protected_eval_verified": False,
        "proves_learning": False,
    }


def summarize_paired_binary_outcomes(
    reference: Mapping[str, bool], challenger: Mapping[str, bool],
) -> dict:
    """Return only aggregate paired binary metrics, with exact McNemar p.

    This is deliberately an *untrusted* score-shape computation, not a
    validated evaluator. Its caller must independently protect unseen DEV
    examples, identities, assignment and disclosure.
    """
    if (type(reference) is not dict or type(challenger) is not dict
            or not 32 <= len(reference) <= 1024
            or set(reference) != set(challenger)
            or any(type(k) is not str or _CASE_ID.fullmatch(k) is None
                   or type(reference[k]) is not bool
                   or type(challenger[k]) is not bool for k in reference)):
        raise MatchedStudyError("invalid or mismatched case-identity/outcome sets")
    gains = sum(not reference[k] and challenger[k] for k in reference)
    losses = sum(reference[k] and not challenger[k] for k in reference)
    discordant = gains + losses
    tail = sum(math.comb(discordant, k)
               for k in range(min(gains, losses) + 1))
    pvalue = min(1.0, 2 * tail / (2 ** discordant))
    return {
        "status": "UNTRUSTED_PAIRED_SCORE_SHAPE_ONLY",
        "cases": len(reference),
        "reference_correct": sum(reference.values()),
        "challenger_correct": sum(challenger.values()),
        "net_improvement_cases": gains - losses,
        "paired_gain": gains,
        "paired_loss": losses,
        "mcnemar_exact_two_sided_p": pvalue,
        "independent_eval_verified": False,
        "generalization_proven": False,
    }
