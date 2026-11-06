"""Public synthetic multi-turn study design and declared causal-control audit.

Not an inference engine, a scorer, or a held-out evaluation bank. No model
responses are generated here. A validated owner-supplied transcript is still
only a declared transcript, not evidence of authentic Vera model behavior.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Mapping, Sequence

from .multiturn_protocol import validate_sessions


PUBLIC_PLAN_SCHEMA = "VERA_PUBLIC_MULTITURN_5ARM_3BUDGET_2FEEDBACK_V1"
ARMS = (
    "base", "persona_base", "r16_zero_step", "r16_correct8", "r16_corrupt8",
)
BUDGETS = (64, 96, 128)
CASES = {
    "public_control": (
        "In this invented workshop story, a crate starts on shelf A. Where is it?",
        "We are still discussing the imaginary crate. What shelf was named first?",
        "What shelf should we use for the invented crate now?",
    ),
    "public_feedback": (
        "In this invented workshop story, a crate starts on shelf A. Where is it?",
        "Correction: the invented crate has moved from shelf A to shelf C. Where now?",
        "What shelf should we use for the invented crate now?",
    ),
}
GENERATION = {"context_window_tokens": 4096, "temperature_milli": 700, "rng_seed": 17}
_NO_ADAPTER = {"base", "persona_base"}
_WITH_PROFILE = set(ARMS) - {"base"}
_R16_PILOTS = {"r16_zero_step", "r16_correct8", "r16_corrupt8"}


class CausalControlError(ValueError):
    """Incomplete or internally confounded *declared* public study design."""


def _plan_cases() -> tuple[tuple[str, str, int], ...]:
    return tuple((f"{case_id}_b{budget}", case_id, budget)
                 for budget in BUDGETS for case_id in CASES)


def public_plan() -> dict:
    """Reproducible original, fictional USER-only script matrix; no answers."""
    cases = {key: list(value) for key, value in CASES.items()}
    cells = [
        {"arm": arm, "case_id": full_case, "budget": budget,
         "profile_intent": "none" if arm == "base" else "external_versioned_required",
         "feedback_scripted": case_id == "public_feedback"}
        for full_case, case_id, budget in _plan_cases() for arm in ARMS
    ]
    commitment = {
        "schema": PUBLIC_PLAN_SCHEMA, "arm_roles": list(ARMS),
        "budgets": list(BUDGETS), "public_cases": cases,
        "generation_except_reply_budget": dict(GENERATION), "cells": cells,
    }
    digest = hashlib.sha256(json.dumps(
        commitment, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        **commitment,
        "plan_sha256": digest,
        "planned_cell_count": len(cells),
        "reply_budgets": list(BUDGETS),
        "public_case_scripts": cases,
        "provenance_class": "ORIGINAL_FICTIONAL_PUBLIC_PROTOCOL",
        "heldout": False,
        "training_authorized": False,
        "owner_rubric_thresholds_established": False,
        "physical_inference_runs": 0,
        "semantic_quality_score": None,
        "protected_evaluator_custody_attested": False,
    }


def audit_public_study(rows: Sequence[Mapping[str, object]]) -> dict:
    """Fail confounded declared comparisons; return structural counts only.

    This function NEVER scores replies. Claimed source, base/adapter hashes and
    profile prompt digests cannot independently attest any loaded model bytes.
    """
    records = validate_sessions(rows)
    expected = {(arm, full_case) for full_case, _, _ in _plan_cases() for arm in ARMS}
    observed = {(row["arm"], row["case_id"]) for row in records}
    if observed != expected or len(records) != len(expected):
        raise CausalControlError("five arms and both public scripts required at all budgets")
    source = {row["source_claim"] for row in records}
    if len(source) != 1 or not source <= {"mock_test", "public_exposed"}:
        raise CausalControlError("do not mix synthetic, public claims or private sources")
    bases = {row["base_sha256"] for row in records}
    origins = {row["origin_claim"] for row in records}
    if len(bases) != 1 or len(origins) != 1:
        raise CausalControlError("identical claimed base digest and origin text required for each arm")

    by_arm: dict[str, dict] = {}
    censor = Counter()
    known_cases = {full_case: (case_id, budget) for full_case, case_id, budget in _plan_cases()}
    for row in records:
        arm = row["arm"]
        if arm not in ARMS:
            raise CausalControlError("unsupported arm role")
        key = row["case_id"]
        if key not in known_cases:
            raise CausalControlError("unsupported public case")
        case_id, budget = known_cases[key]
        if tuple(turn["user"] for turn in row["turns"]) != CASES[case_id]:
            raise CausalControlError("public scripted user turns changed or reordered")
        if len(row["turns"]) != 3:
            raise CausalControlError("three complete user-turn attempts per script required")
        settings = row["generation"]
        if (settings["max_reply_tokens"] != budget or
                any(settings[name] != value for name, value in GENERATION.items())):
            raise CausalControlError("unmatched budget or public sampling/seed declaration")
        digest = row["adapter_sha256"]
        if arm in _NO_ADAPTER and digest != "NO_ADAPTER":
            raise CausalControlError("base/persona baseline must claim no adapter")
        if arm in _R16_PILOTS and digest == "NO_ADAPTER":
            raise CausalControlError("R16 technical controls must declare adapter digests")
        if arm in by_arm and (by_arm[arm]["adapter_sha256"] != digest or
                              by_arm[arm]["prompt_sha256"] != row["prompt_sha256"]):
            raise CausalControlError("arm model/prompt declaration changed across budgets")
        by_arm[arm] = row
        censor.update(turn["outcome"] for turn in row["turns"])

    profiles = {by_arm[arm]["prompt_sha256"] for arm in _WITH_PROFILE}
    if len(profiles) != 1 or by_arm["base"]["prompt_sha256"] in profiles:
        raise CausalControlError("single declared persona prompt, distinct from unprofiled base, required")
    adapter_hashes = {by_arm[arm]["adapter_sha256"] for arm in _R16_PILOTS}
    if len(adapter_hashes) != len(_R16_PILOTS):
        raise CausalControlError("three distinct R16 pilot adapter digest declarations required")

    return {
        "schema": PUBLIC_PLAN_SCHEMA,
        "evidence_class": "DECLARED_PUBLIC_CAPTURE_FORMAT_ONLY",
        "public_plan_sha256": public_plan()["plan_sha256"],
        "source_claim": next(iter(source)),
        "declared_slots_checked": len(records),
        "declared_completed_turns": censor["complete"],
        "declared_technical_censored_turns": sum(censor[state] for state in
                                               ("timeout", "truncated", "upstream_error")),
        "declared_5arm_3budget_2feedback_matrix_complete": True,
        "declared_same_base_digest_origin_claim_and_sampling": True,
        "official_base_revision_attested": False,
        "declared_profile_and_pilot_control_separation": True,
        "physical_base_weight_bytes_rehashed": False,
        "authentic_model_inference_attested": False,
        "runtime_process_and_session_continuity_attested": False,
        "prompt_profile_physically_applied_attested": False,
        "protected_holdout_used": False,
        "independent_review_performed": False,
        "owner_rubric_thresholds_established": False,
        "semantic_improvement_measured": False,
        "raw_dialogue_emitted": False,
    }


if __name__ == "__main__":
    # Public USER-only plan. No live execution or assistant text is generated.
    print(json.dumps(public_plan(), ensure_ascii=False, indent=2, sort_keys=True))
