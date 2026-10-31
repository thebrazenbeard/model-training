"""Public CPU toy four-control pairing; never physical optimizer qualification.

R29's source scalar CE witness checks two optimizer arms, but the declared
zero/frozen/correct/corrupted hashes alone do not establish matched control
cohort, prompt, exposure schedule, or label masks. This independent bounded
consumer checks those *declarations*, not physical control receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from trainingday.r29_optimizer_fairness import ToyTokenCase, _validate_cases


SCHEMA = "VERA_TRAININGDAY_R30_PUBLIC_CONTROL_PAIRING_V1"
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_EXPECTED_ROLES = ("FROZEN", "ZERO_UPDATE", "CORRECT_LABEL", "CORRUPTED_LABEL")
_ALLOWED_OPTIMIZERS = frozenset({"MEZO", "QLORA"})
_MAX_CASES, _MAX_TOKENS, _MAX_VOCAB, _MAX_PRESENTATIONS = 64, 256, 128, 1024


@dataclass(frozen=True)
class ControlArm:
    role: str
    optimizer_method: str
    cases: tuple[ToyTokenCase, ...]
    presented_case_ids: tuple[str, ...]
    optimizer_updates: int
    initial_adapter_sha256: str
    final_adapter_sha256: str


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _snapshot_arm(raw: object) -> ControlArm:
    """Capture all nested toy fields before any source validator is invoked."""
    if type(raw) is not ControlArm:
        raise ValueError("control declaration must use exact public dataclass")
    if (type(raw.role) is not str or type(raw.optimizer_method) is not str
            or type(raw.cases) is not tuple
            or not 1 <= len(raw.cases) <= _MAX_CASES
            or type(raw.presented_case_ids) is not tuple
            or len(raw.presented_case_ids) > _MAX_PRESENTATIONS
            or any(type(x) is not str or len(x) > 128 for x in raw.presented_case_ids)
            or type(raw.optimizer_updates) is not int
            or not 0 <= raw.optimizer_updates <= _MAX_PRESENTATIONS
            or any(type(x) is not str or _SHA.fullmatch(x) is None
                   for x in (raw.initial_adapter_sha256, raw.final_adapter_sha256))):
        raise ValueError("control declaration malformed or unbounded")
    cases = []
    for case in raw.cases:
        if (type(case) is not ToyTokenCase or type(case.case_id) is not str
                or type(case.input_ids) is not tuple
                or not 2 <= len(case.input_ids) <= _MAX_TOKENS
                or type(case.labels) is not tuple
                or len(case.labels) != len(case.input_ids)
                or any(type(n) is not int for n in case.input_ids + case.labels)
                or type(case.target_logits) is not tuple
                or len(case.target_logits) > _MAX_TOKENS):
            raise ValueError("public toy control case malformed or oversized")
        logits = []
        for row in case.target_logits:
            if (type(row) is not tuple or not 2 <= len(row) <= _MAX_VOCAB
                    or any(type(n) not in (int, float) for n in row)):
                raise ValueError("public toy logits malformed or oversized")
            logits.append(tuple(row))
        cases.append(ToyTokenCase(case.case_id, tuple(case.input_ids),
                                  tuple(case.labels), tuple(logits)))
    return ControlArm(raw.role, raw.optimizer_method, tuple(cases),
                      tuple(raw.presented_case_ids), raw.optimizer_updates,
                      raw.initial_adapter_sha256, raw.final_adapter_sha256)


def assess_control_quartet(arms: object) -> dict:
    """Check public-toy control comparability, always deny effects.

    The labels in completion positions may differ in the corrupted arm, and
    causal LM input tokens in those positions can therefore differ too.
    Prompt/masked positions, case order, and supervision mask must not differ.
    """
    if type(arms) is not tuple or len(arms) != 4:
        raise ValueError("exact four-control tuple required")
    frozen, zero, correct, corrupt = tuple(_snapshot_arm(arm) for arm in arms)
    private = (frozen, zero, correct, corrupt)
    if tuple(a.role for a in private) != _EXPECTED_ROLES:
        raise ValueError("missing or reordered control role")
    if (len({a.optimizer_method for a in private}) != 1
            or private[0].optimizer_method not in _ALLOWED_OPTIMIZERS):
        raise ValueError("control optimizer method mismatch")
    for arm in private:
        _validate_cases(arm.cases)

    reasons: list[str] = []
    def block(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    baseline = correct.cases
    ids = tuple(row.case_id for row in baseline)
    initial = correct.initial_adapter_sha256
    if any(a.initial_adapter_sha256 != initial for a in private):
        block("INITIAL_ADAPTER_MISMATCH")
    for control in (frozen, zero):
        if control.optimizer_updates or control.presented_case_ids:
            block("ZERO_UPDATE_CONTROL_HAS_EXPOSURE")
        if control.final_adapter_sha256 != control.initial_adapter_sha256:
            block("ZERO_UPDATE_CONTROL_CHANGED")
        if (tuple(row.case_id for row in control.cases) != ids
                or any((a.input_ids, a.labels) != (b.input_ids, b.labels)
                       for a, b in zip(control.cases, baseline))):
            block("BASELINE_CONTROL_DATA_MISMATCH")
        if any(a.target_logits != b.target_logits
               for a, b in zip(control.cases, baseline)):
            block("FROZEN_LOGIT_SNAPSHOT_MISMATCH")

    if (correct.optimizer_updates == 0 or corrupt.optimizer_updates == 0):
        block("LABELED_CONTROL_NO_UPDATES")
    if (correct.final_adapter_sha256 == correct.initial_adapter_sha256
            or corrupt.final_adapter_sha256 == corrupt.initial_adapter_sha256):
        block("LABELED_CONTROL_TENSOR_UNCHANGED")
    if correct.optimizer_updates != corrupt.optimizer_updates:
        block("LABELED_UPDATE_COUNT_MISMATCH")
    if (correct.presented_case_ids != corrupt.presented_case_ids
            or len(correct.presented_case_ids) == 0):
        block("CONTROL_EXPOSURE_ORDER_MISMATCH")
    if (set(correct.presented_case_ids) != set(ids)
            or set(corrupt.presented_case_ids) != set(ids)):
        block("CONTROL_CASE_EXPOSURE_INCOMPLETE")
    if tuple(row.case_id for row in corrupt.cases) != ids:
        block("CASE_ORDER_MISMATCH")
    else:
        changed_target = False
        for good, bad in zip(baseline, corrupt.cases, strict=True):
            good_mask = tuple(x != -100 for x in good.labels)
            bad_mask = tuple(x != -100 for x in bad.labels)
            if good_mask != bad_mask:
                block("TARGET_MASK_MISMATCH")
                continue
            # In a frozen deterministic causal LM, logits predicting token
            # at position i depend only on input_ids[:i]. A corrupted target
            # label is not permission to alter logits when that prefix is
            # still identical. Later changed completion inputs may legitimately
            # diverge, so do not demand whole-sequence logit equality.
            positions = (index for index, supervised in enumerate(good_mask)
                         if supervised)
            for index, good_logits, corrupt_logits in zip(
                    positions, good.target_logits, bad.target_logits, strict=True):
                if (good.input_ids[:index] == bad.input_ids[:index]
                        and good_logits != corrupt_logits):
                    block("FROZEN_PREFIX_LOGIT_MISMATCH")
            if any(a != b for a, b, supervised in
                   zip(good.input_ids, bad.input_ids, good_mask, strict=True)
                   if not supervised):
                block("PROMPT_CONTEXT_MISMATCH")
            if any(a != b for a, b, supervised in
                   zip(good.labels, bad.labels, good_mask, strict=True)
                   if supervised):
                changed_target = True
        if not changed_target:
            block("CORRUPTED_LABEL_NOT_DISTINCT")

    return {
        "schema": SCHEMA,
        "status": "HOLD",
        "evidence_class": "PUBLIC_SYNTHETIC_CONTROL_DECLARATIONS_ONLY",
        "hold_reasons": reasons,
        "synthetic_control_conditions_satisfied": not reasons,
        "correct_presentations": len(correct.presented_case_ids),
        "corrupt_presentations": len(corrupt.presented_case_ids),
        "correct_target_id_commitment_sha256": _digest([
            (row.case_id, row.input_ids, row.labels) for row in correct.cases]),
        "corrupt_target_id_commitment_sha256": _digest([
            (row.case_id, row.input_ids, row.labels) for row in corrupt.cases]),
        "presented_case_order_sha256": _digest(correct.presented_case_ids),
        "independent_control_custody_verified": False,
        "runtime_optimizer_defaults_verified": False,
        "physical_equal_time_verified": False,
        "neural_tensor_update_verified": False,
        "behavior_improvement_verified": False,
        "training_authorized": False,
        "model_promotion_authorized": False,
    }
