from __future__ import annotations

from collections import defaultdict
import json
from typing import Callable

from successor.experiments.retention_audit_arch_v2_reviewer import (
    ollama_text,
)


DEFECT_CLASSES_V3 = (
    "referent_ambiguity",
    "prompt_grader_semantic_mismatch",
    "contradictory_instructions",
    "underspecified_output_format",
    "evidence_policy_ambiguity",
    "scope_negation_ambiguity",
)


def _fresh_control_text(
    defect_class: str,
    *,
    defect: bool,
    variant: int,
) -> tuple[str, dict]:
    token = f"R{variant:02d}"

    if defect_class == "referent_ambiguity":
        if defect:
            return (
                f"Morgan told Riley after Casey arrived that they should log {token}. "
                "Return only the name of the person who should log it.",
                {
                    "candidate_referents": ["Morgan", "Riley", "Casey"],
                    "required_answer_type": "person name",
                },
            )
        return (
            f"Morgan told Riley, 'Riley, log {token} now.' "
            "Return only the name of the person who should log it.",
            {
                "candidate_referents": ["Morgan", "Riley"],
                "explicit_addressee": "Riley",
            },
        )

    if defect_class == "prompt_grader_semantic_mismatch":
        if defect:
            return (
                f"Return the recorded city name for identifier {token}.",
                {
                    "grader_checks": "recorded population count",
                    "identifier": token,
                },
            )
        return (
            f"Return only the recorded population count for identifier {token}.",
            {
                "grader_checks": "recorded population count",
                "identifier": token,
            },
        )

    if defect_class == "contradictory_instructions":
        if defect:
            return (
                f"Return exactly two copies of {token} and return exactly one copy "
                f"of {token}.",
                {"constraint_a": "exactly two copies", "constraint_b": "exactly one copy"},
            )
        return (
            f"Return exactly two copies of {token}, separated by one space, "
            "with no punctuation.",
            {
                "constraint_a": "exactly two copies",
                "separator": "one space",
                "punctuation": "none",
            },
        )

    if defect_class == "underspecified_output_format":
        if defect:
            return (
                f"Provide the profile for record {token}.",
                {
                    "available_fields": ["id", "label", "count"],
                    "acceptable_shapes": ["plain text", "JSON", "key-value pairs"],
                },
            )
        return (
            f"Return exactly one compact JSON object with keys "
            f"'id','label','count' for record {token}, in that key order.",
            {
                "required_shape": {
                    "id": token,
                    "label": f"record-{variant}",
                    "count": variant + 2,
                },
                "key_order": ["id", "label", "count"],
            },
        )

    if defect_class == "evidence_policy_ambiguity":
        if defect:
            return (
                f"Use the supplied note for {token}; if it is incomplete, use "
                "ordinary background knowledge to finish the answer.",
                {
                    "note": f"{token} has status recorded.",
                    "outside_knowledge_rule": "unclear priority versus supplied note",
                },
            )
        return (
            f"Use only the supplied note for {token}. If the note does not contain "
            "the requested fact, answer exactly NOT_ENOUGH_INFO.",
            {
                "note": f"{token} has status recorded.",
                "outside_knowledge_rule": "forbidden",
            },
        )

    if defect_class == "scope_negation_ambiguity":
        if defect:
            return (
                f"For {token}, exclude entries that are not unapproved.",
                {
                    "candidate_states": ["approved", "unapproved", "pending"],
                    "negation": "not unapproved",
                },
            )
        return (
            f"For {token}, include only entries whose state is approved; exclude "
            "unapproved and pending entries.",
            {
                "candidate_states": ["approved", "unapproved", "pending"],
                "included_state": "approved",
            },
        )

    raise ValueError(f"unknown V3 defect class:{defect_class}")


def build_reviewer_qualification_controls_v3() -> list[dict]:
    controls: list[dict] = []
    for defect_class in DEFECT_CLASSES_V3:
        for variant in range(4):
            for expected_defect in (True, False):
                prompt, context = _fresh_control_text(
                    defect_class,
                    defect=expected_defect,
                    variant=variant,
                )
                polarity = "defect" if expected_defect else "good"
                controls.append({
                    "case_id": (
                        f"arch-v3-reviewer-{defect_class}-{polarity}-{variant}"
                    ),
                    "defect_class": defect_class,
                    "expected_defect": expected_defect,
                    "prompt": prompt,
                    "context": context,
                })
    return controls


def qualification_batch_prompt_v3(controls: list[dict]) -> str:
    if not controls:
        raise ValueError("V3 qualification controls must be non-empty")
    visible = []
    case_ids: set[str] = set()
    for control in controls:
        case_id = control.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("V3 qualification case_id invalid")
        if case_id in case_ids:
            raise ValueError("duplicate V3 qualification case_id")
        case_ids.add(case_id)
        visible.append({
            "case_id": case_id,
            "defect_class": control.get("defect_class"),
            "prompt": control.get("prompt"),
            "context": control.get("context"),
        })

    payload = json.dumps(
        {"controls": visible},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        "Evaluate each control only for its named semantic defect class. "
        "Return strict JSON with exactly one top-level key named reviews. "
        "reviews must contain exactly one object per control with exactly these "
        "keys: case_id, observed_defect, defect_class, witness, reason. "
        "observed_defect must be a JSON boolean. witness must always be present. "
        "When observed_defect is false, witness must be exactly NONE. "
        "When observed_defect is true, witness must be a short concrete example "
        "showing the ambiguity, contradiction, mismatch, underspecification, or "
        "scope problem; never use NONE for a claimed defect. Do not infer or "
        "discuss any hidden expected label.\n"
        f"{payload}"
    )


def _parse_v3_row(row: dict, *, control: dict) -> dict:
    required = {
        "case_id",
        "observed_defect",
        "defect_class",
        "witness",
        "reason",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("V3 qualification row schema mismatch")
    if row["case_id"] != control.get("case_id"):
        raise ValueError("V3 qualification case_id mismatch")
    if row["defect_class"] != control.get("defect_class"):
        raise ValueError("V3 qualification defect_class mismatch")
    if not isinstance(row["observed_defect"], bool):
        raise ValueError("V3 observed_defect must be boolean")
    if not isinstance(row["reason"], str) or not row["reason"].strip():
        raise ValueError("V3 qualification reason must be non-empty")
    witness = row["witness"]
    if witness is not None and not isinstance(witness, str):
        raise ValueError("V3 witness must be string or null")
    return dict(row)


def parse_qualification_batch_response_v3(
    text: str,
    *,
    controls: list[dict],
) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid V3 qualification JSON:{exc}") from exc
    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("V3 qualification batch must contain only reviews")
    rows = value["reviews"]
    if not isinstance(rows, list) or len(rows) != len(controls):
        raise ValueError("V3 qualification review count mismatch")

    by_id = {control["case_id"]: control for control in controls}
    parsed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("V3 qualification review must be object")
        case_id = row.get("case_id")
        if case_id not in by_id or case_id in parsed:
            raise ValueError("V3 qualification case_id mismatch or duplicate")
        parsed[case_id] = _parse_v3_row(
            row,
            control=by_id[case_id],
        )

    if set(parsed) != set(by_id):
        raise ValueError("V3 qualification case set mismatch")
    return [parsed[control["case_id"]] for control in controls]


def _is_contradiction(result: dict) -> bool:
    witness = result.get("witness")
    observed = result.get("observed_defect")
    if observed is True:
        return (
            witness is None
            or not isinstance(witness, str)
            or not witness.strip()
            or witness.strip().upper() == "NONE"
        )
    if observed is False:
        return (
            not isinstance(witness, str)
            or witness.strip().upper() != "NONE"
        )
    return True


def qualify_reviewer_v3(call_control: Callable[[dict], dict]) -> dict:
    controls = build_reviewer_qualification_controls_v3()
    true_positive = 0
    true_negative = 0
    defect_total = 0
    good_total = 0
    per_class_hits: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    contradictions: list[str] = []
    rows: list[dict] = []

    for control in controls:
        result = call_control(control)
        parsed = _parse_v3_row(result, control=control)
        observed = parsed["observed_defect"]
        expected = control["expected_defect"]
        defect_class = control["defect_class"]

        if _is_contradiction(parsed):
            contradictions.append(control["case_id"])

        if expected:
            defect_total += 1
            per_class_total[defect_class] += 1
            if observed:
                true_positive += 1
                per_class_hits[defect_class] += 1
        else:
            good_total += 1
            if not observed:
                true_negative += 1

        rows.append({
            "case_id": control["case_id"],
            "defect_class": defect_class,
            "expected_defect": expected,
            "observed_defect": observed,
            "witness": parsed["witness"],
            "reason": parsed["reason"],
        })

    per_class = {
        defect_class: [
            per_class_hits[defect_class],
            per_class_total[defect_class],
        ]
        for defect_class in DEFECT_CLASSES_V3
    }
    qualified = (
        true_positive >= 22
        and defect_total == 24
        and true_negative >= 22
        and good_total == 24
        and all(value[0] >= 3 and value[1] == 4 for value in per_class.values())
        and not contradictions
    )
    return {
        "schema": "RETENTION_AUDIT_ARCH_V3_REVIEWER_QUALIFICATION_V1",
        "status": (
            "REVIEWER_QUALIFIED"
            if qualified
            else "REVIEWER_UNQUALIFIED"
        ),
        "sensitivity": [true_positive, defect_total],
        "specificity": [true_negative, good_total],
        "per_class": per_class,
        "structured_contradictions": contradictions,
        "controls": rows,
    }


def run_batched_reviewer_qualification_v3(
    *,
    call_text: Callable[[str], str] = ollama_text,
    batch_size: int = 4,
    max_attempts: int = 3,
) -> dict:
    if batch_size < 1:
        raise ValueError("V3 batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("V3 max_attempts must be positive")

    controls = build_reviewer_qualification_controls_v3()
    parsed_by_id: dict[str, dict] = {}
    batches: list[dict] = []

    for start in range(0, len(controls), batch_size):
        batch = controls[start:start + batch_size]
        prompt = qualification_batch_prompt_v3(batch)
        failures: list[str] = []
        parsed: list[dict] | None = None
        attempts = 0

        for attempts in range(1, max_attempts + 1):
            try:
                response_text = call_text(prompt)
                if (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                ):
                    raise ValueError("empty V3 qualification response")
                parsed = parse_qualification_batch_response_v3(
                    response_text,
                    controls=batch,
                )
                break
            except (ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    f"attempt_{attempts}:{type(exc).__name__}:{exc}"
                )

        if parsed is None:
            return {
                "schema": "RETENTION_AUDIT_ARCH_V3_REVIEWER_QUALIFICATION_V1",
                "status": "REVIEWER_UNQUALIFIED",
                "reason": "qualification_transport_failure",
                "batch_size": batch_size,
                "batch_count": len(batches) + 1,
                "failed_batch_case_ids": [
                    control["case_id"] for control in batch
                ],
                "transport_failures": failures,
                "batches": batches,
            }

        for row in parsed:
            parsed_by_id[row["case_id"]] = row
        batches.append({
            "case_ids": [control["case_id"] for control in batch],
            "attempts": attempts,
            "prior_attempt_failures": failures,
        })

    result = qualify_reviewer_v3(
        lambda control: parsed_by_id[control["case_id"]]
    )
    result["batch_size"] = batch_size
    result["batch_count"] = len(batches)
    result["batches"] = batches
    return result
