from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Callable

from successor.experiments.retention_audit_arch_v2_reviewer import ollama_text


DEFECT_CLASSES_V4 = (
    "referent_ambiguity",
    "prompt_grader_semantic_mismatch",
    "contradictory_instructions",
    "underspecified_output_format",
    "evidence_policy_ambiguity",
    "scope_negation_ambiguity",
)


def _fresh_control_text_v4(
    defect_class: str,
    *,
    defect: bool,
    variant: int,
) -> tuple[str, dict]:
    token = f"S{variant:02d}"

    if defect_class == "referent_ambiguity":
        if defect:
            return (
                f"Avery told Blake after Devon arrived that they should archive "
                f"{token}. Return only the name of the person who should archive it.",
                {
                    "candidate_referents": ["Avery", "Blake", "Devon"],
                    "required_answer_type": "person name",
                },
            )
        return (
            f"Avery told Blake, 'Blake, archive {token} now.' Return only the "
            "name of the person who should archive it.",
            {
                "candidate_referents": ["Avery", "Blake"],
                "explicit_addressee": "Blake",
            },
        )

    if defect_class == "prompt_grader_semantic_mismatch":
        if defect:
            return (
                f"Return the recorded label for entry {token}.",
                {
                    "grader_checks": "recorded item count",
                    "entry": token,
                },
            )
        return (
            f"Return only the recorded item count for entry {token}.",
            {
                "grader_checks": "recorded item count",
                "entry": token,
            },
        )

    if defect_class == "contradictory_instructions":
        if defect:
            return (
                f"Return exactly three copies of {token} and exactly one copy "
                f"of {token}.",
                {
                    "constraint_a": "exactly three copies",
                    "constraint_b": "exactly one copy",
                },
            )
        return (
            f"Return exactly three copies of {token}, separated by single "
            "spaces, with no punctuation.",
            {
                "constraint_a": "exactly three copies",
                "separator": "single space",
                "punctuation": "none",
            },
        )

    if defect_class == "underspecified_output_format":
        if defect:
            return (
                f"Send the complete entry for {token}.",
                {
                    "available_fields": ["id", "status", "count"],
                    "acceptable_shapes": [
                        "plain text",
                        "JSON",
                        "key-value pairs",
                    ],
                },
            )
        return (
            f"Return exactly one compact JSON object with keys "
            f"'id','status','count' for {token}, in that key order.",
            {
                "required_shape": {
                    "id": token,
                    "status": f"state-{variant}",
                    "count": variant + 5,
                },
                "key_order": ["id", "status", "count"],
            },
        )

    if defect_class == "evidence_policy_ambiguity":
        if defect:
            return (
                f"Use the supplied memo for {token}; if it leaves a gap, fill "
                "the gap from ordinary background knowledge when helpful.",
                {
                    "memo": f"{token} has a recorded status.",
                    "outside_knowledge_rule": "priority unclear",
                },
            )
        return (
            f"Use only the supplied memo for {token}. If the requested fact is "
            "not in the memo, answer exactly NOT_ENOUGH_INFO.",
            {
                "memo": f"{token} has a recorded status.",
                "outside_knowledge_rule": "forbidden",
            },
        )

    if defect_class == "scope_negation_ambiguity":
        if defect:
            return (
                f"For {token}, remove entries that are not inactive.",
                {
                    "candidate_states": ["active", "inactive", "pending"],
                    "scope_phrase": "not inactive",
                },
            )
        return (
            f"For {token}, keep only inactive entries; remove active and "
            "pending entries.",
            {
                "candidate_states": ["active", "inactive", "pending"],
                "included_state": "inactive",
            },
        )

    raise ValueError(f"unknown V4 defect class:{defect_class}")


def build_reviewer_qualification_controls_v4() -> list[dict]:
    controls: list[dict] = []
    for defect_class in DEFECT_CLASSES_V4:
        for variant in range(4):
            for expected_defect in (True, False):
                prompt, context = _fresh_control_text_v4(
                    defect_class,
                    defect=expected_defect,
                    variant=variant,
                )
                polarity = "defect" if expected_defect else "good"
                controls.append({
                    "case_id": (
                        f"arch-v4-reviewer-{defect_class}-{polarity}-{variant}"
                    ),
                    "defect_class": defect_class,
                    "expected_defect": expected_defect,
                    "prompt": prompt,
                    "context": context,
                })
    return controls


def qualification_batch_prompt_v4(controls: list[dict]) -> str:
    if not controls:
        raise ValueError("V4 qualification controls must be non-empty")
    visible = []
    seen: set[str] = set()
    for control in controls:
        case_id = control.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("V4 qualification case_id invalid")
        if case_id in seen:
            raise ValueError("duplicate V4 qualification case_id")
        seen.add(case_id)
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
        "Each review must contain exactly case_id, observed_defect, "
        "defect_class, witness, reason. observed_defect must be a JSON boolean. "
        "When observed_defect is false, witness must be the exact string NONE. "
        "When observed_defect is true, witness must contain a concrete semantic "
        "example. A concrete witness may be a nonempty JSON string, array, or "
        "object. Do not use null, numbers, booleans, empty containers, or NONE "
        "for a claimed defect. Do not infer or discuss hidden expected labels.\n"
        f"{payload}"
    )


def witness_text_v4(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def meaningful_witness_v4(value: object) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and stripped.upper() != "NONE"
    if isinstance(value, list):
        return bool(value) and any(meaningful_witness_v4(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and any(
            meaningful_witness_v4(item)
            for item in value.values()
        )
    return False


def _parse_v4_row(row: dict, *, control: dict) -> dict:
    required = {
        "case_id",
        "observed_defect",
        "defect_class",
        "witness",
        "reason",
    }
    allowed = required | {"witness_text"}
    if (
        not isinstance(row, dict)
        or not required <= set(row)
        or not set(row) <= allowed
    ):
        raise ValueError("V4 qualification row schema mismatch")
    if row["case_id"] != control.get("case_id"):
        raise ValueError("V4 qualification case_id mismatch")
    if row["defect_class"] != control.get("defect_class"):
        raise ValueError("V4 qualification defect_class mismatch")
    if not isinstance(row["observed_defect"], bool):
        raise ValueError("V4 observed_defect must be boolean")
    if not isinstance(row["reason"], str) or not row["reason"].strip():
        raise ValueError("V4 qualification reason must be non-empty")

    parsed = dict(row)
    parsed["witness_text"] = witness_text_v4(row["witness"])
    return parsed


def parse_qualification_batch_response_v4(
    text: str,
    *,
    controls: list[dict],
) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid V4 qualification JSON:{exc}") from exc
    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("V4 qualification batch must contain only reviews")
    rows = value["reviews"]
    if not isinstance(rows, list) or len(rows) != len(controls):
        raise ValueError("V4 qualification review count mismatch")

    by_id = {control["case_id"]: control for control in controls}
    parsed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("V4 qualification review must be object")
        case_id = row.get("case_id")
        if case_id not in by_id or case_id in parsed:
            raise ValueError("V4 qualification case_id mismatch or duplicate")
        parsed[case_id] = _parse_v4_row(
            row,
            control=by_id[case_id],
        )

    if set(parsed) != set(by_id):
        raise ValueError("V4 qualification case set mismatch")
    return [parsed[control["case_id"]] for control in controls]


def _is_contradiction_v4(result: dict) -> bool:
    observed = result.get("observed_defect")
    witness = result.get("witness")
    if observed is True:
        return not meaningful_witness_v4(witness)
    if observed is False:
        return witness != "NONE"
    return True


def qualify_reviewer_v4(call_control: Callable[[dict], dict]) -> dict:
    controls = build_reviewer_qualification_controls_v4()
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
        parsed = _parse_v4_row(result, control=control)
        observed = parsed["observed_defect"]
        expected = control["expected_defect"]
        defect_class = control["defect_class"]

        if _is_contradiction_v4(parsed):
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
            "witness_text": parsed["witness_text"],
            "witness_meaningful": meaningful_witness_v4(parsed["witness"]),
            "reason": parsed["reason"],
        })

    per_class = {
        defect_class: [
            per_class_hits[defect_class],
            per_class_total[defect_class],
        ]
        for defect_class in DEFECT_CLASSES_V4
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
        "schema": "RETENTION_AUDIT_ARCH_V4_REVIEWER_QUALIFICATION_V1",
        "status": "REVIEWER_QUALIFIED" if qualified else "REVIEWER_UNQUALIFIED",
        "sensitivity": [true_positive, defect_total],
        "specificity": [true_negative, good_total],
        "per_class": per_class,
        "structured_contradictions": contradictions,
        "controls": rows,
    }


def run_batched_reviewer_qualification_v4(
    *,
    call_text: Callable[[str], str] = ollama_text,
    batch_size: int = 4,
    max_attempts: int = 3,
) -> dict:
    if batch_size < 1:
        raise ValueError("V4 batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("V4 max_attempts must be positive")

    controls = build_reviewer_qualification_controls_v4()
    parsed_by_id: dict[str, dict] = {}
    batches: list[dict] = []

    for start in range(0, len(controls), batch_size):
        batch = controls[start:start + batch_size]
        prompt = qualification_batch_prompt_v4(batch)
        failures: list[dict] = []
        parsed: list[dict] | None = None
        attempts = 0

        for attempts in range(1, max_attempts + 1):
            response_text = ""
            try:
                response_text = call_text(prompt)
                if (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                ):
                    raise ValueError("empty V4 qualification response")
                parsed = parse_qualification_batch_response_v4(
                    response_text,
                    controls=batch,
                )
                break
            except (ValueError, json.JSONDecodeError) as exc:
                raw = (
                    response_text.encode("utf-8")
                    if isinstance(response_text, str)
                    else b""
                )
                failures.append({
                    "attempt": attempts,
                    "error": f"{type(exc).__name__}:{exc}",
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                })

        if parsed is None:
            return {
                "schema": "RETENTION_AUDIT_ARCH_V4_REVIEWER_QUALIFICATION_V1",
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

    result = qualify_reviewer_v4(
        lambda control: parsed_by_id[control["case_id"]]
    )
    result["batch_size"] = batch_size
    result["batch_count"] = len(batches)
    result["batches"] = batches
    return result
