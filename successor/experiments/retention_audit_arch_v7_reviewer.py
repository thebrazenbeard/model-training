from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Callable

from successor.experiments.retention_audit_arch_v4_reviewer import meaningful_witness_v4
from successor.experiments.retention_audit_arch_v5_reviewer import (
    ollama_structured_text_v5,
    parse_qualification_batch_response_v5,
    qualification_response_schema_v5,
    reviewer_identity_v5,
    sha256_json,
)

DEFECT_CLASSES_V7 = (
    "referent_ambiguity",
    "prompt_grader_semantic_mismatch",
    "contradictory_instructions",
    "underspecified_output_format",
    "evidence_policy_ambiguity",
    "scope_negation_ambiguity",
)


def _fresh_control_text_v7(
    defect_class: str,
    *,
    defect: bool,
    variant: int,
) -> tuple[str, dict]:
    token = f"W{variant:02d}"

    if defect_class == "referent_ambiguity":
        if defect:
            return (
                f"Jordan told Casey after Morgan arrived that they should archive "
                f"{token}. Return only the name of the person who should archive it.",
                {
                    "candidate_referents": ["Jordan", "Casey", "Morgan"],
                    "required_answer_type": "person name",
                },
            )
        return (
            f"Jordan told Casey, 'Casey, archive {token} now.' Return only the "
            "name of the person who should archive it.",
            {
                "candidate_referents": ["Jordan", "Casey"],
                "explicit_addressee": "Casey",
            },
        )

    if defect_class == "prompt_grader_semantic_mismatch":
        if defect:
            return (
                f"Return the stored label for entry {token}.",
                {
                    "grader_checks": "stored count",
                    "entry": token,
                },
            )
        return (
            f"Return only the stored count for entry {token}.",
            {
                "grader_checks": "stored count",
                "entry": token,
            },
        )

    if defect_class == "contradictory_instructions":
        if defect:
            return (
                f"Return exactly three copies of {token} and exactly five copies "
                f"of {token}.",
                {
                    "constraint_a": "exactly three copies",
                    "constraint_b": "exactly five copies",
                },
            )
        return (
            f"Return exactly three copies of {token}, separated by one space, "
            "with no punctuation.",
            {
                "constraint_a": "exactly three copies",
                "separator": "one space",
                "punctuation": "none",
            },
        )

    if defect_class == "underspecified_output_format":
        if defect:
            return (
                f"Provide the complete entry for {token}.",
                {
                    "available_fields": ["id", "state", "count"],
                    "acceptable_shapes": ["plain text", "JSON", "table row"],
                },
            )
        return (
            f"Return exactly one compact JSON object with keys "
            f"'id','state','count' for {token}, in that key order.",
            {
                "required_shape": {
                    "id": token,
                    "state": f"stage-{variant}",
                    "count": variant + 11,
                },
                "key_order": ["id", "state", "count"],
            },
        )

    if defect_class == "evidence_policy_ambiguity":
        if defect:
            return (
                f"Use the supplied record for {token}; when it omits a fact, "
                "use common background knowledge if helpful.",
                {
                    "record": f"{token} has a stored category.",
                    "outside_knowledge_rule": "priority unclear",
                },
            )
        return (
            f"Use only the supplied record for {token}. If the requested fact "
            "is absent, answer exactly NOT_ENOUGH_INFO.",
            {
                "record": f"{token} has a stored category.",
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
            f"For {token}, keep only inactive entries; remove active and pending "
            "entries.",
            {
                "candidate_states": ["active", "inactive", "pending"],
                "included_state": "inactive",
            },
        )

    raise ValueError(f"unknown V7 defect class:{defect_class}")


def build_reviewer_qualification_controls_v7() -> list[dict]:
    controls: list[dict] = []
    for defect_class in DEFECT_CLASSES_V7:
        for variant in range(4):
            for expected_defect in (True, False):
                prompt, context = _fresh_control_text_v7(
                    defect_class,
                    defect=expected_defect,
                    variant=variant,
                )
                polarity = "defect" if expected_defect else "good"
                controls.append({
                    "case_id": (
                        f"arch-v7-reviewer-{defect_class}-{polarity}-{variant}"
                    ),
                    "defect_class": defect_class,
                    "expected_defect": expected_defect,
                    "prompt": prompt,
                    "context": context,
                })
    return controls


def qualification_batch_prompt_v7(controls: list[dict]) -> str:
    if not controls:
        raise ValueError("V7 qualification controls must be non-empty")
    visible = []
    seen: set[str] = set()
    for control in controls:
        case_id = control.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("V7 qualification case_id invalid")
        if case_id in seen:
            raise ValueError("duplicate V7 qualification case_id")
        seen.add(case_id)
        visible.append({
            "case_id": case_id,
            "defect_class": control.get("defect_class"),
            "prompt": control.get("prompt"),
            "context": control.get("context"),
        })
    return (
        "Evaluate each control only for its named semantic defect class. "
        "Populate the constrained response schema. observed_defect is your "
        "independent semantic judgment. When observed_defect is false, witness "
        "must be the exact string NONE. When observed_defect is true, witness "
        "must contain a concrete semantic example. Do not infer or discuss "
        "hidden expected labels.\n"
        + json.dumps(
            {"controls": visible},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def reviewer_identity_v7() -> dict:
    return dict(reviewer_identity_v5())


def _score_v7(parsed_by_id: dict[str, dict]) -> dict:
    controls = build_reviewer_qualification_controls_v7()
    true_positive = 0
    true_negative = 0
    defect_total = 0
    good_total = 0
    per_class_hits: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    contradictions: list[str] = []
    rows: list[dict] = []

    for control in controls:
        row = parsed_by_id[control["case_id"]]
        observed = row["observed_defect"]
        expected = control["expected_defect"]
        witness = row["witness"]
        defect_class = control["defect_class"]

        contradiction = (
            (observed is True and not meaningful_witness_v4(witness))
            or (observed is False and witness != "NONE")
        )
        if contradiction:
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
            "witness": witness,
            "witness_text": row.get("witness_text"),
            "witness_meaningful": meaningful_witness_v4(witness),
            "reason": row["reason"],
        })

    per_class = {
        defect_class: [
            per_class_hits[defect_class],
            per_class_total[defect_class],
        ]
        for defect_class in DEFECT_CLASSES_V7
    }
    qualified = (
        true_positive >= 22
        and defect_total == 24
        and true_negative >= 22
        and good_total == 24
        and all(
            value[0] >= 3 and value[1] == 4
            for value in per_class.values()
        )
        and not contradictions
    )
    return {
        "schema": "RETENTION_AUDIT_ARCH_V7_REVIEWER_QUALIFICATION_V1",
        "status": "REVIEWER_QUALIFIED" if qualified else "REVIEWER_UNQUALIFIED",
        "sensitivity": [true_positive, defect_total],
        "specificity": [true_negative, good_total],
        "per_class": per_class,
        "structured_contradictions": contradictions,
        "controls": rows,
    }


def run_batched_reviewer_qualification_v7(
    *,
    call_structured: Callable[[str, dict], str] = ollama_structured_text_v5,
    batch_size: int = 4,
    max_attempts: int = 3,
) -> dict:
    if batch_size < 1:
        raise ValueError("V7 batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("V7 max_attempts must be positive")

    controls = build_reviewer_qualification_controls_v7()
    parsed_by_id: dict[str, dict] = {}
    batches: list[dict] = []

    for start in range(0, len(controls), batch_size):
        batch = controls[start:start + batch_size]
        prompt = qualification_batch_prompt_v7(batch)
        schema = qualification_response_schema_v5(batch)
        schema_sha256 = sha256_json(schema)
        failures: list[dict] = []
        parsed: list[dict] | None = None
        attempts = 0

        for attempts in range(1, max_attempts + 1):
            response_text = ""
            try:
                response_text = call_structured(prompt, schema)
                if (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                ):
                    raise ValueError("empty V7 qualification response")
                parsed = parse_qualification_batch_response_v5(
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
                "schema": "RETENTION_AUDIT_ARCH_V7_REVIEWER_QUALIFICATION_V1",
                "status": "REVIEWER_UNQUALIFIED",
                "reason": "qualification_transport_failure",
                "batch_size": batch_size,
                "batch_count": len(batches) + 1,
                "response_schema_sha256": schema_sha256,
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
            "response_schema_sha256": schema_sha256,
            "prior_attempt_failures": failures,
        })

    result = _score_v7(parsed_by_id)
    result["batch_size"] = batch_size
    result["batch_count"] = len(batches)
    result["batches"] = batches
    return result
