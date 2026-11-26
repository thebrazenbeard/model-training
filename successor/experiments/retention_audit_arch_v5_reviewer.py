from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Callable
from urllib.request import Request, urlopen

from successor.experiments.retention_audit_arch_v2_reviewer import (
    MODEL,
    MODEL_BLOB_SHA256,
    OLLAMA_URL,
    SEED,
    TEMPERATURE,
    reviewer_identity as base_reviewer_identity,
)
from successor.experiments.retention_audit_arch_v4_reviewer import (
    meaningful_witness_v4,
    witness_text_v4,
)


DEFECT_CLASSES_V5 = (
    "referent_ambiguity",
    "prompt_grader_semantic_mismatch",
    "contradictory_instructions",
    "underspecified_output_format",
    "evidence_policy_ambiguity",
    "scope_negation_ambiguity",
)

_JSON_TYPES = ["string", "array", "object", "null", "number", "boolean"]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _fresh_control_text_v5(
    defect_class: str,
    *,
    defect: bool,
    variant: int,
) -> tuple[str, dict]:
    token = f"T{variant:02d}"

    if defect_class == "referent_ambiguity":
        if defect:
            return (
                f"Morgan told Riley after Casey arrived that they should file "
                f"{token}. Return only the name of the person who should file it.",
                {
                    "candidate_referents": ["Morgan", "Riley", "Casey"],
                    "required_answer_type": "person name",
                },
            )
        return (
            f"Morgan told Riley, 'Riley, file {token} now.' Return only the "
            "name of the person who should file it.",
            {
                "candidate_referents": ["Morgan", "Riley"],
                "explicit_addressee": "Riley",
            },
        )

    if defect_class == "prompt_grader_semantic_mismatch":
        if defect:
            return (
                f"Return the stored title for record {token}.",
                {
                    "grader_checks": "stored quantity",
                    "record": token,
                },
            )
        return (
            f"Return only the stored quantity for record {token}.",
            {
                "grader_checks": "stored quantity",
                "record": token,
            },
        )

    if defect_class == "contradictory_instructions":
        if defect:
            return (
                f"Return exactly four copies of {token} and exactly two copies "
                f"of {token}.",
                {
                    "constraint_a": "exactly four copies",
                    "constraint_b": "exactly two copies",
                },
            )
        return (
            f"Return exactly four copies of {token}, separated by single "
            "spaces, with no punctuation.",
            {
                "constraint_a": "exactly four copies",
                "separator": "single space",
                "punctuation": "none",
            },
        )

    if defect_class == "underspecified_output_format":
        if defect:
            return (
                f"Provide the full record for {token}.",
                {
                    "available_fields": ["key", "state", "amount"],
                    "acceptable_shapes": [
                        "plain text",
                        "JSON",
                        "key-value pairs",
                    ],
                },
            )
        return (
            f"Return exactly one compact JSON object with keys "
            f"'key','state','amount' for {token}, in that key order.",
            {
                "required_shape": {
                    "key": token,
                    "state": f"phase-{variant}",
                    "amount": variant + 9,
                },
                "key_order": ["key", "state", "amount"],
            },
        )

    if defect_class == "evidence_policy_ambiguity":
        if defect:
            return (
                f"Use the supplied note for {token}; if it omits a detail, "
                "supplement it with ordinary background knowledge when useful.",
                {
                    "note": f"{token} has a stored state.",
                    "outside_knowledge_rule": "priority unclear",
                },
            )
        return (
            f"Use only the supplied note for {token}. If the requested fact is "
            "absent, answer exactly NOT_ENOUGH_INFO.",
            {
                "note": f"{token} has a stored state.",
                "outside_knowledge_rule": "forbidden",
            },
        )

    if defect_class == "scope_negation_ambiguity":
        if defect:
            return (
                f"For {token}, discard items that are not unapproved.",
                {
                    "candidate_states": ["approved", "unapproved", "pending"],
                    "scope_phrase": "not unapproved",
                },
            )
        return (
            f"For {token}, keep only unapproved items; discard approved and "
            "pending items.",
            {
                "candidate_states": ["approved", "unapproved", "pending"],
                "included_state": "unapproved",
            },
        )

    raise ValueError(f"unknown V5 defect class:{defect_class}")


def build_reviewer_qualification_controls_v5() -> list[dict]:
    controls: list[dict] = []
    for defect_class in DEFECT_CLASSES_V5:
        for variant in range(4):
            for expected_defect in (True, False):
                prompt, context = _fresh_control_text_v5(
                    defect_class,
                    defect=expected_defect,
                    variant=variant,
                )
                polarity = "defect" if expected_defect else "good"
                controls.append({
                    "case_id": (
                        f"arch-v5-reviewer-{defect_class}-{polarity}-{variant}"
                    ),
                    "defect_class": defect_class,
                    "expected_defect": expected_defect,
                    "prompt": prompt,
                    "context": context,
                })
    return controls


def qualification_batch_prompt_v5(controls: list[dict]) -> str:
    if not controls:
        raise ValueError("V5 qualification controls must be non-empty")
    visible = []
    seen: set[str] = set()
    for control in controls:
        case_id = control.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("V5 qualification case_id invalid")
        if case_id in seen:
            raise ValueError("duplicate V5 qualification case_id")
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
        "must contain a concrete semantic example; it may be a nonempty JSON "
        "string, array, or object. Do not infer or discuss hidden expected "
        "labels.\n"
        + canonical_json({"controls": visible})
    )


def _witness_schema() -> dict:
    return {"type": list(_JSON_TYPES)}


def qualification_response_schema_v5(controls: list[dict]) -> dict:
    if not controls:
        raise ValueError("V5 qualification controls must be non-empty")
    case_ids = [control["case_id"] for control in controls]
    defect_classes = sorted({control["defect_class"] for control in controls})
    item = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "enum": case_ids},
            "observed_defect": {"type": "boolean"},
            "defect_class": {
                "type": "string",
                "enum": defect_classes,
            },
            "witness": _witness_schema(),
            "reason": {"type": "string"},
        },
        "required": [
            "case_id",
            "observed_defect",
            "defect_class",
            "witness",
            "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": item,
                "minItems": len(controls),
                "maxItems": len(controls),
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def semantic_response_schema_v5(case_ids: list[str]) -> dict:
    if not case_ids:
        raise ValueError("V5 semantic case_ids must be non-empty")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate V5 semantic case_id")

    item = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "enum": list(case_ids)},
            "observation": {
                "type": "string",
                "enum": [
                    "DERIVED_ANSWER",
                    "AMBIGUITY_WITNESS",
                    "CONTRACT_COUNTEREXAMPLE",
                    "NO_SEMANTIC_DEFECT_FOUND",
                    "CANNOT_DETERMINE",
                ],
            },
            "derived_answer": {"type": list(_JSON_TYPES)},
            "witness": _witness_schema(),
            "reason": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": [
            "case_id",
            "observation",
            "derived_answer",
            "witness",
            "reason",
            "confidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": item,
                "minItems": len(case_ids),
                "maxItems": len(case_ids),
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def structured_request_body_v5(prompt: str, schema: dict) -> dict:
    return {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
    }


def reviewer_identity_v5() -> dict:
    identity = dict(base_reviewer_identity())
    identity["format"] = "DYNAMIC_JSON_SCHEMA"
    identity["structured_output"] = "JSON_SCHEMA"
    return identity


def ollama_structured_text_v5(prompt: str, schema: dict) -> str:
    request = Request(
        OLLAMA_URL,
        data=json.dumps(
            structured_request_body_v5(prompt, schema)
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=900) as response:
        outer = json.loads(response.read().decode("utf-8"))
    text = outer.get("response")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty V5 structured reviewer response")
    return text


def _parse_qualification_row_v5(row: dict, *, control: dict) -> dict:
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
        raise ValueError("V5 qualification row schema mismatch")
    if row["case_id"] != control.get("case_id"):
        raise ValueError("V5 qualification case_id mismatch")
    if row["defect_class"] != control.get("defect_class"):
        raise ValueError("V5 qualification defect_class mismatch")
    if not isinstance(row["observed_defect"], bool):
        raise ValueError("V5 observed_defect must be boolean")
    if not isinstance(row["reason"], str) or not row["reason"].strip():
        raise ValueError("V5 qualification reason must be non-empty")
    parsed = dict(row)
    parsed["witness_text"] = witness_text_v4(row["witness"])
    return parsed


def parse_qualification_batch_response_v5(
    text: str,
    *,
    controls: list[dict],
) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid V5 qualification JSON:{exc}") from exc
    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("V5 qualification batch must contain only reviews")
    rows = value["reviews"]
    if not isinstance(rows, list) or len(rows) != len(controls):
        raise ValueError("V5 qualification review count mismatch")

    by_id = {control["case_id"]: control for control in controls}
    parsed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("V5 qualification review must be object")
        case_id = row.get("case_id")
        if case_id not in by_id or case_id in parsed:
            raise ValueError("V5 qualification case_id mismatch or duplicate")
        parsed[case_id] = _parse_qualification_row_v5(
            row,
            control=by_id[case_id],
        )

    if set(parsed) != set(by_id):
        raise ValueError("V5 qualification case set mismatch")
    return [parsed[control["case_id"]] for control in controls]


def _is_contradiction_v5(result: dict) -> bool:
    observed = result.get("observed_defect")
    witness = result.get("witness")
    if observed is True:
        return not meaningful_witness_v4(witness)
    if observed is False:
        return witness != "NONE"
    return True


def qualify_reviewer_v5(call_control: Callable[[dict], dict]) -> dict:
    controls = build_reviewer_qualification_controls_v5()
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
        parsed = _parse_qualification_row_v5(result, control=control)
        observed = parsed["observed_defect"]
        expected = control["expected_defect"]
        defect_class = control["defect_class"]

        if _is_contradiction_v5(parsed):
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
        for defect_class in DEFECT_CLASSES_V5
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
        "schema": "RETENTION_AUDIT_ARCH_V5_REVIEWER_QUALIFICATION_V1",
        "status": "REVIEWER_QUALIFIED" if qualified else "REVIEWER_UNQUALIFIED",
        "sensitivity": [true_positive, defect_total],
        "specificity": [true_negative, good_total],
        "per_class": per_class,
        "structured_contradictions": contradictions,
        "controls": rows,
    }


def run_batched_reviewer_qualification_v5(
    *,
    call_structured: Callable[[str, dict], str] = ollama_structured_text_v5,
    batch_size: int = 4,
    max_attempts: int = 3,
) -> dict:
    if batch_size < 1:
        raise ValueError("V5 batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("V5 max_attempts must be positive")

    controls = build_reviewer_qualification_controls_v5()
    parsed_by_id: dict[str, dict] = {}
    batches: list[dict] = []

    for start in range(0, len(controls), batch_size):
        batch = controls[start:start + batch_size]
        prompt = qualification_batch_prompt_v5(batch)
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
                    raise ValueError("empty V5 qualification response")
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
                    "raw_response": (
                        response_text
                        if isinstance(response_text, str)
                        else ""
                    ),
                })

        if parsed is None:
            return {
                "schema": "RETENTION_AUDIT_ARCH_V5_REVIEWER_QUALIFICATION_V1",
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

    result = qualify_reviewer_v5(
        lambda control: parsed_by_id[control["case_id"]]
    )
    result["batch_size"] = batch_size
    result["batch_count"] = len(batches)
    result["batches"] = batches
    return result
