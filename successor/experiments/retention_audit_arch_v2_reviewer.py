from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable
from urllib.request import Request, urlopen


MODEL = "ministral-3:14b"
MODEL_BLOB_SHA256 = (
    "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
SEED = 20261002
TEMPERATURE = 0


def ollama_request_body(prompt: str) -> dict:
    return {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
    }


def ollama_version() -> str:
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _model_blob_path() -> Path:
    shown = subprocess.run(
        ["ollama", "show", MODEL, "--modelfile"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in shown.stdout.splitlines():
        if line.startswith("FROM "):
            return Path(line[5:].strip().strip('"'))
    raise RuntimeError("Ollama Modelfile has no FROM blob")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reviewer_identity() -> dict:
    actual = _sha256_file(_model_blob_path())
    if actual != MODEL_BLOB_SHA256:
        raise RuntimeError(f"reviewer model blob mismatch:{actual}")
    return {
        "provider": "OLLAMA_LOCAL",
        "model": MODEL,
        "model_blob_sha256": actual,
        "runtime": ollama_version(),
        "temperature": TEMPERATURE,
        "seed": SEED,
        "url": OLLAMA_URL,
        "format": "json",
        "stream": False,
    }


def ollama_text(prompt: str) -> str:
    body = ollama_request_body(prompt)
    request = Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=900) as response:
        outer = json.loads(response.read().decode("utf-8"))
    text = outer.get("response")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty Ollama reviewer response")
    return text


ALLOWED_OBSERVATIONS = {
    "DERIVED_ANSWER",
    "AMBIGUITY_WITNESS",
    "CONTRACT_COUNTEREXAMPLE",
    "NO_SEMANTIC_DEFECT_FOUND",
    "CANNOT_DETERMINE",
}

DEFECT_CLASSES = (
    "referent_ambiguity",
    "prompt_grader_semantic_mismatch",
    "contradictory_instructions",
    "underspecified_output_format",
    "evidence_policy_ambiguity",
    "scope_negation_ambiguity",
)

REVIEW_KEYS = {
    "case_id",
    "observation",
    "derived_answer",
    "witness",
    "reason",
    "confidence",
}


def _control_text(
    defect_class: str,
    *,
    defect: bool,
    variant: int,
) -> tuple[str, dict]:
    token = f"Q{variant:02d}"
    if defect_class == "referent_ambiguity":
        if defect:
            return (
                f"Alex told Jordan that they should record {token}. "
                "Return the name of the person who should record it.",
                {"note": "Both Alex and Jordan are singular candidate referents."},
            )
        return (
            f"Alex told Jordan, 'Jordan, record {token}.' "
            "Return the name of the person who should record it.",
            {"note": "The quoted imperative explicitly addresses Jordan."},
        )

    if defect_class == "prompt_grader_semantic_mismatch":
        if defect:
            return (
                f"Return the country name for code {token}.",
                {"grader_checks": "two-character code rather than country name"},
            )
        return (
            f"Return only the two-character code {token}.",
            {"grader_checks": "two-character code"},
        )

    if defect_class == "contradictory_instructions":
        if defect:
            return (
                f"Return exactly {token} and do not include the character "
                f"{token[0]} anywhere.",
                {"constraint": "exact token and forbidden leading character"},
            )
        return (
            f"Return exactly {token} and do not include '?' anywhere.",
            {"constraint": "compatible exact-token and forbidden-character rules"},
        )

    if defect_class == "underspecified_output_format":
        if defect:
            return (
                f"Provide the record for {token}.",
                {"acceptable_shapes": ["plain text", "JSON", "key-value pairs"]},
            )
        return (
            f"Return exactly one compact JSON object with key 'id' and value "
            f"'{token}'.",
            {"required_shape": {"id": token}},
        )

    if defect_class == "evidence_policy_ambiguity":
        if defect:
            return (
                f"Evidence: {{\"id\":\"{token}\"}}. Decide whether the "
                "record is active.",
                {"missing_policy": "No rule states whether outside knowledge is allowed."},
            )
        return (
            f"Use only this evidence: {{\"id\":\"{token}\"}}. "
            "Do not use outside knowledge. Decide TRUE, FALSE, or "
            "NOT_ENOUGH_INFO for the claim that the record is active.",
            {"policy": "closed evidence; missing active field implies NOT_ENOUGH_INFO"},
        )

    if defect_class == "scope_negation_ambiguity":
        if defect:
            return (
                f"Do not report all values except {token}.",
                {"note": "Negation can scope over report-all or the exception."},
            )
        return (
            f"Report only {token}; report no other value.",
            {"note": "Scope and exclusion are explicit."},
        )

    raise ValueError(f"unknown defect class: {defect_class}")


def build_reviewer_qualification_controls() -> list[dict]:
    controls: list[dict] = []
    for defect_class in DEFECT_CLASSES:
        for variant in range(4):
            for expected_defect in (True, False):
                prompt, context = _control_text(
                    defect_class,
                    defect=expected_defect,
                    variant=variant,
                )
                polarity = "defect" if expected_defect else "good"
                controls.append({
                    "case_id": (
                        f"arch-v2-reviewer-{defect_class}-{polarity}-{variant}"
                    ),
                    "defect_class": defect_class,
                    "expected_defect": expected_defect,
                    "prompt": prompt,
                    "context": context,
                })
    return controls


def _blind_case(case: dict) -> dict:
    grader = case.get("grader_contract", {})
    visible_grader = {
        key: value
        for key, value in grader.items()
        if key not in {"answer_key", "answer_key_digest", "verdict"}
    }
    return {
        "case_id": case["case_id"],
        "category": case.get("category"),
        "family_id": case.get("family_id"),
        "prompt": case["prompt"],
        "source_evidence": case.get("source_evidence"),
        "grader_specification": visible_grader,
    }


def review_prompt(cases: list[dict]) -> str:
    if not cases:
        raise ValueError("semantic review requires at least one case")
    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("semantic review case_id invalid")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("semantic review case IDs must be unique")

    subjects = [_blind_case(case) for case in cases]
    return (
        "You are a semantic red-team reviewer for benchmark prompts. "
        "Do not grade a model response and do not make an admission decision. "
        "For each supplied case, independently derive what the prompt requests "
        "from the visible source/specification, then search for a concrete "
        "ambiguity or contract counterexample. A concern must be expressed as "
        "a specific witness; unsupported suspicion is not a defect. "
        "Return JSON only as an object with exactly one key, reviews. "
        "reviews must contain exactly one object per case, preserving case IDs. "
        "Each review object must contain exactly: case_id, observation, "
        "derived_answer, witness, reason, confidence. observation must be one "
        "of DERIVED_ANSWER, AMBIGUITY_WITNESS, CONTRACT_COUNTEREXAMPLE, "
        "NO_SEMANTIC_DEFECT_FOUND, CANNOT_DETERMINE. confidence must be "
        "high, medium, or low. For AMBIGUITY_WITNESS or "
        "CONTRACT_COUNTEREXAMPLE, witness must be a non-empty concrete string. "
        "For NO_SEMANTIC_DEFECT_FOUND, witness must be null. "
        "SUBJECTS="
        + json.dumps(
            subjects,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def parse_witness_response(
    text: str,
    *,
    expected_case_ids: list[str],
) -> list[dict]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("witness response must use exact object schema")
    rows = value["reviews"]
    if not isinstance(rows, list):
        raise ValueError("reviews must be an array")
    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise ValueError("expected case IDs must be unique")

    parsed_by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != REVIEW_KEYS:
            raise ValueError("review row must use closed witness schema")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("invalid case_id")
        if case_id in parsed_by_id:
            raise ValueError(f"duplicate case_id: {case_id}")

        observation = row["observation"]
        if observation not in ALLOWED_OBSERVATIONS:
            raise ValueError(f"invalid observation: {observation}")
        confidence = row["confidence"]
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("invalid confidence")
        reason = row["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        witness = row["witness"]
        if observation in {"AMBIGUITY_WITNESS", "CONTRACT_COUNTEREXAMPLE"}:
            if not isinstance(witness, str) or not witness.strip():
                raise ValueError("semantic-defect observation requires witness")
        if observation == "NO_SEMANTIC_DEFECT_FOUND" and witness is not None:
            raise ValueError("no-defect observation requires null witness")

        parsed_by_id[case_id] = dict(row)

    if set(parsed_by_id) != set(expected_case_ids):
        raise ValueError(
            "review case set mismatch: "
            f"got={sorted(parsed_by_id)} expected={sorted(expected_case_ids)}"
        )
    return [parsed_by_id[case_id] for case_id in expected_case_ids]



def qualification_batch_prompt(controls: list[dict]) -> str:
    if not controls:
        raise ValueError("qualification controls must be non-empty")
    visible = []
    case_ids: set[str] = set()
    for control in controls:
        case_id = control.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("qualification control case_id invalid")
        if case_id in case_ids:
            raise ValueError("duplicate qualification case_id")
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
        "Evaluate each hidden qualification control for its named semantic "
        "defect class. Return strict JSON with exactly one key, reviews. "
        "reviews must contain one object per control with exactly case_id, "
        "observed_defect, defect_class, witness, reason. observed_defect is a "
        "JSON boolean. A claimed defect requires a concrete witness; otherwise "
        "witness must be null. Do not infer or discuss any hidden expected "
        "label.\n"
        f"{payload}"
    )


def parse_qualification_batch_response(
    text: str,
    *,
    controls: list[dict],
) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid qualification batch JSON:{exc}") from exc
    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("qualification batch must contain only reviews")
    reviews = value["reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(controls):
        raise ValueError("qualification batch review count mismatch")
    by_id = {control["case_id"]: control for control in controls}
    parsed: dict[str, dict] = {}
    for row in reviews:
        if not isinstance(row, dict):
            raise ValueError("qualification batch review must be object")
        case_id = row.get("case_id")
        if case_id not in by_id or case_id in parsed:
            raise ValueError("qualification batch case_id mismatch or duplicate")
        parsed[case_id] = parse_qualification_response(
            json.dumps(row),
            control=by_id[case_id],
        )
    if set(parsed) != set(by_id):
        raise ValueError("qualification batch case set mismatch")
    return [parsed[control["case_id"]] for control in controls]


def qualification_prompt(control: dict) -> str:
    required = {"case_id", "defect_class", "prompt"}
    if not required <= set(control):
        raise ValueError("qualification control is missing required fields")
    visible = {
        "case_id": control["case_id"],
        "defect_class": control["defect_class"],
        "prompt": control["prompt"],
    }
    payload = json.dumps(
        visible,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        "Evaluate this hidden qualification control for the named semantic "
        "defect class. Return strict JSON with exactly case_id, "
        "observed_defect, defect_class, witness, reason. observed_defect must "
        "be a JSON boolean. If a defect is observed, witness must identify the "
        "concrete ambiguity, contradiction, mismatch, or underspecification; "
        "otherwise witness must be null. Do not infer or discuss any hidden "
        "expected label.\n"
        f"{payload}"
    )


def parse_qualification_response(text: str, *, control: dict) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid qualification JSON:{exc}") from exc
    required = {
        "case_id",
        "observed_defect",
        "defect_class",
        "witness",
        "reason",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("qualification response schema mismatch")
    if value["case_id"] != control.get("case_id"):
        raise ValueError("qualification case_id mismatch")
    if value["defect_class"] != control.get("defect_class"):
        raise ValueError("qualification defect_class mismatch")
    if not isinstance(value["observed_defect"], bool):
        raise ValueError("observed_defect must be boolean")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("qualification reason must be non-empty")
    witness = value["witness"]
    if value["observed_defect"]:
        if not isinstance(witness, str) or not witness.strip():
            raise ValueError("qualification witness is required")
    elif witness not in (None, ""):
        raise ValueError("qualification witness must be null when no defect")
    return value



def run_batched_reviewer_qualification(
    *,
    call_text: Callable[[str], str] = ollama_text,
    batch_size: int = 8,
    max_attempts: int = 3,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    controls = build_reviewer_qualification_controls()
    parsed_by_id: dict[str, dict] = {}
    batches: list[dict] = []

    for start in range(0, len(controls), batch_size):
        batch = controls[start:start + batch_size]
        prompt = qualification_batch_prompt(batch)
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
                    raise ValueError("empty qualification reviewer response")
                parsed = parse_qualification_batch_response(
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
                "schema": "RETENTION_AUDIT_ARCH_V2_REVIEWER_QUALIFICATION_V1",
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

    result = qualify_reviewer(
        lambda control: parsed_by_id[control["case_id"]]
    )
    result["batch_size"] = batch_size
    result["batch_count"] = len(batches)
    result["batches"] = batches
    return result


def qualify_reviewer(
    call_control: Callable[[dict], dict],
) -> dict:
    controls = build_reviewer_qualification_controls()
    per_class_hits: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    true_positive = 0
    defect_total = 0
    true_negative = 0
    good_total = 0
    contradictions: list[str] = []
    rows: list[dict] = []

    for control in controls:
        result = call_control(control)
        if not isinstance(result, dict):
            raise ValueError("qualification reviewer must return an object")
        required = {
            "case_id",
            "observed_defect",
            "defect_class",
            "witness",
            "reason",
        }
        if set(result) != required:
            raise ValueError("qualification result uses invalid schema")
        if result["case_id"] != control["case_id"]:
            raise ValueError("qualification case_id mismatch")
        if result["defect_class"] != control["defect_class"]:
            raise ValueError("qualification defect_class mismatch")
        if not isinstance(result["observed_defect"], bool):
            raise ValueError("observed_defect must be boolean")
        if not isinstance(result["reason"], str) or not result["reason"].strip():
            raise ValueError("qualification reason must be non-empty")

        observed = result["observed_defect"]
        witness = result["witness"]
        if observed and (not isinstance(witness, str) or not witness.strip()):
            contradictions.append(control["case_id"])
        if not observed and witness is not None:
            contradictions.append(control["case_id"])

        expected = control["expected_defect"]
        defect_class = control["defect_class"]
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
            "reason": result["reason"],
        })

    sensitivity_ok = true_positive >= 22 and defect_total == 24
    specificity_ok = true_negative >= 22 and good_total == 24
    class_ok = all(
        per_class_hits[name] >= 3 and per_class_total[name] == 4
        for name in DEFECT_CLASSES
    )
    contradiction_ok = not contradictions

    return {
        "status": (
            "REVIEWER_QUALIFIED"
            if sensitivity_ok
            and specificity_ok
            and class_ok
            and contradiction_ok
            else "REVIEWER_UNQUALIFIED"
        ),
        "sensitivity": [true_positive, defect_total],
        "specificity": [true_negative, good_total],
        "per_class": {
            name: [per_class_hits[name], per_class_total[name]]
            for name in DEFECT_CLASSES
        },
        "schema_contradictions": contradictions,
        "rows": rows,
    }
