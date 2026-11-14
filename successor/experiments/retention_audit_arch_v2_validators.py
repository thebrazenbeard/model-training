from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any, Callable
from urllib.request import urlopen


LOCATION_MANIFEST_URL = "https://geo.mindstellar.com/releases/2026-09-20T1335Z/manifest.json"
LOCATION_MANIFEST_SHA256 = "bf5a2dc0bb0f1ba15f7149e9fa1184aa632f280a72fe306c154a9e05cfacab29"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class SourceBindingError(ValueError):
    pass


def _fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read()


def load_pinned_location_manifest(
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict:
    fetch = fetch_bytes or _fetch_bytes
    raw = fetch(LOCATION_MANIFEST_URL)
    if not isinstance(raw, bytes):
        raise SourceBindingError("location manifest fetch did not return bytes")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != LOCATION_MANIFEST_SHA256:
        raise SourceBindingError(
            "location manifest hash mismatch:"
            f"{actual}!={LOCATION_MANIFEST_SHA256}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceBindingError(
            f"location manifest parse failure:{exc}"
        ) from exc
    countries = value.get("countries") if isinstance(value, dict) else None
    if not isinstance(countries, list) or len(countries) < 200:
        raise SourceBindingError("location manifest countries invalid")
    return value


def _require_source_hash(row: dict, evidence: dict) -> dict:
    actual = sha256_json(evidence)
    expected = row.get("source_hash")
    if actual != expected:
        raise SourceBindingError(
            f"{row.get('case_id')}: source hash mismatch:{actual}!={expected}"
        )
    return evidence


def resolve_source_evidence(
    row: dict,
    *,
    location_manifest: dict | None = None,
) -> dict:
    source_id = str(row.get("source_id", ""))
    grader = row.get("grader_contract", {})

    if ":country:" in source_id:
        if location_manifest is None:
            raise SourceBindingError("location manifest is required")
        code = source_id.rsplit(":country:", 1)[1]
        countries = {
            str(country["code"]): country
            for country in location_manifest.get("countries", [])
        }
        if code not in countries:
            raise SourceBindingError(f"location code not found:{code}")
        return _require_source_hash(row, countries[code])

    if ":pair:" in source_id:
        if location_manifest is None:
            raise SourceBindingError("location manifest is required")
        suffix = source_id.rsplit(":pair:", 1)[1]
        parts = suffix.split(":")
        if len(parts) != 2:
            raise SourceBindingError(f"invalid pair source id:{source_id}")
        countries = {
            str(country["code"]): country
            for country in location_manifest.get("countries", [])
        }
        left_code, right_code = parts
        if left_code not in countries or right_code not in countries:
            raise SourceBindingError(
                f"location pair not found:{left_code}:{right_code}"
            )
        return _require_source_hash(
            row,
            {
                "left": countries[left_code],
                "right": countries[right_code],
            },
        )

    if ":coding:" in source_id:
        suffix = source_id.rsplit(":coding:", 1)[1]
        family, variant_text = suffix.rsplit(":", 1)
        lines = str(row.get("prompt", "")).splitlines()
        if len(lines) != 3:
            raise SourceBindingError("generated code prompt shape mismatch")
        first_prefix = "Implement this Python function exactly: "
        second_prefix = "Contract: "
        if not lines[0].startswith(first_prefix):
            raise SourceBindingError("generated code signature line missing")
        if not lines[1].startswith(second_prefix):
            raise SourceBindingError("generated code contract line missing")
        if lines[2] != "Return only executable Python code.":
            raise SourceBindingError("generated code return instruction mismatch")
        evidence = {
            "family": family,
            "variant": int(variant_text),
            "function": grader.get("function"),
            "signature": lines[0].removeprefix(first_prefix),
            "task": lines[1].removeprefix(second_prefix),
            "tests": list(grader.get("tests", [])),
        }
        return _require_source_hash(row, evidence)

    if ":instruction:" in source_id:
        suffix = source_id.rsplit(":instruction:", 1)[1]
        family, variant_text = suffix.rsplit(":", 1)
        evidence = {
            "family": family,
            "variant": int(variant_text),
            "prompt": row.get("prompt"),
            "constraints": grader.get("constraints"),
        }
        return _require_source_hash(row, evidence)

    raise SourceBindingError(f"unrecognized retention source id:{source_id}")


def validator_class_for_row(row: dict) -> str:
    family_id = row.get("family_id")
    if isinstance(family_id, str):
        if family_id.startswith("generated-code:"):
            return "generated_code"
        if family_id.startswith("generated-instruction:"):
            return "generated_instruction"
        if family_id.startswith("evidence-calibration:"):
            return "evidence_calibration"
        if family_id.startswith("location-arithmetic:"):
            return "source_arithmetic"
        if family_id == "location:structured-extraction":
            return "structured_extraction"
        if family_id.startswith("location:"):
            return "direct_source_lookup"
    raise ValueError(f"unsupported Architecture V2 family: {family_id}")


def _direct_source_expected_prompt(row: dict, source_evidence: dict) -> str:
    relation = row["family_id"].split(":", 1)[1]
    revision = row["source_revision"]
    if relation == "code_from_name":
        return (
            f"In the frozen {revision} location-data release, what two-character "
            f"country code is assigned to {source_evidence['name']}? "
            "Return only the code."
        )
    if relation == "name_from_code":
        return (
            f"In the frozen {revision} location-data release, which country or "
            f"territory name is assigned to code {source_evidence['code']}? "
            "Return only the recorded name."
        )
    if relation == "region_count":
        return (
            f"In the frozen {revision} location-data release, how many "
            f"administrative regions are recorded for {source_evidence['name']}? "
            "Return only the integer."
        )
    if relation == "settlement_count":
        return (
            f"In the frozen {revision} location-data release, how many "
            f"settlements are recorded for {source_evidence['name']}? "
            "Return only the integer."
        )
    raise ValueError(f"unsupported direct lookup relation: {relation}")


def _direct_source_answer(row: dict, source_evidence: dict) -> str:
    relation = row["family_id"].split(":", 1)[1]
    if relation == "code_from_name":
        return str(source_evidence["code"])
    if relation == "name_from_code":
        return str(source_evidence["name"])
    if relation == "region_count":
        return str(source_evidence["regions"])
    if relation == "settlement_count":
        return str(source_evidence["settlements"])
    raise ValueError(f"unsupported direct lookup relation: {relation}")


def _source_arithmetic_expected_prompt(
    row: dict,
    source_evidence: dict,
) -> str:
    relation = row["family_id"].split(":", 1)[1]
    left = source_evidence["left"]
    right = source_evidence["right"]
    if relation == "settlement_sum":
        return (
            f"A frozen source record lists {left['name']} with "
            f"{left['settlements']} settlements and {right['name']} with "
            f"{right['settlements']} settlements. What is the sum? "
            "Return only the integer."
        )
    if relation == "settlement_difference":
        return (
            f"A frozen source record lists {left['name']} with "
            f"{left['settlements']} settlements and {right['name']} with "
            f"{right['settlements']} settlements. What is the absolute "
            "difference? Return only the integer."
        )
    if relation == "region_sum":
        return (
            f"A frozen source record lists {left['name']} with "
            f"{left['regions']} regions and {right['name']} with "
            f"{right['regions']} regions. What is the total number of regions? "
            "Return only the integer."
        )
    raise ValueError(f"unsupported source arithmetic relation: {relation}")


def _source_arithmetic_answer(row: dict, source_evidence: dict) -> str:
    relation = row["family_id"].split(":", 1)[1]
    left = source_evidence["left"]
    right = source_evidence["right"]
    if relation == "region_sum":
        answer = int(left["regions"]) + int(right["regions"])
    elif relation == "settlement_sum":
        answer = int(left["settlements"]) + int(right["settlements"])
    elif relation == "settlement_difference":
        answer = abs(
            int(left["settlements"]) - int(right["settlements"])
        )
    else:
        raise ValueError(f"unsupported source arithmetic relation: {relation}")
    return str(answer)


def _calibration_prompt_evidence(prompt: str) -> tuple[dict, str]:
    evidence_line = next(
        (
            line.removeprefix("Evidence: ")
            for line in prompt.splitlines()
            if line.startswith("Evidence: ")
        ),
        None,
    )
    claim_line = next(
        (
            line.removeprefix("Claim: ")
            for line in prompt.splitlines()
            if line.startswith("Claim: ")
        ),
        None,
    )
    if evidence_line is None or claim_line is None:
        raise ValueError("calibration prompt missing Evidence or Claim line")
    evidence = json.loads(evidence_line)
    if not isinstance(evidence, dict):
        raise ValueError("calibration prompt evidence must be an object")
    return evidence, claim_line


def _evidence_calibration_answer(row: dict) -> str:
    prompt_evidence, claim = _calibration_prompt_evidence(row["prompt"])
    code_match = re.fullmatch(
        r"The record assigns code (?P<code>\S+) to (?P<name>.+)\.",
        claim,
    )
    if code_match:
        expected_code = str(prompt_evidence.get("code"))
        expected_name = str(prompt_evidence.get("name"))
        return (
            "TRUE"
            if (
                code_match.group("code") == expected_code
                and code_match.group("name") == expected_name
            )
            else "FALSE"
        )

    language_match = re.fullmatch(
        r"The official language of (?P<name>.+) is English\.",
        claim,
    )
    if language_match:
        if "official_language" not in prompt_evidence:
            return "NOT_ENOUGH_INFO"
        if language_match.group("name") != str(prompt_evidence.get("name")):
            return "FALSE"
        return (
            "TRUE"
            if str(prompt_evidence["official_language"]).casefold()
            == "english"
            else "FALSE"
        )

    raise ValueError(f"unsupported calibration claim: {claim}")


def _validate_generated_code(
    row: dict,
    source_evidence: dict,
) -> tuple[list[str], dict]:
    grader = row["grader_contract"]
    expected_prompt = (
        f"Implement this Python function exactly: "
        f"{source_evidence['signature']}\n"
        f"Contract: {source_evidence['task']}\n"
        "Return only executable Python code."
    )
    expected_digest = sha256_json({"tests": source_evidence["tests"]})
    defects: list[str] = []
    if row["prompt"] != expected_prompt:
        defects.append("prompt_contract_mismatch")
    if grader.get("family") != source_evidence["family"]:
        defects.append("grader_family_mismatch")
    if grader.get("function") != source_evidence["function"]:
        defects.append("grader_function_mismatch")
    if grader.get("tests") != source_evidence["tests"]:
        defects.append("grader_tests_mismatch")
    if grader.get("answer_key_digest") != expected_digest:
        defects.append("grader_digest_mismatch")
    return defects, {
        "family": source_evidence["family"],
        "function": source_evidence["function"],
        "signature": source_evidence["signature"],
        "test_count": len(source_evidence["tests"]),
        "expected_grader_digest": expected_digest,
    }


def _validate_generated_instruction(
    row: dict,
    source_evidence: dict,
) -> tuple[list[str], dict]:
    grader = row["grader_contract"]
    constraints = source_evidence["constraints"]
    expected_digest = sha256_json(constraints)
    defects: list[str] = []
    if row["prompt"] != source_evidence["prompt"]:
        defects.append("prompt_contract_mismatch")
    if grader.get("constraints") != constraints:
        defects.append("grader_constraints_mismatch")
    if grader.get("answer_key_digest") != expected_digest:
        defects.append("grader_digest_mismatch")
    return defects, {
        "family": source_evidence["family"],
        "constraint_type": constraints["type"],
        "expected_grader_digest": expected_digest,
    }


def _structured_extraction_answer(source_evidence: dict) -> str:
    selected = {
        "name": source_evidence["name"],
        "code": source_evidence["code"],
        "regions": int(source_evidence["regions"]),
        "settlements": int(source_evidence["settlements"]),
    }
    return canonical_json(selected)


def _validate_structured_extraction(
    row: dict,
    source_evidence: dict,
) -> tuple[list[str], dict]:
    derived_answer = _structured_extraction_answer(source_evidence)
    expected_prompt = (
        "Read this frozen source record:\n"
        f"name={source_evidence['name']}; code={source_evidence['code']}; "
        f"regions={source_evidence['regions']}; "
        f"settlements={source_evidence['settlements']}\n"
        "Return exactly one compact JSON object with keys "
        '"name","code","regions","settlements" and the values from the record.'
    )
    grader = row["grader_contract"]
    expected_digest = sha256_json({"answer": derived_answer})
    defects: list[str] = []
    if row["prompt"] != expected_prompt:
        defects.append("prompt_contract_mismatch")
    if grader.get("answer_key") != derived_answer:
        defects.append("grader_answer_mismatch")
    if grader.get("answer_key_digest") != expected_digest:
        defects.append("grader_digest_mismatch")
    return defects, {
        "derived_answer": derived_answer,
        "grader_answer": grader.get("answer_key"),
        "expected_grader_digest": expected_digest,
    }


def validate_row_mechanically(
    row: dict,
    *,
    source_evidence: dict | None = None,
) -> dict:
    validator_class = validator_class_for_row(row)
    if source_evidence is None:
        raise ValueError("source_evidence is required")

    source_evidence_sha256 = sha256_json(source_evidence)
    if source_evidence_sha256 != row.get("source_hash"):
        return {
            "case_id": row["case_id"],
            "validator_class": validator_class,
            "status": "BINDING_DEFECT",
            "defects": ["source_hash_mismatch"],
            "mechanical_facts": {
                "source_evidence_sha256": source_evidence_sha256,
                "expected_source_sha256": row.get("source_hash"),
            },
        }

    if validator_class == "generated_code":
        defects, facts = _validate_generated_code(row, source_evidence)
        facts["source_evidence_sha256"] = source_evidence_sha256
        return {
            "case_id": row["case_id"],
            "validator_class": validator_class,
            "status": "BANK_DEFECT" if defects else "MECHANICAL_VALID",
            "defects": defects,
            "mechanical_facts": facts,
        }

    if validator_class == "generated_instruction":
        defects, facts = _validate_generated_instruction(row, source_evidence)
        facts["source_evidence_sha256"] = source_evidence_sha256
        return {
            "case_id": row["case_id"],
            "validator_class": validator_class,
            "status": "BANK_DEFECT" if defects else "MECHANICAL_VALID",
            "defects": defects,
            "mechanical_facts": facts,
        }

    if validator_class == "structured_extraction":
        defects, facts = _validate_structured_extraction(row, source_evidence)
        facts["source_evidence_sha256"] = source_evidence_sha256
        return {
            "case_id": row["case_id"],
            "validator_class": validator_class,
            "status": "BANK_DEFECT" if defects else "MECHANICAL_VALID",
            "defects": defects,
            "mechanical_facts": facts,
        }

    defects: list[str] = []
    if validator_class == "direct_source_lookup":
        derived_answer = _direct_source_answer(row, source_evidence)
        if row["prompt"] != _direct_source_expected_prompt(
            row,
            source_evidence,
        ):
            defects.append("prompt_source_mismatch")
    elif validator_class == "source_arithmetic":
        derived_answer = _source_arithmetic_answer(row, source_evidence)
        if row["prompt"] != _source_arithmetic_expected_prompt(
            row,
            source_evidence,
        ):
            defects.append("prompt_source_mismatch")
    elif validator_class == "evidence_calibration":
        prompt_evidence, claim = _calibration_prompt_evidence(row["prompt"])
        expected_prompt_evidence = {
            "name": source_evidence["name"],
            "code": source_evidence["code"],
            "regions": int(source_evidence["regions"]),
            "settlements": int(source_evidence["settlements"]),
        }
        prompt_lines = row["prompt"].splitlines()
        if (
            prompt_evidence != expected_prompt_evidence
            or len(prompt_lines) != 4
            or prompt_lines[0]
            != "Use only the evidence record below; do not add outside knowledge."
            or prompt_lines[3]
            != "Answer exactly TRUE, FALSE, or NOT_ENOUGH_INFO."
        ):
            defects.append("prompt_source_mismatch")
        code_match = re.fullmatch(
            r"The record assigns code (?P<code>\S+) to (?P<name>.+)\.",
            claim,
        )
        language_match = re.fullmatch(
            r"The official language of (?P<name>.+) is English\.",
            claim,
        )
        claim_name = (
            code_match.group("name")
            if code_match
            else language_match.group("name")
            if language_match
            else None
        )
        if claim_name != str(source_evidence["name"]):
            if "prompt_source_mismatch" not in defects:
                defects.append("prompt_source_mismatch")
        derived_answer = _evidence_calibration_answer(row)
    else:
        raise ValueError(f"unsupported validator class: {validator_class}")

    grader_answer = str(row["grader_contract"].get("answer_key"))
    if grader_answer != derived_answer:
        defects.append("grader_answer_mismatch")

    return {
        "case_id": row["case_id"],
        "validator_class": validator_class,
        "status": "BANK_DEFECT" if defects else "MECHANICAL_VALID",
        "defects": defects,
        "mechanical_facts": {
            "derived_answer": derived_answer,
            "grader_answer": grader_answer,
            "source_evidence_sha256": source_evidence_sha256,
        },
    }

def validate_candidate_mechanically(
    rows: list[dict],
    *,
    location_manifest: dict | None = None,
) -> dict:
    results: list[dict] = []
    for row in rows:
        try:
            evidence = resolve_source_evidence(
                row,
                location_manifest=location_manifest,
            )
        except SourceBindingError as exc:
            try:
                validator_class = validator_class_for_row(row)
            except ValueError:
                validator_class = "unclassified"
            results.append({
                "case_id": row.get("case_id"),
                "validator_class": validator_class,
                "status": "BINDING_DEFECT",
                "defects": ["source_resolution_failure"],
                "mechanical_facts": {"error": str(exc)},
            })
            continue
        results.append(
            validate_row_mechanically(
                row,
                source_evidence=evidence,
            )
        )

    status_counts = Counter(result["status"] for result in results)
    aggregate_status = (
        "MECHANICAL_VALIDATED"
        if results
        and status_counts == {"MECHANICAL_VALID": len(results)}
        else "MECHANICAL_HOLD"
    )
    return {
        "status": aggregate_status,
        "reviewed": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": results,
    }
