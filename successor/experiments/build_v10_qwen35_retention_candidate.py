from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

from successor.experiments.v10_qwen35_bank import (
    RETENTION_ALLOCATION,
    preflight_retention_rows,
    review_subject_digest,
)

LOCATION_RELEASE = "2026-09-20T1335Z"
LOCATION_REPO = "mindstellar/location-data"
LOCATION_REPO_COMMIT = "a0cf799530bb61d1ce2adaf00b6473e39c006c1a"
LOCATION_MANIFEST_URL = (
    "https://geo.mindstellar.com/releases/"
    + LOCATION_RELEASE
    + "/manifest.json"
)
LOCATION_MANIFEST_SHA256 = "bf5a2dc0bb0f1ba15f7149e9fa1184aa632f280a72fe306c154a9e05cfacab29"
LOCATION_CONTENT_FINGERPRINT = "0ebd427903c150a6"
LOCATION_TERMS = "CC0-1.0"
CONTRACT_REVISION = "V10_RETENTION_OBJECTIVE_CONTRACTS_20261001_V1"
GENERATOR_ID = "v10-retention-candidate-generator-v1"
AUDITOR_ID = "v10-retention-deterministic-contract-auditor-v1"

TEST_ALLOCATION = {
    "knowledge_factuality": 30,
    "reasoning_math": 30,
    "coding": 25,
    "instruction_following": 25,
    "extraction_structured": 20,
    "truthfulness_factual_calibration": 20,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_take(
    rows: list[Any],
    count: int,
    *,
    key: Callable[[Any], str],
    namespace: str,
) -> list[Any]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (namespace + "\0" + key(row)).encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"{namespace}: requested {count}, only {len(ranked)} available")
    return ranked[:count]


def _source_audit(subject_digest: str, artifact_digest: str) -> dict:
    return {
        "verdict": "SOURCE_CONTRACT_VALID",
        "auditor_id": AUDITOR_ID,
        "auditor_class": "SOURCE_BOUND_DETERMINISTIC_AUDIT",
        "provider": "vera_model_training",
        "runtime": "source-contract-recompute-v1",
        "independent_review": False,
        "artifact_digest": artifact_digest,
        "subject_digest": subject_digest,
    }


def _pending_review(subject_digest: str, artifact_digest: str) -> dict:
    return {
        "verdict": "PENDING_INDEPENDENT_REVIEW",
        "reviewer_id": "UNBOUND",
        "reviewer_class": "UNBOUND",
        "provider": "UNBOUND",
        "runtime": "UNBOUND",
        "independent_from_generation": False,
        "artifact_digest": artifact_digest,
        "subject_digest": subject_digest,
    }


def _grader(grader_id: str, answer_contract: Any, **extra: Any) -> dict:
    value = {
        "kind": "deterministic",
        "grader_id": grader_id,
        "grader_version": "1",
        "answer_key_digest": sha256_json(answer_contract),
    }
    value.update(extra)
    return value


def _row(
    *,
    case_id: str,
    category: str,
    family_id: str,
    prompt: str,
    source_id: str,
    source_revision: str,
    source_terms: str,
    source_record: Any,
    grader_contract: dict,
    artifact_digest: str,
) -> dict:
    source_digest = sha256_json(source_record)
    row = {
        "case_id": case_id,
        "lane": "retention",
        "category": category,
        "family_id": family_id,
        "prompt": prompt,
        "source_id": source_id,
        "source_revision": source_revision,
        "source_terms": source_terms,
        "source_hash": source_digest,
        "generation_method": "novel_deterministic_source_transform_v1",
        "generation_actor_id": GENERATOR_ID,
        "grader_contract": grader_contract,
    }
    row["review_receipt"] = _pending_review(
        review_subject_digest(row),
        artifact_digest,
    )
    row["source_audit_receipt"] = _source_audit(
        source_digest,
        artifact_digest,
    )
    return row


def validate_location_manifest(value: dict) -> None:
    if value.get("license") != LOCATION_TERMS:
        raise RuntimeError(
            f"location manifest license mismatch: {value.get('license')!r}"
        )
    if value.get("s_version") != LOCATION_CONTENT_FINGERPRINT:
        raise RuntimeError(
            f"location manifest s_version mismatch: {value.get('s_version')!r}"
        )
    if value.get("version") != LOCATION_CONTENT_FINGERPRINT:
        raise RuntimeError(
            f"location manifest version mismatch: {value.get('version')!r}"
        )
    if "wikidata" not in str(value.get("source", "")).casefold():
        raise RuntimeError("location manifest source is not Wikidata-bound")
    countries = value.get("countries")
    if not isinstance(countries, list) or len(countries) < 200:
        raise RuntimeError("location manifest countries missing or unexpectedly small")


def load_location_manifest() -> dict:
    with urlopen(LOCATION_MANIFEST_URL, timeout=30) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LOCATION_MANIFEST_SHA256:
        raise RuntimeError(
            f"location manifest hash mismatch: {digest} != {LOCATION_MANIFEST_SHA256}"
        )
    value = json.loads(raw.decode("utf-8"))
    validate_location_manifest(value)
    return value


def _location_source_id(country: dict) -> str:
    return f"{LOCATION_REPO}:{LOCATION_RELEASE}:country:{country['code']}"


def _location_row(
    *,
    case_id: str,
    category: str,
    family_id: str,
    prompt: str,
    country: dict,
    grader_contract: dict,
) -> dict:
    return _row(
        case_id=case_id,
        category=category,
        family_id=family_id,
        prompt=prompt,
        source_id=_location_source_id(country),
        source_revision=LOCATION_RELEASE,
        source_terms=LOCATION_TERMS,
        source_record=country,
        grader_contract=grader_contract,
        artifact_digest=LOCATION_MANIFEST_SHA256,
    )


def _build_knowledge(countries: list[dict], count: int) -> list[dict]:
    pool = []
    for country in countries:
        relations = (
            (
                "code_from_name",
                f"In the frozen {LOCATION_RELEASE} location-data release, what two-character country code is assigned to {country['name']}? Return only the code.",
                str(country["code"]),
            ),
            (
                "name_from_code",
                f"In the frozen {LOCATION_RELEASE} location-data release, which country or territory name is assigned to code {country['code']}? Return only the recorded name.",
                str(country["name"]),
            ),
            (
                "region_count",
                f"In the frozen {LOCATION_RELEASE} location-data release, how many administrative regions are recorded for {country['name']}? Return only the integer.",
                str(country["regions"]),
            ),
            (
                "settlement_count",
                f"In the frozen {LOCATION_RELEASE} location-data release, how many settlements are recorded for {country['name']}? Return only the integer.",
                str(country["settlements"]),
            ),
        )
        for relation, prompt, answer in relations:
            contract = {"answer": answer}
            pool.append(
                _location_row(
                    case_id=f"ret-kf-{country['code'].lower()}-{relation}",
                    category="knowledge_factuality",
                    family_id=f"location:{relation}",
                    prompt=prompt,
                    country=country,
                    grader_contract=_grader(
                        "exact_text_v1",
                        contract,
                        answer_key=answer,
                    ),
                )
            )
    return stable_take(
        pool, count, key=lambda row: row["case_id"], namespace="retention:knowledge"
    )


def _build_math(countries: list[dict], count: int) -> list[dict]:
    pool = []
    ordered = sorted(countries, key=lambda row: str(row["code"]))
    n = len(ordered)
    for i, left in enumerate(ordered):
        right = ordered[(i * 37 + 17) % n]
        if right["code"] == left["code"]:
            right = ordered[(i + 1) % n]
        specs = (
            (
                "settlement_sum",
                int(left["settlements"]) + int(right["settlements"]),
                (
                    f"A frozen source record lists {left['name']} with {left['settlements']} settlements "
                    f"and {right['name']} with {right['settlements']} settlements. "
                    "What is the sum? Return only the integer."
                ),
            ),
            (
                "settlement_difference",
                abs(int(left["settlements"]) - int(right["settlements"])),
                (
                    f"A frozen source record lists {left['name']} with {left['settlements']} settlements "
                    f"and {right['name']} with {right['settlements']} settlements. "
                    "What is the absolute difference? Return only the integer."
                ),
            ),
            (
                "region_sum",
                int(left["regions"]) + int(right["regions"]),
                (
                    f"A frozen source record lists {left['name']} with {left['regions']} regions "
                    f"and {right['name']} with {right['regions']} regions. "
                    "What is the total number of regions? Return only the integer."
                ),
            ),
        )
        source_record = {"left": left, "right": right}
        for relation, answer, prompt in specs:
            answer_text = str(answer)
            pool.append(
                _row(
                    case_id=f"ret-math-{left['code'].lower()}-{right['code'].lower()}-{relation}",
                    category="reasoning_math",
                    family_id=f"location-arithmetic:{relation}",
                    prompt=prompt,
                    source_id=(
                        f"{LOCATION_REPO}:{LOCATION_RELEASE}:pair:"
                        f"{left['code']}:{right['code']}"
                    ),
                    source_revision=LOCATION_RELEASE,
                    source_terms=LOCATION_TERMS,
                    source_record=source_record,
                    grader_contract=_grader(
                        "exact_numeric_v1",
                        {"answer": answer_text},
                        answer_key=answer_text,
                    ),
                    artifact_digest=LOCATION_MANIFEST_SHA256,
                )
            )
    return stable_take(pool, count, key=lambda row: row["case_id"], namespace="retention:math")


CODING_FAMILIES = (
    "rotate_left",
    "count_vowels",
    "unique_preserve",
    "digit_sum",
    "reverse_words",
    "clamp_values",
    "pairwise_sums",
    "flatten_once",
    "chunk_list",
    "frequency_map",
)


def _coding_spec(family: str, variant: int) -> dict:
    fn = f"solve_{family}_{variant:03d}"
    if family == "rotate_left":
        tests = [
            f"assert {fn}([1,2,3,4,5], {variant % 5}) == "
            + repr([1,2,3,4,5][variant % 5:] + [1,2,3,4,5][:variant % 5]),
            f"assert {fn}([], 3) == []",
        ]
        task = "Return a new list rotated left by k positions; handle empty lists."
        signature = f"def {fn}(values, k):"
    elif family == "count_vowels":
        sample = f"Vera{variant} Measurement"
        expected = sum(ch.lower() in "aeiou" for ch in sample)
        tests = [
            f"assert {fn}({sample!r}) == {expected}",
            f"assert {fn}('rhythms') == 0",
        ]
        task = "Count ASCII vowels a, e, i, o, u case-insensitively."
        signature = f"def {fn}(text):"
    elif family == "unique_preserve":
        values = [variant % 4, 2, variant % 4, 3, 2]
        expected = list(dict.fromkeys(values))
        tests = [f"assert {fn}({values!r}) == {expected!r}", f"assert {fn}([]) == []"]
        task = "Remove duplicates while preserving first-occurrence order."
        signature = f"def {fn}(values):"
    elif family == "digit_sum":
        value = 10000 + variant * 137
        expected = sum(int(ch) for ch in str(value))
        tests = [f"assert {fn}({value}) == {expected}", f"assert {fn}(0) == 0"]
        task = "Return the sum of decimal digits of a non-negative integer."
        signature = f"def {fn}(value):"
    elif family == "reverse_words":
        text = f"alpha beta item{variant}"
        expected = " ".join(reversed(text.split()))
        tests = [f"assert {fn}({text!r}) == {expected!r}", f"assert {fn}('one') == 'one'"]
        task = "Reverse whitespace-separated word order and join with single spaces."
        signature = f"def {fn}(text):"
    elif family == "clamp_values":
        values = [-3, variant % 8, 12]
        low, high = 1, 7
        expected = [min(high, max(low, x)) for x in values]
        tests = [f"assert {fn}({values!r}, {low}, {high}) == {expected!r}"]
        task = "Clamp every numeric list item to the inclusive [low, high] interval."
        signature = f"def {fn}(values, low, high):"
    elif family == "pairwise_sums":
        values = [variant % 7, 3, 5, 8]
        expected = [a + b for a, b in zip(values, values[1:])]
        tests = [f"assert {fn}({values!r}) == {expected!r}", f"assert {fn}([4]) == []"]
        task = "Return sums of each adjacent pair in order."
        signature = f"def {fn}(values):"
    elif family == "flatten_once":
        values = [[variant, variant + 1], [], [3, 4]]
        expected = [item for group in values for item in group]
        tests = [f"assert {fn}({values!r}) == {expected!r}", f"assert {fn}([]) == []"]
        task = "Flatten a list of lists by exactly one level."
        signature = f"def {fn}(groups):"
    elif family == "chunk_list":
        values = list(range(variant % 4 + 6))
        size = variant % 3 + 2
        expected = [values[i:i+size] for i in range(0, len(values), size)]
        tests = [f"assert {fn}({values!r}, {size}) == {expected!r}"]
        task = "Split a list into consecutive chunks of size n; keep a shorter final chunk."
        signature = f"def {fn}(values, n):"
    elif family == "frequency_map":
        values = ["a", f"x{variant%3}", "a", "b", f"x{variant%3}"]
        expected = dict(Counter(values))
        tests = [f"assert {fn}({values!r}) == {expected!r}", f"assert {fn}([]) == {{}}"]
        task = "Return a dictionary mapping each hashable item to its occurrence count."
        signature = f"def {fn}(values):"
    else:
        raise ValueError(family)
    return {"family": family, "variant": variant, "function": fn, "signature": signature, "task": task, "tests": tests}


def _build_coding(count: int) -> list[dict]:
    pool = []
    for family in CODING_FAMILIES:
        for variant in range(40):
            spec = _coding_spec(family, variant)
            prompt = (
                f"Implement this Python function exactly: {spec['signature']}\n"
                f"Contract: {spec['task']}\nReturn only executable Python code."
            )
            digest = sha256_json(spec)
            pool.append(
                _row(
                    case_id=f"ret-code-{family}-{variant:03d}",
                    category="coding",
                    family_id=f"generated-code:{family}",
                    prompt=prompt,
                    source_id=f"vera_model_training:{CONTRACT_REVISION}:coding:{family}:{variant}",
                    source_revision=CONTRACT_REVISION,
                    source_terms="INTERNAL_OBJECTIVE_CONTRACT",
                    source_record=spec,
                    grader_contract=_grader(
                        "python_unit_tests_v1",
                        {"tests": spec["tests"]},
                        tests=spec["tests"],
                        function=spec["function"],
                        family=family,
                    ),
                    artifact_digest=digest,
                )
            )
    return stable_take(pool, count, key=lambda row: row["case_id"], namespace="retention:coding")


INSTRUCTION_FAMILIES = (
    "token_count",
    "prefix_suffix",
    "line_prefixes",
    "json_keys",
    "forbidden_character",
)


def _instruction_spec(family: str, variant: int) -> dict:
    token = f"V{variant:03d}"
    if family == "token_count":
        constraints = {
            "type": "token_exact_count",
            "token": token,
            "count": variant % 3 + 1,
        }
        prompt = (
            f"Write one short sentence about careful measurement. Include the exact token {token} "
            f"exactly {constraints['count']} time(s)."
        )
    elif family == "prefix_suffix":
        constraints = {
            "type": "prefix_suffix",
            "prefix": f"START-{variant}:",
            "suffix": f":END-{variant}",
        }
        prompt = (
            f"Write a one-line status update that begins exactly with {constraints['prefix']!r} "
            f"and ends exactly with {constraints['suffix']!r}."
        )
    elif family == "line_prefixes":
        prefixes = [f"A{variant}-", f"B{variant}-", f"C{variant}-"]
        constraints = {"type": "line_prefixes", "prefixes": prefixes}
        prompt = (
            "Write exactly three nonempty lines. "
            + " ".join(
                f"Line {i+1} must begin exactly with {prefix!r}."
                for i, prefix in enumerate(prefixes)
            )
        )
    elif family == "json_keys":
        keys = ["result", f"code_{variant}", "verified"]
        constraints = {"type": "json_object_exact_keys", "keys": keys}
        prompt = (
            "Return exactly one valid JSON object and no surrounding prose. "
            f"Its keys must be exactly {keys!r}; values may be any valid JSON scalars."
        )
    elif family == "forbidden_character":
        forbidden = [",", ";", "!", "?"][variant % 4]
        constraints = {
            "type": "forbidden_character_and_required_token",
            "forbidden": forbidden,
            "required_token": token,
        }
        prompt = (
            f"Write a short plain-text note containing the exact token {token}. "
            f"Do not use the character {forbidden!r} anywhere."
        )
    else:
        raise ValueError(family)
    return {"family": family, "variant": variant, "prompt": prompt, "constraints": constraints}


def _build_instruction_following(count: int) -> list[dict]:
    pool = []
    for family in INSTRUCTION_FAMILIES:
        for variant in range(80):
            spec = _instruction_spec(family, variant)
            digest = sha256_json(spec)
            pool.append(
                _row(
                    case_id=f"ret-if-{family}-{variant:03d}",
                    category="instruction_following",
                    family_id=f"generated-instruction:{family}",
                    prompt=spec["prompt"],
                    source_id=(
                        f"vera_model_training:{CONTRACT_REVISION}:"
                        f"instruction:{family}:{variant}"
                    ),
                    source_revision=CONTRACT_REVISION,
                    source_terms="INTERNAL_OBJECTIVE_CONTRACT",
                    source_record=spec,
                    grader_contract=_grader(
                        "instruction_contract_v1",
                        spec["constraints"],
                        constraints=spec["constraints"],
                    ),
                    artifact_digest=digest,
                )
            )
    return stable_take(
        pool,
        count,
        key=lambda row: row["case_id"],
        namespace="retention:instruction-following",
    )


def _evidence_record(country: dict) -> dict:
    return {
        "name": country["name"],
        "code": country["code"],
        "regions": int(country["regions"]),
        "settlements": int(country["settlements"]),
    }


def _build_extraction(countries: list[dict], count: int) -> list[dict]:
    selected = stable_take(
        countries,
        count,
        key=lambda row: str(row["code"]),
        namespace="retention:extraction",
    )
    rows = []
    for country in selected:
        evidence = _evidence_record(country)
        prompt = (
            "Read this frozen source record:\n"
            f"name={evidence['name']}; code={evidence['code']}; "
            f"regions={evidence['regions']}; settlements={evidence['settlements']}\n"
            "Return exactly one compact JSON object with keys "
            '"name","code","regions","settlements" and the values from the record.'
        )
        answer = canonical_json(evidence)
        rows.append(
            _location_row(
                case_id=f"ret-extract-{country['code'].lower()}",
                category="extraction_structured",
                family_id="location:structured-extraction",
                prompt=prompt,
                country=country,
                grader_contract=_grader(
                    "exact_json_v1",
                    {"answer": answer},
                    answer_key=answer,
                ),
            )
        )
    return rows


def _build_calibration(countries: list[dict], count: int) -> list[dict]:
    selected = stable_take(
        countries,
        count,
        key=lambda row: str(row["code"]),
        namespace="retention:calibration",
    )
    ordered = sorted(countries, key=lambda row: str(row["code"]))
    by_code = {row["code"]: i for i, row in enumerate(ordered)}
    rows = []
    for index, country in enumerate(selected):
        evidence = _evidence_record(country)
        mode = index % 3
        if mode == 0:
            claim = f"The record assigns code {country['code']} to {country['name']}."
            answer = "TRUE"
        elif mode == 1:
            pos = by_code[country["code"]]
            wrong = ordered[(pos + 1) % len(ordered)]["code"]
            claim = f"The record assigns code {wrong} to {country['name']}."
            answer = "FALSE"
        else:
            claim = f"The official language of {country['name']} is English."
            answer = "NOT_ENOUGH_INFO"
        prompt = (
            "Use only the evidence record below; do not add outside knowledge.\n"
            f"Evidence: {canonical_json(evidence)}\n"
            f"Claim: {claim}\n"
            "Answer exactly TRUE, FALSE, or NOT_ENOUGH_INFO."
        )
        rows.append(
            _location_row(
                case_id=f"ret-cal-{country['code'].lower()}-{mode}",
                category="truthfulness_factual_calibration",
                family_id=f"evidence-calibration:mode-{mode}",
                prompt=prompt,
                country=country,
                grader_contract=_grader(
                    "exact_choice_v1",
                    {"answer": answer},
                    answer_key=answer,
                ),
            )
        )
    return rows


def build_candidate_rows(
    countries: list[dict],
    *,
    scale_for_test: bool = False,
) -> list[dict]:
    allocation = TEST_ALLOCATION if scale_for_test else RETENTION_ALLOCATION
    rows = []
    rows.extend(_build_knowledge(countries, allocation["knowledge_factuality"]))
    rows.extend(_build_math(countries, allocation["reasoning_math"]))
    rows.extend(_build_coding(allocation["coding"]))
    rows.extend(_build_instruction_following(allocation["instruction_following"]))
    rows.extend(_build_extraction(countries, allocation["extraction_structured"]))
    rows.extend(
        _build_calibration(
            countries,
            allocation["truthfulness_factual_calibration"],
        )
    )
    return sorted(rows, key=lambda row: row["case_id"])


def load_exclusion_hashes(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    }


def build_candidate(*, exclusion_hashes: set[str]) -> tuple[list[dict], dict]:
    location_manifest = load_location_manifest()
    countries = location_manifest["countries"]
    rows = build_candidate_rows(countries)
    preflight = preflight_retention_rows(rows, exclusion_hashes=exclusion_hashes)
    if preflight["status"] != "STRUCTURE_READY":
        raise RuntimeError("retention candidate preflight failed: " + canonical_json(preflight))

    payload = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    manifest = {
        "schema": "V10_QWEN35_RETENTION_CANDIDATE_MANIFEST_V1",
        "bank_id": "V10_QWEN35_RETENTION_CANDIDATE_1500_20261001_V1",
        "status": "CANDIDATE_OBJECTIVE_STRUCTURE_READY_FAMILY_AUDIT_PENDING",
        "case_count": len(rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "location_source": {
            "repo": LOCATION_REPO,
            "repo_commit": LOCATION_REPO_COMMIT,
            "release": LOCATION_RELEASE,
            "release_content_fingerprint": LOCATION_CONTENT_FINGERPRINT,
            "manifest_url": LOCATION_MANIFEST_URL,
            "manifest_sha256": LOCATION_MANIFEST_SHA256,
            "terms": LOCATION_TERMS,
        },
        "generated_contracts": {
            "revision": CONTRACT_REVISION,
            "coding_families": list(CODING_FAMILIES),
            "instruction_families": list(INSTRUCTION_FAMILIES),
        },
        "public_benchmark_raw_imports": [],
        "external_shadow_policy": (
            "MMLU-Pro, GSM8K, MBPP, IFEval, TruthfulQA and similar public "
            "benchmarks remain shadow/design references and are not copied into this candidate."
        ),
        "preflight": preflight,
        "independent_family_audit": {
            "policy": "V10_QWEN35_RETENTION_ADMISSION_V3",
            "required": True,
            "cases_per_family": 5,
            "selection": (
                "SHA256(candidate_sha256, family_id, case_id) "
                "lowest 5 per family"
            ),
            "status": "UNBOUND",
            "source_bound_deterministic_audit_is_not_external_independence": True,
        },
        "claim_ceiling": (
            "OBJECTIVE_RETENTION_CANDIDATE_1500_STRUCTURALLY_READY / "
            "NO_RAW_PUBLIC_BENCHMARK_IMPORT / "
            "V3_FAMILY_AUDIT_PENDING / "
            "SEMANTIC_CONTAMINATION_SCREEN_EXTERNAL_RECEIPT_REQUIRED / "
            "NOT_FINAL_BANK"
        ),
    }
    return rows, manifest


def write_candidate_files(
    rows: list[dict],
    manifest: dict,
    output_path: Path,
    manifest_path: Path,
) -> dict:
    payload = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)

    written_manifest = dict(manifest)
    written_manifest["data_sha256"] = hashlib.sha256(payload).hexdigest()
    written_manifest["data_path"] = output_path.as_posix()
    written_manifest.pop("manifest_sha256", None)
    written_manifest["manifest_sha256"] = sha256_json(written_manifest)
    manifest_path.write_bytes(
        (json.dumps(written_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return written_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion-hashes", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = build_candidate(
        exclusion_hashes=load_exclusion_hashes(args.exclusion_hashes)
    )
    manifest = write_candidate_files(rows, manifest, args.output, args.manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
