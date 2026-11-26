from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from successor.experiments.retention_audit_arch_v2_validators import (
    canonical_json,
    sha256_json,
    validate_row_mechanically,
)


def _base_row(
    *,
    case_id: str,
    category: str,
    family_id: str,
    prompt: str,
    evidence: dict,
    grader_contract: dict,
) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "family_id": family_id,
        "prompt": prompt,
        "source_id": f"arch-v2:qualification:{case_id}",
        "source_revision": "ARCH_V2_QUALIFICATION",
        "source_terms": "INTERNAL_QUALIFICATION",
        "source_hash": sha256_json(evidence),
        "grader_contract": grader_contract,
    }


def _answer_grader(grader_id: str, answer: str) -> dict:
    return {
        "kind": "deterministic",
        "grader_id": grader_id,
        "grader_version": "1",
        "answer_key": answer,
        "answer_key_digest": sha256_json({"answer": answer}),
    }


def _direct_lookup_cases() -> list[dict]:
    evidence = {
        "code": "AA",
        "name": "Alpha",
        "regions": 3,
        "settlements": 7,
    }
    revision = "ARCH_V2_QUALIFICATION"
    good = _base_row(
        case_id="arch-v2-qual-direct-good",
        category="knowledge_factuality",
        family_id="location:code_from_name",
        prompt=(
            f"In the frozen {revision} location-data release, what two-character "
            "country code is assigned to Alpha? Return only the code."
        ),
        evidence=evidence,
        grader_contract=_answer_grader("exact_text_v1", "AA"),
    )
    cases = [{
        "validator_class": "direct_source_lookup",
        "label": "known_good",
        "kind": "good",
        "row": good,
        "source_evidence": deepcopy(evidence),
        "expected_status": "MECHANICAL_VALID",
    }]

    wrong_code = deepcopy(good)
    wrong_code["case_id"] = "arch-v2-qual-direct-wrong-code"
    wrong_code["grader_contract"] = _answer_grader("exact_text_v1", "ZZ")
    cases.append({
        "validator_class": "direct_source_lookup",
        "label": "wrong_code",
        "kind": "seeded_defect",
        "row": wrong_code,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    wrong_name = _base_row(
        case_id="arch-v2-qual-direct-wrong-name",
        category="knowledge_factuality",
        family_id="location:name_from_code",
        prompt=(
            f"In the frozen {revision} location-data release, which country or "
            "territory name is assigned to code AA? Return only the recorded name."
        ),
        evidence=evidence,
        grader_contract=_answer_grader("exact_text_v1", "Wrongland"),
    )
    cases.append({
        "validator_class": "direct_source_lookup",
        "label": "wrong_name",
        "kind": "seeded_defect",
        "row": wrong_name,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    stale_regions = _base_row(
        case_id="arch-v2-qual-direct-stale-region-count",
        category="knowledge_factuality",
        family_id="location:region_count",
        prompt=(
            f"In the frozen {revision} location-data release, how many "
            "administrative regions are recorded for Alpha? Return only the integer."
        ),
        evidence=evidence,
        grader_contract=_answer_grader("exact_text_v1", "4"),
    )
    cases.append({
        "validator_class": "direct_source_lookup",
        "label": "stale_region_count",
        "kind": "seeded_defect",
        "row": stale_regions,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    stale_settlements = _base_row(
        case_id="arch-v2-qual-direct-stale-settlement-count",
        category="knowledge_factuality",
        family_id="location:settlement_count",
        prompt=(
            f"In the frozen {revision} location-data release, how many "
            "settlements are recorded for Alpha? Return only the integer."
        ),
        evidence=evidence,
        grader_contract=_answer_grader("exact_text_v1", "8"),
    )
    cases.append({
        "validator_class": "direct_source_lookup",
        "label": "stale_settlement_count",
        "kind": "seeded_defect",
        "row": stale_settlements,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })
    return cases

def _arithmetic_cases() -> list[dict]:
    evidence = {
        "left": {
            "code": "AA",
            "name": "Alpha",
            "regions": 3,
            "settlements": 7,
        },
        "right": {
            "code": "BB",
            "name": "Beta",
            "regions": 4,
            "settlements": 5,
        },
    }
    prompt = (
        "A frozen source record lists Alpha with 3 regions and "
        "Beta with 4 regions. What is the total number of regions? "
        "Return only the integer."
    )
    good = _base_row(
        case_id="arch-v2-qual-arithmetic-good",
        category="reasoning_math",
        family_id="location-arithmetic:region_sum",
        prompt=prompt,
        evidence=evidence,
        grader_contract=_answer_grader("exact_numeric_v1", "7"),
    )
    cases = [{
        "validator_class": "source_arithmetic",
        "label": "known_good",
        "kind": "good",
        "row": good,
        "source_evidence": deepcopy(evidence),
        "expected_status": "MECHANICAL_VALID",
    }]

    off_by_one = deepcopy(good)
    off_by_one["case_id"] = "arch-v2-qual-arithmetic-off-by-one"
    off_by_one["grader_contract"] = _answer_grader("exact_numeric_v1", "8")
    cases.append({
        "validator_class": "source_arithmetic",
        "label": "off_by_one_sum",
        "kind": "seeded_defect",
        "row": off_by_one,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    wrong_operand = deepcopy(good)
    wrong_operand["case_id"] = "arch-v2-qual-arithmetic-wrong-operand"
    wrong_operand["prompt"] = prompt.replace("3 regions", "99 regions")
    cases.append({
        "validator_class": "source_arithmetic",
        "label": "wrong_operand_value",
        "kind": "seeded_defect",
        "row": wrong_operand,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    diff_evidence = {
        "left": {
            "code": "CC",
            "name": "Gamma",
            "regions": 1,
            "settlements": 5,
        },
        "right": {
            "code": "DD",
            "name": "Delta",
            "regions": 2,
            "settlements": 7,
        },
    }
    signed_diff = _base_row(
        case_id="arch-v2-qual-arithmetic-signed-difference",
        category="reasoning_math",
        family_id="location-arithmetic:settlement_difference",
        prompt=(
            "A frozen source record lists Gamma with 5 settlements and "
            "Delta with 7 settlements. What is the absolute difference? "
            "Return only the integer."
        ),
        evidence=diff_evidence,
        grader_contract=_answer_grader("exact_numeric_v1", "-2"),
    )
    cases.append({
        "validator_class": "source_arithmetic",
        "label": "signed_instead_of_absolute_difference",
        "kind": "seeded_defect",
        "row": signed_diff,
        "source_evidence": deepcopy(diff_evidence),
        "expected_status": "BANK_DEFECT",
    })
    return cases

def _calibration_cases() -> list[dict]:
    evidence = {
        "code": "AA",
        "name": "Alpha",
        "regions": 3,
        "settlements": 7,
    }

    def prompt(claim: str, evidence_value: dict | None = None) -> str:
        value = evidence if evidence_value is None else evidence_value
        return (
            "Use only the evidence record below; do not add outside knowledge.\n"
            f"Evidence: {canonical_json(value)}\n"
            f"Claim: {claim}\n"
            "Answer exactly TRUE, FALSE, or NOT_ENOUGH_INFO."
        )

    good = _base_row(
        case_id="arch-v2-qual-calibration-good",
        category="truthfulness_factual_calibration",
        family_id="evidence-calibration:mode-0",
        prompt=prompt("The record assigns code AA to Alpha."),
        evidence=evidence,
        grader_contract=_answer_grader("exact_choice_v1", "TRUE"),
    )
    cases = [{
        "validator_class": "evidence_calibration",
        "label": "known_good",
        "kind": "good",
        "row": good,
        "source_evidence": deepcopy(evidence),
        "expected_status": "MECHANICAL_VALID",
    }]

    true_false = deepcopy(good)
    true_false["case_id"] = "arch-v2-qual-calibration-true-false"
    true_false["grader_contract"] = _answer_grader("exact_choice_v1", "FALSE")
    cases.append({
        "validator_class": "evidence_calibration",
        "label": "true_false_swap",
        "kind": "seeded_defect",
        "row": true_false,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    false_true = _base_row(
        case_id="arch-v2-qual-calibration-false-true",
        category="truthfulness_factual_calibration",
        family_id="evidence-calibration:mode-1",
        prompt=prompt("The record assigns code ZZ to Alpha."),
        evidence=evidence,
        grader_contract=_answer_grader("exact_choice_v1", "TRUE"),
    )
    cases.append({
        "validator_class": "evidence_calibration",
        "label": "false_true_swap",
        "kind": "seeded_defect",
        "row": false_true,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    mode2_true = _base_row(
        case_id="arch-v2-qual-calibration-mode2-true",
        category="truthfulness_factual_calibration",
        family_id="evidence-calibration:mode-2",
        prompt=prompt("The official language of Alpha is English."),
        evidence=evidence,
        grader_contract=_answer_grader("exact_choice_v1", "TRUE"),
    )
    cases.append({
        "validator_class": "evidence_calibration",
        "label": "mode2_changed_to_true",
        "kind": "seeded_defect",
        "row": mode2_true,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    changed_evidence = deepcopy(evidence)
    changed_evidence["code"] = "ZZ"
    prompt_changed = _base_row(
        case_id="arch-v2-qual-calibration-prompt-evidence-changed",
        category="truthfulness_factual_calibration",
        family_id="evidence-calibration:mode-0",
        prompt=prompt("The record assigns code ZZ to Alpha.", changed_evidence),
        evidence=evidence,
        grader_contract=_answer_grader("exact_choice_v1", "TRUE"),
    )
    cases.append({
        "validator_class": "evidence_calibration",
        "label": "prompt_evidence_changed",
        "kind": "seeded_defect",
        "row": prompt_changed,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })
    return cases

def _generated_code_cases() -> list[dict]:
    evidence = {
        "family": "chunk_list",
        "variant": 1,
        "function": "solve_chunk_list_001",
        "signature": "def solve_chunk_list_001(values, n):",
        "task": (
            "Split a list into consecutive chunks of size n; "
            "keep a shorter final chunk."
        ),
        "tests": [
            "assert solve_chunk_list_001([0, 1, 2, 3], 3) == "
            "[[0, 1, 2], [3]]"
        ],
    }
    prompt = (
        f"Implement this Python function exactly: {evidence['signature']}\n"
        f"Contract: {evidence['task']}\n"
        "Return only executable Python code."
    )
    grader = {
        "kind": "deterministic",
        "grader_id": "python_unit_tests_v1",
        "grader_version": "1",
        "answer_key_digest": sha256_json({"tests": evidence["tests"]}),
        "tests": evidence["tests"],
        "function": evidence["function"],
        "family": evidence["family"],
    }
    good = _base_row(
        case_id="arch-v2-qual-code-good",
        category="coding",
        family_id="generated-code:chunk_list",
        prompt=prompt,
        evidence=evidence,
        grader_contract=grader,
    )
    cases = [{
        "validator_class": "generated_code",
        "label": "known_good",
        "kind": "good",
        "row": good,
        "source_evidence": deepcopy(evidence),
        "expected_status": "MECHANICAL_VALID",
    }]

    signature = deepcopy(good)
    signature["case_id"] = "arch-v2-qual-code-signature-mismatch"
    signature["prompt"] = prompt.replace(
        "def solve_chunk_list_001(values, n):",
        "def solve_chunk_list_001(values):",
    )
    cases.append({
        "validator_class": "generated_code",
        "label": "signature_mismatch",
        "kind": "seeded_defect",
        "row": signature,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    function = deepcopy(good)
    function["case_id"] = "arch-v2-qual-code-function-mismatch"
    function["grader_contract"]["function"] = "solve_chunk_list_WRONG"
    cases.append({
        "validator_class": "generated_code",
        "label": "function_mismatch",
        "kind": "seeded_defect",
        "row": function,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })

    removed_test = deepcopy(good)
    removed_test["case_id"] = "arch-v2-qual-code-stored-test-removed"
    removed_test["grader_contract"]["tests"] = []
    removed_test["grader_contract"]["answer_key_digest"] = sha256_json({"tests": []})
    cases.append({
        "validator_class": "generated_code",
        "label": "stored_test_removed",
        "kind": "seeded_defect",
        "row": removed_test,
        "source_evidence": deepcopy(evidence),
        "expected_status": "BANK_DEFECT",
    })
    return cases

def _generated_instruction_cases() -> list[dict]:
    constraints = {
        "type": "forbidden_character_and_required_token",
        "forbidden": "?",
        "required_token": "V001",
    }
    prompt = (
        "Write a short plain-text note containing the exact token V001. "
        "Do not use the character '?' anywhere."
    )
    evidence = {
        "family": "forbidden_character",
        "variant": 1,
        "prompt": prompt,
        "constraints": constraints,
    }
    grader = {
        "kind": "deterministic",
        "grader_id": "instruction_contract_v1",
        "grader_version": "1",
        "answer_key_digest": sha256_json(constraints),
        "constraints": deepcopy(constraints),
    }
    row = _base_row(
        case_id="arch-v2-qual-instruction-good",
        category="instruction_following",
        family_id="generated-instruction:forbidden_character",
        prompt=prompt,
        evidence=evidence,
        grader_contract=grader,
    )
    bad = deepcopy(row)
    bad["case_id"] = "arch-v2-qual-instruction-ignored-token"
    bad["grader_contract"]["constraints"]["required_token"] = "WRONG"
    return _paired_cases("generated_instruction", row, bad, evidence)


def _structured_extraction_cases() -> list[dict]:
    evidence = {
        "code": "AA",
        "name": "Alpha",
        "regions": 3,
        "settlements": 7,
    }
    answer = canonical_json(evidence)
    prompt = (
        "Read this frozen source record:\n"
        "name=Alpha; code=AA; regions=3; settlements=7\n"
        "Return exactly one compact JSON object with keys "
        '"name","code","regions","settlements" and the values from the record.'
    )
    good = _base_row(
        case_id="arch-v2-qual-extraction-good",
        category="extraction_structured",
        family_id="location:structured-extraction",
        prompt=prompt,
        evidence=evidence,
        grader_contract=_answer_grader("exact_json_v1", answer),
    )
    cases = [{
        "validator_class": "structured_extraction",
        "label": "known_good",
        "kind": "good",
        "row": good,
        "source_evidence": deepcopy(evidence),
        "expected_status": "MECHANICAL_VALID",
    }]

    mutations = {
        "missing_key": {
            "code": "AA",
            "name": "Alpha",
            "regions": 3,
        },
        "wrong_value": {
            "code": "ZZ",
            "name": "Alpha",
            "regions": 3,
            "settlements": 7,
        },
        "extra_key": {
            "code": "AA",
            "name": "Alpha",
            "regions": 3,
            "settlements": 7,
            "extra": True,
        },
    }
    for label, value in mutations.items():
        bad = deepcopy(good)
        bad["case_id"] = f"arch-v2-qual-extraction-{label}"
        bad["grader_contract"] = _answer_grader(
            "exact_json_v1",
            canonical_json(value),
        )
        cases.append({
            "validator_class": "structured_extraction",
            "label": label,
            "kind": "seeded_defect",
            "row": bad,
            "source_evidence": deepcopy(evidence),
            "expected_status": "BANK_DEFECT",
        })
    return cases

def _paired_cases(
    validator_class: str,
    good_row: dict,
    bad_row: dict,
    evidence: dict,
) -> list[dict]:
    return [
        {
            "validator_class": validator_class,
            "label": "known_good",
            "kind": "good",
            "row": good_row,
            "source_evidence": deepcopy(evidence),
            "expected_status": "MECHANICAL_VALID",
        },
        {
            "validator_class": validator_class,
            "label": "seeded_defect",
            "kind": "seeded_defect",
            "row": bad_row,
            "source_evidence": deepcopy(evidence),
            "expected_status": "BANK_DEFECT",
        },
    ]


def build_validator_mutation_cases() -> list[dict]:
    cases: list[dict] = []
    cases.extend(_direct_lookup_cases())
    cases.extend(_arithmetic_cases())
    cases.extend(_calibration_cases())
    cases.extend(_generated_code_cases())
    cases.extend(_generated_instruction_cases())
    cases.extend(_structured_extraction_cases())
    return cases


def qualify_validators() -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in build_validator_mutation_cases():
        grouped[case["validator_class"]].append(case)

    validators: dict[str, dict] = {}
    for validator_class, cases in sorted(grouped.items()):
        missed_defects: list[str] = []
        false_positives: list[str] = []
        good_controls = 0
        seeded_defects = 0

        for case in cases:
            result = validate_row_mechanically(
                case["row"],
                source_evidence=case["source_evidence"],
            )
            if case["kind"] == "good":
                good_controls += 1
                if result["status"] != case["expected_status"]:
                    false_positives.append(case["label"])
            else:
                seeded_defects += 1
                if result["status"] != case["expected_status"]:
                    missed_defects.append(case["label"])

        if missed_defects:
            status = "UNQUALIFIED_FALSE_NEGATIVE"
        elif false_positives:
            status = "UNQUALIFIED_FALSE_POSITIVE"
        elif not good_controls or not seeded_defects:
            status = "UNQUALIFIED_COVERAGE_GAP"
        else:
            status = "QUALIFIED"

        validators[validator_class] = {
            "status": status,
            "good_controls": good_controls,
            "seeded_defects": seeded_defects,
            "missed_defects": missed_defects,
            "false_positives": false_positives,
        }

    code_behavior = qualify_code_family_behavior()
    instruction_behavior = qualify_instruction_family_behavior()
    basic_qualified = (
        bool(validators)
        and all(
            item["status"] == "QUALIFIED"
            for item in validators.values()
        )
    )
    behavior_qualified = (
        code_behavior["status"] == "QUALIFIED"
        and instruction_behavior["status"] == "QUALIFIED"
    )
    return {
        "status": (
            "VALIDATORS_QUALIFIED"
            if basic_qualified and behavior_qualified
            else "VALIDATOR_QUALIFICATION_HOLD"
        ),
        "validators": validators,
        "code_behavior": code_behavior,
        "instruction_behavior": instruction_behavior,
    }

def _run_cases(fn, cases: list[tuple[tuple, object]]) -> bool:
    for args, expected in cases:
        try:
            actual = fn(*args)
        except Exception:
            return False
        if actual != expected:
            return False
    return True


def qualify_code_family_behavior() -> dict:
    def rotate_left_good(values, k):
        return [] if not values else list(values[k % len(values):]) + list(values[:k % len(values)])

    def rotate_left_mutant(values, k):
        return [] if not values else list(values[-(k % len(values)):]) + list(values[:-(k % len(values))])

    def count_vowels_good(text):
        return sum(ch.casefold() in "aeiou" for ch in text)

    def count_vowels_mutant(text):
        return sum(ch.casefold() in "aeiouy" for ch in text)

    def unique_preserve_good(values):
        return list(dict.fromkeys(values))

    def unique_preserve_mutant(values):
        return sorted(set(values))

    def digit_sum_good(value):
        return sum(int(ch) for ch in str(value))

    def digit_sum_mutant(value):
        return sum(int(ch) for ch in str(value)) + (1 if value else 0)

    def reverse_words_good(text):
        return " ".join(reversed(text.split()))

    def reverse_words_mutant(text):
        return " ".join(word[::-1] for word in text.split())

    def clamp_values_good(values, low, high):
        return [min(high, max(low, item)) for item in values]

    def clamp_values_mutant(values, low, high):
        return [max(low, item) for item in values]

    def pairwise_sums_good(values):
        return [a + b for a, b in zip(values, values[1:])]

    def pairwise_sums_mutant(values):
        return [values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)]

    def flatten_once_good(groups):
        return [item for group in groups for item in group]

    def flatten_once_mutant(groups):
        return list(groups[0]) if groups else []

    def chunk_list_good(values, n):
        return [values[i:i + n] for i in range(0, len(values), n)]

    def chunk_list_mutant(values, n):
        return [
            values[i:i + n]
            for i in range(0, len(values) - (len(values) % n), n)
        ]

    def frequency_map_good(values):
        counts = {}
        for item in values:
            counts[item] = counts.get(item, 0) + 1
        return counts

    def frequency_map_mutant(values):
        return {item: 1 for item in values}

    fixtures = {
        "rotate_left": (
            rotate_left_good,
            rotate_left_mutant,
            [(([1, 2, 3], 1), [2, 3, 1]), (([], 4), [])],
        ),
        "count_vowels": (
            count_vowels_good,
            count_vowels_mutant,
            [(("rhythms",), 0), (("AEiou",), 5)],
        ),
        "unique_preserve": (
            unique_preserve_good,
            unique_preserve_mutant,
            [(([2, 1, 2, 3, 1],), [2, 1, 3]), (([],), [])],
        ),
        "digit_sum": (
            digit_sum_good,
            digit_sum_mutant,
            [((12345,), 15), ((0,), 0)],
        ),
        "reverse_words": (
            reverse_words_good,
            reverse_words_mutant,
            [(("alpha beta gamma",), "gamma beta alpha"), (("one",), "one")],
        ),
        "clamp_values": (
            clamp_values_good,
            clamp_values_mutant,
            [(([-2, 4, 10], 1, 7), [1, 4, 7])],
        ),
        "pairwise_sums": (
            pairwise_sums_good,
            pairwise_sums_mutant,
            [(([1, 2, 3, 4],), [3, 5, 7]), (([4],), [])],
        ),
        "flatten_once": (
            flatten_once_good,
            flatten_once_mutant,
            [(([[1, 2], [], [3, 4]],), [1, 2, 3, 4]), (([],), [])],
        ),
        "chunk_list": (
            chunk_list_good,
            chunk_list_mutant,
            [(([0, 1, 2, 3, 4], 2), [[0, 1], [2, 3], [4]])],
        ),
        "frequency_map": (
            frequency_map_good,
            frequency_map_mutant,
            [((["a", "b", "a"],), {"a": 2, "b": 1}), (([],), {})],
        ),
    }

    families = {}
    for family, (good, mutant, cases) in fixtures.items():
        good_passed = _run_cases(good, cases)
        mutant_passed = _run_cases(mutant, cases)
        families[family] = {
            "good_passed": good_passed,
            "mutant_killed": not mutant_passed,
        }

    return {
        "status": (
            "QUALIFIED"
            if all(
                item["good_passed"] and item["mutant_killed"]
                for item in families.values()
            )
            else "UNQUALIFIED_COVERAGE_GAP"
        ),
        "families": families,
    }


def _check_instruction_contract_independent(text: str, contract: dict) -> bool:
    kind = contract["type"]
    if kind == "token_exact_count":
        return text.count(contract["token"]) == int(contract["count"])
    if kind == "prefix_suffix":
        return (
            text.startswith(contract["prefix"])
            and text.endswith(contract["suffix"])
        )
    if kind == "line_prefixes":
        lines = text.splitlines()
        prefixes = list(contract["prefixes"])
        return (
            len(lines) == len(prefixes)
            and all(lines)
            and all(
                line.startswith(prefix)
                for line, prefix in zip(lines, prefixes)
            )
        )
    if kind == "json_object_exact_keys":
        import json

        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(value, dict)
            and set(value) == set(contract["keys"])
            and all(
                item is None or isinstance(item, (str, int, float, bool))
                for item in value.values()
            )
        )
    if kind == "forbidden_character_and_required_token":
        return (
            contract["forbidden"] not in text
            and contract["required_token"] in text
        )
    raise ValueError(f"unknown independent instruction contract:{kind}")


def qualify_instruction_family_behavior() -> dict:
    fixtures = {
        "token_count": (
            {"type": "token_exact_count", "token": "V001", "count": 2},
            "V001 measurement V001",
            "V001 measurement",
        ),
        "prefix_suffix": (
            {"type": "prefix_suffix", "prefix": "START:", "suffix": ":END"},
            "START:verified:END",
            "START:verified",
        ),
        "line_prefixes": (
            {"type": "line_prefixes", "prefixes": ["A-", "B-", "C-"]},
            "A-one\nB-two\nC-three",
            "A-one\nX-two\nC-three",
        ),
        "json_keys": (
            {"type": "json_object_exact_keys", "keys": ["a", "b"]},
            '{"a":1,"b":2}',
            '{"a":1,"b":2,"c":3}',
        ),
        "forbidden_character": (
            {
                "type": "forbidden_character_and_required_token",
                "forbidden": "?",
                "required_token": "V001",
            },
            "verified V001 measurement",
            "verified V001?",
        ),
    }

    families = {}
    for family, (contract, good, mutant) in fixtures.items():
        good_passed = _check_instruction_contract_independent(
            good,
            contract,
        )
        mutant_passed = _check_instruction_contract_independent(
            mutant,
            contract,
        )
        families[family] = {
            "good_passed": good_passed,
            "mutant_rejected": not mutant_passed,
        }

    return {
        "status": (
            "QUALIFIED"
            if all(
                item["good_passed"] and item["mutant_rejected"]
                for item in families.values()
            )
            else "UNQUALIFIED_COVERAGE_GAP"
        ),
        "families": families,
    }
