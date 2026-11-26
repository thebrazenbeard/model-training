from __future__ import annotations


def test_direct_lookup_validator_qualification_passes_good_and_bad_controls() -> None:
    from successor.experiments.retention_audit_arch_v2_qualification import (
        qualify_validators,
    )

    result = qualify_validators()
    direct = result["validators"]["direct_source_lookup"]

    assert direct["status"] == "QUALIFIED"
    assert direct["good_controls"] >= 1
    assert direct["seeded_defects"] >= 1
    assert direct["missed_defects"] == []
    assert direct["false_positives"] == []


def test_all_six_validator_classes_are_mutation_qualified() -> None:
    from successor.experiments.retention_audit_arch_v2_qualification import (
        qualify_validators,
    )

    result = qualify_validators()
    expected = {
        "direct_source_lookup",
        "source_arithmetic",
        "evidence_calibration",
        "generated_code",
        "generated_instruction",
        "structured_extraction",
    }

    assert set(result["validators"]) == expected
    assert result["status"] == "VALIDATORS_QUALIFIED"
    assert all(
        item["status"] == "QUALIFIED"
        for item in result["validators"].values()
    )


def test_code_family_mutations_cover_all_ten_families() -> None:
    from successor.experiments.retention_audit_arch_v2_qualification import (
        qualify_code_family_behavior,
    )

    result = qualify_code_family_behavior()

    assert result["status"] == "QUALIFIED"
    assert set(result["families"]) == {
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
    }
    assert all(item["good_passed"] for item in result["families"].values())
    assert all(item["mutant_killed"] for item in result["families"].values())


def test_instruction_family_mutations_cover_all_five_families() -> None:
    from successor.experiments.retention_audit_arch_v2_qualification import (
        qualify_instruction_family_behavior,
    )

    result = qualify_instruction_family_behavior()

    assert result["status"] == "QUALIFIED"
    assert set(result["families"]) == {
        "token_count",
        "prefix_suffix",
        "line_prefixes",
        "json_keys",
        "forbidden_character",
    }
    assert all(item["good_passed"] for item in result["families"].values())
    assert all(item["mutant_rejected"] for item in result["families"].values())


def test_aggregate_qualification_includes_behavioral_gates() -> None:
    from successor.experiments.retention_audit_arch_v2_qualification import (
        qualify_validators,
    )

    result = qualify_validators()

    assert result["status"] == "VALIDATORS_QUALIFIED"
    assert result["code_behavior"]["status"] == "QUALIFIED"
    assert result["instruction_behavior"]["status"] == "QUALIFIED"


def test_mutation_corpus_covers_required_defect_classes() -> None:
    from successor.experiments.retention_audit_arch_v2_qualification import (
        build_validator_mutation_cases,
    )

    by_validator: dict[str, set[str]] = {}
    for case in build_validator_mutation_cases():
        if case["kind"] != "seeded_defect":
            continue
        by_validator.setdefault(case["validator_class"], set()).add(case["label"])

    assert {
        "wrong_code",
        "wrong_name",
        "stale_region_count",
        "stale_settlement_count",
    } <= by_validator["direct_source_lookup"]

    assert {
        "off_by_one_sum",
        "wrong_operand_value",
        "signed_instead_of_absolute_difference",
    } <= by_validator["source_arithmetic"]

    assert {
        "true_false_swap",
        "false_true_swap",
        "mode2_changed_to_true",
        "prompt_evidence_changed",
    } <= by_validator["evidence_calibration"]

    assert {
        "signature_mismatch",
        "function_mismatch",
        "stored_test_removed",
    } <= by_validator["generated_code"]

    assert {
        "missing_key",
        "wrong_value",
        "extra_key",
    } <= by_validator["structured_extraction"]
