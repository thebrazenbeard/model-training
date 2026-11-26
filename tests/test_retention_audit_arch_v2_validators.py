from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from successor.experiments.retention_audit_arch_v2_validators import (
    validate_row_mechanically,
)


ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "successor/evaluation/v10_qwen35/retention_candidate_v1.jsonl"
V5_PACKET = ROOT / "successor/evaluation/v10_qwen35/retention_family_audit_v5.packet.jsonl"


def _jsonl_row(path: Path, case_id: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["case_id"] == case_id:
            return row
    raise AssertionError(f"missing row: {case_id}")


def _candidate_row(case_id: str) -> dict:
    return _jsonl_row(CANDIDATE, case_id)


def _source_evidence(case_id: str) -> dict:
    return _jsonl_row(V5_PACKET, case_id)["source_evidence"]


def test_direct_source_lookup_validates_montenegro_code() -> None:
    result = validate_row_mechanically(
        _candidate_row("ret-kf-me-code_from_name"),
        source_evidence=_source_evidence("ret-kf-me-code_from_name"),
    )

    assert result["case_id"] == "ret-kf-me-code_from_name"
    assert result["validator_class"] == "direct_source_lookup"
    assert result["status"] == "MECHANICAL_VALID"
    assert result["defects"] == []
    assert result["mechanical_facts"]["derived_answer"] == "ME"


def test_direct_source_lookup_rejects_wrong_grader_answer() -> None:
    row = deepcopy(_candidate_row("ret-kf-me-code_from_name"))
    row["grader_contract"]["answer_key"] = "MT"

    result = validate_row_mechanically(
        row,
        source_evidence=_source_evidence("ret-kf-me-code_from_name"),
    )

    assert result["status"] == "BANK_DEFECT"
    assert result["defects"] == ["grader_answer_mismatch"]
    assert result["mechanical_facts"]["derived_answer"] == "ME"
    assert result["mechanical_facts"]["grader_answer"] == "MT"


def test_source_arithmetic_recomputes_region_sum() -> None:
    result = validate_row_mechanically(
        _candidate_row("ret-math-eg-uy-region_sum"),
        source_evidence=_source_evidence("ret-math-eg-uy-region_sum"),
    )

    assert result["validator_class"] == "source_arithmetic"
    assert result["status"] == "MECHANICAL_VALID"
    assert result["mechanical_facts"]["derived_answer"] == "47"


def test_evidence_calibration_mode2_derives_not_enough_info() -> None:
    result = validate_row_mechanically(
        _candidate_row("ret-cal-bd-2"),
        source_evidence=_source_evidence("ret-cal-bd-2"),
    )

    assert result["validator_class"] == "evidence_calibration"
    assert result["status"] == "MECHANICAL_VALID"
    assert result["mechanical_facts"]["derived_answer"] == "NOT_ENOUGH_INFO"


def test_source_hash_mismatch_is_binding_defect() -> None:
    evidence = deepcopy(_source_evidence("ret-kf-me-code_from_name"))
    evidence["code"] = "MT"

    result = validate_row_mechanically(
        _candidate_row("ret-kf-me-code_from_name"),
        source_evidence=evidence,
    )

    assert result["status"] == "BINDING_DEFECT"
    assert result["defects"] == ["source_hash_mismatch"]


def test_generated_code_contract_is_bound_to_prompt_and_grader() -> None:
    result = validate_row_mechanically(
        _candidate_row("ret-code-chunk_list-037"),
        source_evidence=_source_evidence("ret-code-chunk_list-037"),
    )

    assert result["validator_class"] == "generated_code"
    assert result["status"] == "MECHANICAL_VALID"
    assert result["mechanical_facts"]["function"] == "solve_chunk_list_037"
    assert result["mechanical_facts"]["family"] == "chunk_list"


def test_generated_instruction_contract_is_bound_to_prompt_and_grader() -> None:
    result = validate_row_mechanically(
        _candidate_row("ret-if-forbidden_character-043"),
        source_evidence=_source_evidence("ret-if-forbidden_character-043"),
    )

    assert result["validator_class"] == "generated_instruction"
    assert result["status"] == "MECHANICAL_VALID"
    assert result["mechanical_facts"]["constraint_type"] == (
        "forbidden_character_and_required_token"
    )


def test_structured_extraction_derives_exact_compact_json() -> None:
    result = validate_row_mechanically(
        _candidate_row("ret-extract-me"),
        source_evidence=_source_evidence("ret-extract-me"),
    )

    assert result["validator_class"] == "structured_extraction"
    assert result["status"] == "MECHANICAL_VALID"
    assert result["mechanical_facts"]["derived_answer"] == (
        '{"code":"ME","name":"Montenegro","regions":26,"settlements":1357}'
    )


import successor.experiments.retention_audit_arch_v2_validators as arch_v2


def test_source_resolver_reconstructs_generated_code_record() -> None:
    actual = arch_v2.resolve_source_evidence(
        _candidate_row("ret-code-chunk_list-037")
    )
    assert actual == _source_evidence("ret-code-chunk_list-037")


def test_source_resolver_reconstructs_generated_instruction_record() -> None:
    actual = arch_v2.resolve_source_evidence(
        _candidate_row("ret-if-forbidden_character-043")
    )
    assert actual == _source_evidence("ret-if-forbidden_character-043")


def test_source_resolver_reads_location_record_from_manifest() -> None:
    evidence = _source_evidence("ret-kf-me-code_from_name")
    actual = arch_v2.resolve_source_evidence(
        _candidate_row("ret-kf-me-code_from_name"),
        location_manifest={"countries": [evidence]},
    )
    assert actual == evidence


def test_candidate_validator_routes_multiple_generated_rows() -> None:
    rows = [
        _candidate_row("ret-code-chunk_list-037"),
        _candidate_row("ret-if-forbidden_character-043"),
    ]

    result = arch_v2.validate_candidate_mechanically(rows)

    assert result["status"] == "MECHANICAL_VALIDATED"
    assert result["reviewed"] == 2
    assert result["status_counts"] == {"MECHANICAL_VALID": 2}
    assert [row["case_id"] for row in result["rows"]] == [
        "ret-code-chunk_list-037",
        "ret-if-forbidden_character-043",
    ]


def test_pinned_location_manifest_rejects_wrong_bytes() -> None:
    try:
        arch_v2.load_pinned_location_manifest(
            fetch_bytes=lambda _: b"{}",
        )
    except arch_v2.SourceBindingError as exc:
        assert "location manifest hash mismatch" in str(exc)
    else:
        raise AssertionError("wrong manifest bytes were accepted")


def test_direct_lookup_detects_prompt_source_mismatch() -> None:
    row = deepcopy(_candidate_row("ret-kf-me-code_from_name"))
    row["prompt"] = row["prompt"].replace("Montenegro", "Malta")

    result = validate_row_mechanically(
        row,
        source_evidence=_source_evidence("ret-kf-me-code_from_name"),
    )

    assert result["status"] == "BANK_DEFECT"
    assert "prompt_source_mismatch" in result["defects"]


def test_source_arithmetic_detects_prompt_operand_mismatch() -> None:
    row = deepcopy(_candidate_row("ret-math-eg-uy-region_sum"))
    row["prompt"] = row["prompt"].replace("28 regions", "999 regions")

    result = validate_row_mechanically(
        row,
        source_evidence=_source_evidence("ret-math-eg-uy-region_sum"),
    )

    assert result["status"] == "BANK_DEFECT"
    assert "prompt_source_mismatch" in result["defects"]


def test_calibration_detects_prompt_evidence_source_mismatch() -> None:
    row = deepcopy(_candidate_row("ret-cal-bd-2"))
    prompt_lines = row["prompt"].splitlines()
    prompt_lines[1] = prompt_lines[1].replace('"code":"BD"', '"code":"ZZ"')
    row["prompt"] = "\n".join(prompt_lines)

    result = validate_row_mechanically(
        row,
        source_evidence=_source_evidence("ret-cal-bd-2"),
    )

    assert result["status"] == "BANK_DEFECT"
    assert "prompt_source_mismatch" in result["defects"]
