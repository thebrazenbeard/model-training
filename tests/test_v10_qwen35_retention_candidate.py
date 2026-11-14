from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10_qwen35_retention_candidate import (
    LOCATION_CONTENT_FINGERPRINT,
    LOCATION_MANIFEST_SHA256,
    LOCATION_TERMS,
    build_candidate_rows,
    stable_take,
    validate_location_manifest,
    write_candidate_files,
)
from successor.experiments.v10_qwen35_bank import (
    preflight_bank,
    preflight_retention_rows,
    review_subject_digest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture_countries() -> list[dict]:
    return [
        {
            "id": i + 100,
            "code": f"X{i:02d}",
            "name": f"Country {i:02d}",
            "regions": (i % 9) + 1,
            "settlements": 100 + i * 7,
            "source": "wikidata",
            "slug": f"country-{i:02d}",
        }
        for i in range(40)
    ]


def test_stable_take_is_order_invariant() -> None:
    rows = [{"id": str(i)} for i in range(20)]
    a = stable_take(rows, 5, key=lambda r: r["id"], namespace="x")
    b = stable_take(list(reversed(rows)), 5, key=lambda r: r["id"], namespace="x")
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_location_manifest_hash_is_pinned() -> None:
    assert LOCATION_MANIFEST_SHA256 == (
        "bf5a2dc0bb0f1ba15f7149e9fa1184aa632f280a72fe306c154a9e05cfacab29"
    )


def test_location_manifest_requires_cc0_and_frozen_content_version() -> None:
    value = {
        "license": LOCATION_TERMS,
        "s_version": LOCATION_CONTENT_FINGERPRINT,
        "version": LOCATION_CONTENT_FINGERPRINT,
        "source": "https://www.wikidata.org/ (Wikidata truthy dump, CC0)",
        "countries": _fixture_countries() * 6,
    }
    validate_location_manifest(value)
    bad = dict(value)
    bad["license"] = "GPL-3.0"
    try:
        validate_location_manifest(bad)
    except RuntimeError as exc:
        assert "license mismatch" in str(exc)
    else:
        raise AssertionError("wrong data license was accepted")


def test_builder_emits_exact_allocation_without_public_benchmark_imports() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
        assert "MMLU" not in row["source_id"]
        assert "IFEval" not in row["source_id"]
        assert "TruthfulQA" not in row["source_id"]
        assert "MBPP" not in row["source_id"]
        assert row["generation_method"].startswith("novel_deterministic")
    assert counts == {
        "knowledge_factuality": 30,
        "reasoning_math": 30,
        "coding": 25,
        "instruction_following": 25,
        "extraction_structured": 20,
        "truthfulness_factual_calibration": 20,
    }


def test_candidate_rows_pass_objective_structural_preflight() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    result = preflight_retention_rows(
        rows,
        exclusion_hashes=set(),
        expected_allocation={
            "knowledge_factuality": 30,
            "reasoning_math": 30,
            "coding": 25,
            "instruction_following": 25,
            "extraction_structured": 20,
            "truthfulness_factual_calibration": 20,
        },
    )
    assert result["status"] == "STRUCTURE_READY"
    assert not result["reasons"]
    assert result["review_policy"] == "SEE_V10_QWEN35_RETENTION_ADMISSION_V3"
    assert result["admission_review_status"] == "OUT_OF_SCOPE_FOR_STRUCTURE_PREFLIGHT"
    assert "required_independent_review_count" not in result


def test_review_receipt_binds_exact_case_subject() -> None:
    row = build_candidate_rows(_fixture_countries(), scale_for_test=True)[0]
    assert row["review_receipt"]["subject_digest"] == review_subject_digest(row)
    changed = dict(row)
    changed["prompt"] = row["prompt"] + " changed"
    assert row["review_receipt"]["subject_digest"] != review_subject_digest(changed)


def test_candidate_keeps_independent_review_unbound() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    assert all(
        row["review_receipt"]["verdict"] == "PENDING_INDEPENDENT_REVIEW"
        for row in rows
    )
    assert all(
        row["source_audit_receipt"]["independent_review"] is False
        for row in rows
    )
    final_result = preflight_bank(rows, exclusion_hashes=set())
    assert final_result["status"] == "HOLD"
    assert any(
        reason.startswith("invalid_review_receipt:")
        for reason in final_result["reasons"]
    )


def test_generated_coding_graders_are_nonempty() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    coding = [r for r in rows if r["category"] == "coding"]
    assert coding
    assert all(r["grader_contract"]["tests"] for r in coding)


def test_instruction_following_cases_have_machine_contracts() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    cases = [r for r in rows if r["category"] == "instruction_following"]
    assert cases
    assert all(r["grader_contract"]["constraints"] for r in cases)


def test_candidate_writer_hashes_exact_persisted_bytes(tmp_path: Path) -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    manifest = {"schema": "test", "data_sha256": "UNSET"}
    data_path = tmp_path / "candidate.jsonl"
    manifest_path = tmp_path / "manifest.json"
    written = write_candidate_files(rows, manifest, data_path, manifest_path)
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == written["data_sha256"]
    assert b"\r\n" not in data_path.read_bytes()