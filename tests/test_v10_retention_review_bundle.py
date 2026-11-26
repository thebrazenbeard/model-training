from __future__ import annotations

import hashlib
import json

import pytest

from successor.experiments.build_v10_qwen35_retention_candidate import (
    build_candidate_rows,
)
from successor.experiments.prepare_v10_qwen35_retention_review import (
    apply_review_receipts,
    build_model_review_receipts,
    build_review_records,
    write_review_bundle,
    write_reviewed_files,
)
from successor.experiments.v10_qwen35_bank import review_subject_digest


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


def _receipt_for(row: dict, *, reviewer_id: str = "reviewer-1") -> dict:
    return {
        "case_id": row["case_id"],
        "review_receipt": {
            "verdict": "ADMIT",
            "reviewer_id": reviewer_id,
            "reviewer_class": "EXTERNAL_MODEL_REVIEW",
            "provider": "test-provider",
            "runtime": "test-runtime",
            "independent_from_generation": True,
            "artifact_digest": row["review_receipt"]["artifact_digest"],
            "subject_digest": review_subject_digest(row),
        },
    }


def test_review_records_reconstruct_source_evidence() -> None:
    countries = _fixture_countries()
    rows = build_candidate_rows(countries, scale_for_test=True)
    records = build_review_records(rows, countries)
    assert len(records) == len(rows)
    assert all(
        record["source_evidence_sha256"] == record["review_subject"]["source_hash"]
        for record in records
    )


def test_apply_requires_receipt_for_every_case() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    receipts = [_receipt_for(row) for row in rows[:-1]]
    with pytest.raises(RuntimeError, match="case set mismatch"):
        apply_review_receipts(rows=rows, receipts=receipts)


def test_apply_rejects_transplanted_subject_digest() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    receipts = [_receipt_for(row) for row in rows]
    receipts[0]["review_receipt"]["subject_digest"] = receipts[1]["review_receipt"][
        "subject_digest"
    ]
    with pytest.raises(RuntimeError, match="invalid independent review receipt"):
        apply_review_receipts(rows=rows, receipts=receipts)


def test_apply_rejects_wrong_artifact_binding() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    receipts = [_receipt_for(row) for row in rows]
    receipts[0]["review_receipt"]["artifact_digest"] = hashlib.sha256(
        b"wrong"
    ).hexdigest()
    with pytest.raises(RuntimeError, match="artifact digest mismatch"):
        apply_review_receipts(rows=rows, receipts=receipts)


def test_apply_accepts_all_valid_independent_receipts() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    receipts = [_receipt_for(row) for row in rows]
    reviewed, manifest = apply_review_receipts(rows=rows, receipts=receipts)
    assert manifest["status"] == "ALL_CASE_RECEIPTS_VALID"
    assert manifest["case_count"] == len(rows)
    assert len(reviewed) == len(rows)
    assert all(row["review_receipt"]["verdict"] == "ADMIT" for row in reviewed)


def test_review_bundle_hashes_exact_persisted_batch_bytes(tmp_path) -> None:
    countries = _fixture_countries()
    rows = build_candidate_rows(countries, scale_for_test=True)
    manifest = write_review_bundle(
        rows=rows,
        countries=countries,
        output_dir=tmp_path,
        batch_size=50,
    )
    first = manifest["batches"][0]
    batch_path = tmp_path / first["filename"]
    assert hashlib.sha256(batch_path.read_bytes()).hexdigest() == first["sha256"]
    assert b"\r\n" not in batch_path.read_bytes()


def test_reviewed_writer_hashes_exact_persisted_bytes(tmp_path) -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    receipts = [_receipt_for(row) for row in rows]
    reviewed, manifest = apply_review_receipts(rows=rows, receipts=receipts)
    output = tmp_path / "reviewed.jsonl"
    manifest_path = tmp_path / "reviewed.manifest.json"
    written = write_reviewed_files(reviewed, manifest, output, manifest_path)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == written["reviewed_sha256"]
    assert b"\r\n" not in output.read_bytes()


def _audit_row(row: dict, verdict: str = "ADMIT") -> dict:
    return {
        "case_id": row["case_id"],
        "category": row["category"],
        "verdict": verdict,
        "review_prompt_sha256": hashlib.sha256(
            ("prompt:" + row["case_id"]).encode()
        ).hexdigest(),
        "review": {
            "verdict": verdict,
            "prompt_well_posed": verdict == "ADMIT",
            "grader_matches_prompt": verdict == "ADMIT",
            "source_evidence_sufficient": verdict == "ADMIT",
            "issue_code": "NONE" if verdict == "ADMIT" else "DEFECT",
            "reason": "No material defect." if verdict == "ADMIT" else "Defect.",
            "confidence": "high",
        },
        "runtime": {},
    }


def _audit_receipt(rows: list[dict]) -> dict:
    return {
        "schema": "V10_RETENTION_INDEPENDENT_MODEL_AUDIT_V1",
        "candidate_sha256": hashlib.sha256(
            (
                "\n".join(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    for row in rows
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest(),
        "packet_sha256": "a" * 64,
        "reviewer": {
            "provider": "OLLAMA_LOCAL",
            "model": "ministral-3:14b",
            "model_blob_sha256": "b" * 64,
            "runtime": "ollama version is 0.34.2",
            "temperature": 0,
            "seed": 20261001,
        },
    }


def test_model_audit_converts_to_bound_external_review_receipts() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    audit_rows = [_audit_row(row) for row in rows]
    receipts = build_model_review_receipts(
        rows=rows,
        audit_rows=audit_rows,
        audit_receipt=_audit_receipt(rows),
    )
    assert len(receipts) == len(rows)
    first = receipts[0]["review_receipt"]
    assert first["verdict"] == "ADMIT"
    assert first["reviewer_class"] == "EXTERNAL_MODEL_REVIEW"
    assert first["provider"] == "OLLAMA_LOCAL"
    assert len(first["audit_record_sha256"]) == 64


def test_model_audit_reject_prevents_receipt_conversion() -> None:
    rows = build_candidate_rows(_fixture_countries(), scale_for_test=True)
    audit_rows = [_audit_row(row) for row in rows]
    audit_rows[0] = _audit_row(rows[0], verdict="REJECT")
    with pytest.raises(RuntimeError, match="non-ADMIT"):
        build_model_review_receipts(
            rows=rows,
            audit_rows=audit_rows,
            audit_receipt=_audit_receipt(rows),
        )
