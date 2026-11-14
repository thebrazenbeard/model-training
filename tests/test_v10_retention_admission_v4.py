from __future__ import annotations

import hashlib
import json

from successor.experiments.build_v10_qwen35_retention_candidate import (
    TEST_ALLOCATION,
    build_candidate_rows,
)
from successor.experiments.build_v10_retention_audit_packet import (
    select_family_audit_sample,
)
from successor.experiments.retention_admission_v3 import (
    candidate_rows_sha256,
)
from successor.experiments.retention_admission_v4 import (
    validate_retention_admission_v4,
)


PROTOCOL_SHA = "4" * 64


def _countries() -> list[dict]:
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


def _fixture():
    rows = build_candidate_rows(_countries(), scale_for_test=True)
    candidate_sha = candidate_rows_sha256(rows)
    packet = select_family_audit_sample(
        rows,
        per_family=1,
        candidate_sha256=candidate_sha,
    )
    packet_sha = hashlib.sha256(
        (
            "\n".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                for row in packet
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    packet_manifest = {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "sample_rows": len(packet),
        "family_count": len({row["family_id"] for row in rows}),
        "per_family": 1,
    }
    audit_rows = []
    for row in packet:
        review = {
            "prompt_well_posed": True,
            "grader_matches_prompt": True,
            "source_evidence_sufficient": True,
            "issue_code": "NONE",
            "reason": "No material defect.",
            "confidence": "high",
        }
        audit_rows.append({
            "case_id": row["case_id"],
            "category": row["category"],
            "family_id": row["family_id"],
            "verdict": "ADMIT",
            "review": review,
            "family_review_prompt_sha256": "a" * 64,
            "protocol_sha256": PROTOCOL_SHA,
        })

    review_output_sha = "c" * 64
    audit_receipt = {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "review_output_sha256": review_output_sha,
        "sample_rows": len(packet),
        "family_count": len({row["family_id"] for row in rows}),
        "cases_per_family": 1,
        "reviewer": {
            "provider": "OLLAMA_LOCAL",
            "model": "ministral-3:14b",
            "model_blob_sha256": (
                "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
            ),
            "runtime": "ollama version is 0.34.2",
            "temperature": 0,
            "seed": 20261001,
        },
        "summary": {
            "status": "PASS",
            "reviewed": len(packet),
            "families_reviewed": len({row["family_id"] for row in rows}),
            "admit": len(packet),
            "reject": 0,
            "reasons": [],
        },
    }
    semantic = {
        "status": "PASS",
        "target_sha256": candidate_sha,
        "reasons": [],
    }
    return (
        rows,
        candidate_sha,
        packet,
        packet_sha,
        packet_manifest,
        audit_rows,
        review_output_sha,
        audit_receipt,
        semantic,
    )


def _validate(parts):
    (
        rows,
        candidate_sha,
        packet,
        packet_sha,
        packet_manifest,
        audit_rows,
        review_output_sha,
        audit_receipt,
        semantic,
    ) = parts
    return validate_retention_admission_v4(
        rows=rows,
        exclusion_hashes=set(),
        candidate_file_sha256=candidate_sha,
        packet_rows=packet,
        packet_file_sha256=packet_sha,
        packet_manifest=packet_manifest,
        audit_rows=audit_rows,
        review_output_sha256=review_output_sha,
        audit_receipt=audit_receipt,
        semantic_binding=semantic,
        expected_protocol_sha256=PROTOCOL_SHA,
        expected_allocation=TEST_ALLOCATION,
        expected_per_family=1,
    )


def test_v4_admits_exact_complete_derived_admit_audit() -> None:
    result = _validate(_fixture())
    assert result["status"] == "RETENTION_ADMITTED_V4"
    assert result["reasons"] == []


def test_v4_holds_on_any_false_boolean_and_derived_reject() -> None:
    parts = list(_fixture())
    parts[5][0]["review"]["grader_matches_prompt"] = False
    parts[5][0]["verdict"] = "REJECT"
    parts[7]["summary"]["status"] = "HOLD"
    parts[7]["summary"]["admit"] -= 1
    parts[7]["summary"]["reject"] = 1
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any("audit_non_admit" in reason for reason in result["reasons"])


def test_v4_holds_if_top_level_verdict_disagrees_with_booleans() -> None:
    parts = list(_fixture())
    parts[5][0]["review"]["source_evidence_sufficient"] = False
    parts[5][0]["verdict"] = "ADMIT"
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any(
        "audit_derived_verdict_mismatch" in reason
        for reason in result["reasons"]
    )


def test_v4_holds_if_review_contains_forbidden_verdict_field() -> None:
    parts = list(_fixture())
    parts[5][0]["review"]["verdict"] = "ADMIT"
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any(
        "audit_review_verdict_field_present" in reason
        for reason in result["reasons"]
    )


def test_v4_holds_on_protocol_substitution() -> None:
    parts = list(_fixture())
    parts[7]["protocol_sha256"] = "f" * 64
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert "audit_protocol_sha256_mismatch" in result["reasons"]


def test_v4_holds_on_incomplete_summary() -> None:
    parts = list(_fixture())
    parts[7]["summary"]["status"] = "HOLD"
    parts[7]["summary"]["reasons"] = ["missing_family_count:1"]
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert "audit_summary_status:HOLD" in result["reasons"]
