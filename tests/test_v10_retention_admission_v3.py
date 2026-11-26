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
    validate_retention_admission_v3,
)


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
        "schema": "V10_RETENTION_FAMILY_AUDIT_PACKET_V1",
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "sample_rows": len(packet),
        "family_count": len({row["family_id"] for row in rows}),
        "per_family": 1,
    }
    audit_rows = [
        {
            "case_id": row["case_id"],
            "category": row["category"],
            "verdict": "ADMIT",
            "review": {
                "verdict": "ADMIT",
                "prompt_well_posed": True,
                "grader_matches_prompt": True,
                "source_evidence_sufficient": True,
                "issue_code": "NONE",
                "reason": "No defect.",
                "confidence": "high",
            },
        }
        for row in packet
    ]
    review_output_sha = "c" * 64
    audit_receipt = {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "review_output_sha256": review_output_sha,
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
    return validate_retention_admission_v3(
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
        expected_allocation=TEST_ALLOCATION,
        expected_per_family=1,
    )


def test_v3_admits_when_all_preregistered_evidence_passes() -> None:
    result = _validate(_fixture())
    assert result["status"] == "RETENTION_ADMITTED_V3"
    assert result["reasons"] == []
    assert result["reviewed_family_count"] > 0


def test_v3_holds_on_any_family_audit_reject() -> None:
    parts = list(_fixture())
    parts[5][0]["verdict"] = "REJECT"
    parts[5][0]["review"]["verdict"] = "REJECT"
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any("audit_non_admit" in reason for reason in result["reasons"])


def test_v3_holds_if_deterministic_family_selection_is_changed() -> None:
    parts = list(_fixture())
    parts[2] = parts[2][1:]
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any("packet_case_set_mismatch" in reason for reason in result["reasons"])


def test_v3_holds_if_semantic_binding_is_not_pass() -> None:
    parts = list(_fixture())
    parts[8] = {
        "status": "HOLD",
        "target_sha256": parts[1],
        "reasons": ["target_sha256_mismatch"],
    }
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert "semantic_binding_status:HOLD" in result["reasons"]
