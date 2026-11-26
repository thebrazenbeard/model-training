from __future__ import annotations

import hashlib
import json

from successor.experiments.build_v10_qwen35_retention_candidate import (
    TEST_ALLOCATION,
    build_candidate_rows,
)
from successor.experiments.build_v10_retention_family_audit_v5 import (
    select_disjoint_family_audit_sample,
)
from successor.experiments.retention_admission_v3 import (
    candidate_rows_sha256,
)
from successor.experiments.retention_admission_v5 import (
    validate_retention_admission_v5,
)


PROTOCOL_SHA = "5" * 64
PREDECESSOR_PACKET_SHA = "4" * 64


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
        for i in range(60)
    ]


def _fixture():
    rows = build_candidate_rows(_countries(), scale_for_test=True)
    candidate_sha = candidate_rows_sha256(rows)
    predecessor = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["family_id"], []).append(row)
    for family in sorted(grouped):
        family_rows = sorted(grouped[family], key=lambda r: r["case_id"])
        if len(family_rows) > 1:
            predecessor.append(family_rows[0])
    excluded = {row["case_id"] for row in predecessor}
    packet = select_disjoint_family_audit_sample(
        rows,
        excluded_case_ids=excluded,
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
        "predecessor_packet_sha256": PREDECESSOR_PACKET_SHA,
        "sample_rows": len(packet),
        "family_count": len({row["family_id"] for row in rows}),
        "per_family": 1,
    }
    audit_rows = []
    for row in packet:
        review = {
            "prompt_well_posed": True,
            "grader_matches_prompt": True,
            "evidence_policy_supports_expected_answer": True,
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
    family_count = len({row["family_id"] for row in rows})
    audit_receipt = {
        "candidate_sha256": candidate_sha,
        "predecessor_packet_sha256": PREDECESSOR_PACKET_SHA,
        "packet_sha256": packet_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "review_output_sha256": review_output_sha,
        "sample_rows": len(packet),
        "family_count": family_count,
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
            "families_reviewed": family_count,
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
        excluded,
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
        excluded,
    ) = parts
    return validate_retention_admission_v5(
        rows=rows,
        exclusion_hashes=set(),
        predecessor_case_ids=excluded,
        candidate_file_sha256=candidate_sha,
        packet_rows=packet,
        packet_file_sha256=packet_sha,
        packet_manifest=packet_manifest,
        audit_rows=audit_rows,
        review_output_sha256=review_output_sha,
        audit_receipt=audit_receipt,
        semantic_binding=semantic,
        expected_protocol_sha256=PROTOCOL_SHA,
        expected_predecessor_packet_sha256=PREDECESSOR_PACKET_SHA,
        expected_allocation=TEST_ALLOCATION,
        expected_per_family=1,
    )


def test_v5_admits_exact_complete_fresh_audit() -> None:
    result = _validate(_fixture())
    assert result["status"] == "RETENTION_ADMITTED_V5"
    assert result["reasons"] == []


def test_v5_holds_on_any_predecessor_case_reuse() -> None:
    parts = list(_fixture())
    predecessor_id = next(iter(parts[9]))
    candidate_by_id = {row["case_id"]: row for row in parts[0]}
    parts[2][0] = candidate_by_id[predecessor_id]
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any("packet_predecessor_overlap" in r for r in result["reasons"])


def test_v5_holds_on_evidence_policy_false() -> None:
    parts = list(_fixture())
    parts[5][0]["review"]["evidence_policy_supports_expected_answer"] = False
    parts[5][0]["verdict"] = "REJECT"
    parts[7]["summary"]["status"] = "HOLD"
    parts[7]["summary"]["admit"] -= 1
    parts[7]["summary"]["reject"] = 1
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any("audit_non_admit" in r for r in result["reasons"])


def test_v5_holds_if_verdict_disagrees_with_three_booleans() -> None:
    parts = list(_fixture())
    parts[5][0]["review"]["evidence_policy_supports_expected_answer"] = False
    parts[5][0]["verdict"] = "ADMIT"
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert any("audit_derived_verdict_mismatch" in r for r in result["reasons"])


def test_v5_holds_on_protocol_substitution() -> None:
    parts = list(_fixture())
    parts[7]["protocol_sha256"] = "f" * 64
    result = _validate(tuple(parts))
    assert result["status"] == "HOLD"
    assert "audit_protocol_sha256_mismatch" in result["reasons"]
