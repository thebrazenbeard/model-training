from __future__ import annotations

import json

from successor.experiments.build_retention_audit_arch_v2_packet import (
    packet_bytes,
)
from successor.experiments.retention_audit_arch_v3_reviewer import (
    build_reviewer_qualification_controls_v3,
)
from successor.experiments.run_retention_audit_arch_v3_live import (
    run_live_review_v3,
)


IDENTITY = {
    "provider": "OLLAMA_LOCAL",
    "model": "ministral-3:14b",
    "model_blob_sha256": (
        "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
    ),
    "runtime": "ollama version is 0.34.2",
    "temperature": 0,
    "seed": 20261002,
    "url": "http://127.0.0.1:11434/api/generate",
    "format": "json",
    "stream": False,
}


def _packet() -> list[dict]:
    return [
        {
            "case_id": f"case-{index}",
            "category": "test",
            "family_id": "family-a",
            "prompt": f"Prompt {index}",
            "source_id": f"source:{index}",
            "source_revision": "TEST",
            "source_terms": "TEST",
            "source_hash": "0" * 64,
            "generation_method": "test",
            "grader_contract": {
                "kind": "deterministic",
                "grader_id": "test",
                "grader_version": "1",
            },
            "source_evidence": {"fact": index},
        }
        for index in range(5)
    ]


def _manifest(packet: list[dict]) -> dict:
    import hashlib

    return {
        "sample_rows": len(packet),
        "family_count": 1,
        "per_family": 5,
        "sample_case_ids": [row["case_id"] for row in packet],
        "packet_sha256": hashlib.sha256(packet_bytes(packet)).hexdigest(),
        "disjoint_from_all_predecessors": True,
    }


def _qualification_response(batch: list[dict], *, contradiction: bool = False) -> str:
    rows = []
    for index, control in enumerate(batch):
        witness = (
            "concrete semantic witness"
            if control["expected_defect"]
            else "NONE"
        )
        if contradiction and index == 0 and control["expected_defect"]:
            witness = None
        rows.append({
            "case_id": control["case_id"],
            "observed_defect": control["expected_defect"],
            "defect_class": control["defect_class"],
            "witness": witness,
            "reason": "qualification",
        })
    return json.dumps({"reviews": rows})


def _semantic_response(packet: list[dict]) -> str:
    return json.dumps({
        "reviews": [
            {
                "case_id": row["case_id"],
                "observation": "NO_SEMANTIC_DEFECT_FOUND",
                "derived_answer": None,
                "witness": None,
                "reason": "clear",
                "confidence": "high",
            }
            for row in packet
        ]
    })


def test_v3_live_runner_qualifies_then_reviews_packet() -> None:
    controls = build_reviewer_qualification_controls_v3()
    packet = _packet()
    calls = []

    def call_text(prompt: str) -> str:
        call_index = len(calls)
        calls.append(prompt)
        if call_index < 12:
            batch = controls[call_index * 4:(call_index + 1) * 4]
            return _qualification_response(batch)
        return _semantic_response(packet)

    result = run_live_review_v3(
        packet,
        _manifest(packet),
        call_text=call_text,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=4,
        semantic_batch_size=5,
        max_attempts=3,
        expected_reviewer_identity=dict(IDENTITY),
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_QUALIFIED"
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["semantic"]["reviewed"] == 5
    assert len(calls) == 13


def test_v3_live_runner_does_not_review_packet_after_qualification_contradiction() -> None:
    controls = build_reviewer_qualification_controls_v3()
    packet = _packet()
    calls = []

    def call_text(prompt: str) -> str:
        call_index = len(calls)
        calls.append(prompt)
        batch = controls[call_index * 4:(call_index + 1) * 4]
        return _qualification_response(
            batch,
            contradiction=(call_index == 0),
        )

    result = run_live_review_v3(
        packet,
        _manifest(packet),
        call_text=call_text,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=4,
        semantic_batch_size=5,
        max_attempts=3,
        expected_reviewer_identity=dict(IDENTITY),
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_UNQUALIFIED"
    assert result["reviewer_qualification"]["structured_contradictions"]
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_HOLD"
    assert result["semantic"]["reviewed"] == 0
    assert result["semantic"]["reasons"] == ["reviewer_unqualified"]
    assert len(calls) == 12
