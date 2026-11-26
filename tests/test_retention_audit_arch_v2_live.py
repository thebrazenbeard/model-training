from __future__ import annotations

import json

from successor.experiments.build_retention_audit_arch_v2_packet import (
    packet_bytes,
)
from successor.experiments.retention_audit_arch_v2_reviewer import (
    build_reviewer_qualification_controls,
)
from successor.experiments.run_retention_audit_arch_v2_live import (
    run_live_review,
)


IDENTITY = {
    "provider": "OLLAMA_LOCAL",
    "model": "ministral-3:14b",
    "model_blob_sha256": "b" * 64,
    "runtime": "ollama version is test",
    "temperature": 0,
    "seed": 20261002,
    "url": "http://127.0.0.1:11434/api/generate",
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


def _qualification_response(*, all_correct: bool = True) -> str:
    controls = build_reviewer_qualification_controls()
    return json.dumps({
        "reviews": [
            {
                "case_id": control["case_id"],
                "observed_defect": (
                    control["expected_defect"] if all_correct else False
                ),
                "defect_class": control["defect_class"],
                "witness": (
                    "concrete witness"
                    if all_correct and control["expected_defect"]
                    else None
                ),
                "reason": "qualification",
            }
            for control in controls
        ]
    })


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


def test_live_runner_qualifies_before_semantic_review() -> None:
    packet = _packet()
    responses = iter([
        _qualification_response(),
        _semantic_response(packet),
    ])
    calls = []

    def call_text(prompt: str) -> str:
        calls.append(prompt)
        return next(responses)

    result = run_live_review(
        packet,
        _manifest(packet),
        call_text=call_text,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=48,
        semantic_batch_size=5,
        max_attempts=3,
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_QUALIFIED"
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert len(calls) == 2


def test_live_runner_never_reviews_packet_if_reviewer_unqualified() -> None:
    packet = _packet()
    calls = []

    def call_text(prompt: str) -> str:
        calls.append(prompt)
        return _qualification_response(all_correct=False)

    result = run_live_review(
        packet,
        _manifest(packet),
        call_text=call_text,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=48,
        semantic_batch_size=5,
        max_attempts=3,
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_UNQUALIFIED"
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_HOLD"
    assert result["semantic"]["reasons"] == ["reviewer_unqualified"]
    assert len(calls) == 1


def test_execution_binding_verifier_detects_file_mutation(tmp_path) -> None:
    import hashlib
    import pytest

    from successor.experiments.run_retention_audit_arch_v2_live import (
        verify_execution_binding,
    )

    target = tmp_path / "subject.txt"
    target.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    binding = {
        "artifacts": {
            "subject": {
                "path": "subject.txt",
                "sha256": digest,
            }
        }
    }

    verified = verify_execution_binding(binding, root=tmp_path)
    assert verified["subject"] == digest

    target.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding hash mismatch"):
        verify_execution_binding(binding, root=tmp_path)
