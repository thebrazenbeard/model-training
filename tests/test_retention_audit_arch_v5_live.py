from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from successor.experiments.build_retention_audit_arch_v2_packet import (
    packet_bytes,
)
from successor.experiments.retention_audit_arch_v5_reviewer import (
    build_reviewer_qualification_controls_v5,
)
from successor.experiments.run_retention_audit_arch_v4_live import (
    exclusive_execution_lock,
)
from successor.experiments.run_retention_audit_arch_v5_live import (
    run_live_review_v5,
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
    "format": "DYNAMIC_JSON_SCHEMA",
    "stream": False,
    "structured_output": "JSON_SCHEMA",
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
    return {
        "sample_rows": len(packet),
        "family_count": 1,
        "per_family": 5,
        "sample_case_ids": [row["case_id"] for row in packet],
        "packet_sha256": hashlib.sha256(packet_bytes(packet)).hexdigest(),
        "disjoint_from_all_predecessors": True,
    }


def _qualification_response(batch: list[dict]) -> str:
    return json.dumps({
        "reviews": [
            {
                "case_id": control["case_id"],
                "observed_defect": control["expected_defect"],
                "defect_class": control["defect_class"],
                "witness": (
                    {"example": "concrete witness"}
                    if control["expected_defect"]
                    else "NONE"
                ),
                "reason": "qualified",
            }
            for control in batch
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


def test_v5_live_runner_qualifies_then_reviews_under_exclusive_lock(
    tmp_path: Path,
) -> None:
    controls = build_reviewer_qualification_controls_v5()
    packet = _packet()
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        index = len(calls)
        calls.append((prompt, schema))
        if index < 12:
            batch = controls[index * 4:(index + 1) * 4]
            return _qualification_response(batch)
        return _semantic_response(packet)

    result = run_live_review_v5(
        packet,
        _manifest(packet),
        call_structured=call_structured,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=4,
        semantic_batch_size=5,
        max_attempts=3,
        expected_reviewer_identity=dict(IDENTITY),
        lock_path=tmp_path / "v5.lock",
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_QUALIFIED"
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["semantic"]["reviewed"] == 5
    assert len(calls) == 13
    assert all(isinstance(schema, dict) for _, schema in calls)
    assert not (tmp_path / "v5.lock").exists()


def test_v5_duplicate_runner_fails_before_structured_call(
    tmp_path: Path,
) -> None:
    packet = _packet()
    lock = tmp_path / "v5.lock"
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        calls.append((prompt, schema))
        raise AssertionError("structured reviewer must not be called")

    with exclusive_execution_lock(lock):
        with pytest.raises(RuntimeError, match="execution lock exists"):
            run_live_review_v5(
                packet,
                _manifest(packet),
                call_structured=call_structured,
                identity_fn=lambda: dict(IDENTITY),
                qualification_batch_size=4,
                semantic_batch_size=5,
                max_attempts=3,
                expected_reviewer_identity=dict(IDENTITY),
                lock_path=lock,
            )

    assert calls == []


def test_v5_unqualified_reviewer_never_reviews_semantic_packet(
    tmp_path: Path,
) -> None:
    packet = _packet()
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        calls.append((prompt, schema))
        return '{"reviews": []}'

    result = run_live_review_v5(
        packet,
        _manifest(packet),
        call_structured=call_structured,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=4,
        semantic_batch_size=5,
        max_attempts=3,
        expected_reviewer_identity=dict(IDENTITY),
        lock_path=tmp_path / "v5.lock",
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_UNQUALIFIED"
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_HOLD"
    assert result["semantic"]["reviewed"] == 0
    assert result["semantic"]["reasons"] == ["reviewer_unqualified"]
    assert len(calls) == 3


def test_v5_reviewer_identity_mismatch_fails_before_call(
    tmp_path: Path,
) -> None:
    packet = _packet()
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        calls.append((prompt, schema))
        raise AssertionError("reviewer must not be called")

    wrong = dict(IDENTITY)
    wrong["model"] = "wrong"

    with pytest.raises(ValueError, match="reviewer identity mismatch"):
        run_live_review_v5(
            packet,
            _manifest(packet),
            call_structured=call_structured,
            identity_fn=lambda: dict(IDENTITY),
            qualification_batch_size=4,
            semantic_batch_size=5,
            max_attempts=3,
            expected_reviewer_identity=wrong,
            lock_path=tmp_path / "v5.lock",
        )

    assert calls == []
    assert not (tmp_path / "v5.lock").exists()
