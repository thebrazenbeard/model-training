from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from successor.experiments.build_retention_audit_arch_v2_packet import (
    packet_bytes,
)
from successor.experiments.retention_audit_arch_v4_reviewer import (
    build_reviewer_qualification_controls_v4,
)
from successor.experiments.run_retention_audit_arch_v4_live import (
    exclusive_execution_lock,
    run_live_review_v4,
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
                "reason": "qualification",
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


def test_v4_execution_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    lock = tmp_path / "v4.lock"

    with exclusive_execution_lock(lock, metadata={"binding": "abc"}):
        assert lock.is_file()
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["binding"] == "abc"
        with pytest.raises(RuntimeError, match="execution lock exists"):
            with exclusive_execution_lock(lock):
                pass

    assert not lock.exists()


def test_v4_execution_lock_releases_after_exception(tmp_path: Path) -> None:
    lock = tmp_path / "v4.lock"

    with pytest.raises(RuntimeError, match="boom"):
        with exclusive_execution_lock(lock):
            raise RuntimeError("boom")

    assert not lock.exists()


def test_v4_live_runner_qualifies_then_reviews_packet_under_lock(
    tmp_path: Path,
) -> None:
    controls = build_reviewer_qualification_controls_v4()
    packet = _packet()
    calls = []

    def call_text(prompt: str) -> str:
        index = len(calls)
        calls.append(prompt)
        if index < 12:
            batch = controls[index * 4:(index + 1) * 4]
            return _qualification_response(batch)
        return _semantic_response(packet)

    result = run_live_review_v4(
        packet,
        _manifest(packet),
        call_text=call_text,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=4,
        semantic_batch_size=5,
        max_attempts=3,
        expected_reviewer_identity=dict(IDENTITY),
        lock_path=tmp_path / "v4.lock",
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_QUALIFIED"
    assert result["semantic"]["status"] == "SEMANTIC_AUDIT_COMPLETE"
    assert result["semantic"]["reviewed"] == 5
    assert len(calls) == 13
    assert not (tmp_path / "v4.lock").exists()


def test_v4_duplicate_live_runner_fails_before_reviewer_call(
    tmp_path: Path,
) -> None:
    packet = _packet()
    lock = tmp_path / "v4.lock"
    calls = []

    def call_text(prompt: str) -> str:
        calls.append(prompt)
        raise AssertionError("reviewer must not be called")

    with exclusive_execution_lock(lock):
        with pytest.raises(RuntimeError, match="execution lock exists"):
            run_live_review_v4(
                packet,
                _manifest(packet),
                call_text=call_text,
                identity_fn=lambda: dict(IDENTITY),
                qualification_batch_size=4,
                semantic_batch_size=5,
                max_attempts=3,
                expected_reviewer_identity=dict(IDENTITY),
                lock_path=lock,
            )

    assert calls == []


def test_v4_qualification_transport_failure_records_response_hash(
    tmp_path: Path,
) -> None:
    packet = _packet()
    calls = []

    def call_text(prompt: str) -> str:
        calls.append(prompt)
        return '{"reviews":"wrong-shape"}'

    result = run_live_review_v4(
        packet,
        _manifest(packet),
        call_text=call_text,
        identity_fn=lambda: dict(IDENTITY),
        qualification_batch_size=4,
        semantic_batch_size=5,
        max_attempts=3,
        expected_reviewer_identity=dict(IDENTITY),
        lock_path=tmp_path / "v4.lock",
    )

    assert result["reviewer_qualification"]["status"] == "REVIEWER_UNQUALIFIED"
    failures = result["reviewer_qualification"]["transport_failures"]
    assert len(failures) == 3
    assert all(len(item["response_sha256"]) == 64 for item in failures)
    assert result["semantic"]["reviewed"] == 0
    assert len(calls) == 3
