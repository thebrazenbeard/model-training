from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import successor.experiments.verify_v10_retention_admission_v3 as admission_files


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = (
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    candidate = tmp_path / "candidate.jsonl"
    packet = tmp_path / "packet.jsonl"
    audit = tmp_path / "audit.jsonl"
    protocol = tmp_path / "protocol.json"
    semantic = tmp_path / "semantic.json"
    exclusion = tmp_path / "exclusion.txt"
    policy = tmp_path / "policy.json"
    packet_manifest = tmp_path / "packet.manifest.json"
    audit_receipt = tmp_path / "audit.receipt.json"
    evidence = tmp_path / "evidence.json"
    candidate_manifest = tmp_path / "candidate.manifest.json"

    candidate_rows = [{
        "case_id": "c1",
        "lane": "retention",
        "category": "knowledge_factuality",
        "family_id": "f1",
        "prompt": "Question?",
        "source_id": "source",
        "source_revision": "1",
        "source_terms": "CC0",
        "source_hash": "a" * 64,
        "generation_method": "method",
        "generation_actor_id": "generator",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "g",
            "grader_version": "1",
            "answer_key_digest": "b" * 64,
        },
        "review_receipt": {
            "verdict": "PENDING_INDEPENDENT_REVIEW",
            "subject_digest": "c" * 64,
        },
        "source_audit_receipt": {"verdict": "SOURCE_CONTRACT_VALID"},
    }]
    candidate_sha = _write_jsonl(candidate, candidate_rows)
    packet_rows = [dict(candidate_rows[0])]
    packet_sha = _write_jsonl(packet, packet_rows)
    audit_rows = [{
        "case_id": "c1",
        "family_id": "f1",
        "category": "knowledge_factuality",
        "verdict": "ADMIT",
        "review": {
            "verdict": "ADMIT",
            "prompt_well_posed": True,
            "grader_matches_prompt": True,
            "source_evidence_sufficient": True,
        },
    }]
    audit_sha = _write_jsonl(audit, audit_rows)

    _write_json(protocol, {
        "schema": "protocol-v3",
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
    })
    protocol_sha = _sha_file(protocol)
    exclusion.write_bytes(b"")
    exclusion_sha = _sha_file(exclusion)
    _write_json(policy, {
        "schema": "V10_QWEN35_RETENTION_ADMISSION_V3",
        "status": "PREREGISTERED_BEFORE_FAMILY_AUDIT_RESULT",
        "subject": {
            "expected_rows": 1,
            "expected_candidate_sha256": candidate_sha,
        },
    })
    policy_sha = _sha_file(policy)
    _write_json(semantic, {
        "status": "PASS",
        "target_sha256": candidate_sha,
        "threshold": 0.9,
        "model_archive_sha256": "d" * 64,
        "registry": {"registry_sha256": exclusion_sha},
        "reasons": [],
    })
    semantic_sha = _sha_file(semantic)
    _write_json(packet_manifest, {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "sample_rows": 1,
        "family_count": 1,
        "per_family": 1,
    })
    packet_manifest_sha = _sha_file(packet_manifest)
    _write_json(audit_receipt, {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "protocol_sha256": protocol_sha,
        "review_output_sha256": audit_sha,
        "sample_rows": 1,
        "family_count": 1,
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
        "summary": {"status": "PASS"},
    })

    _write_json(candidate_manifest, {
        "schema": "V10_QWEN35_RETENTION_CANDIDATE_MANIFEST_V1",
        "case_count": 1,
        "data_sha256": candidate_sha,
        "category_counts": {"knowledge_factuality": 1},
    })

    _write_json(evidence, {
        "schema": "V10_QWEN35_RETENTION_ADMISSION_EVIDENCE_BINDING_V1",
        "status": "FROZEN_AFTER_PROTOCOL_START_BEFORE_AUDIT_COMPLETION",
        "admission_policy": {
            "file_sha256": policy_sha,
        },
        "candidate": {
            "file_sha256": candidate_sha,
            "expected_rows": 1,
        },
        "family_audit_packet": {
            "file_sha256": packet_sha,
            "manifest_file_sha256": packet_manifest_sha,
            "rows": 1,
            "families": 1,
            "cases_per_family": 1,
        },
        "family_audit_protocol": {
            "file_sha256": protocol_sha,
            "required_reviewer": {
                "provider": "OLLAMA_LOCAL",
                "model": "ministral-3:14b",
                "model_blob_sha256": (
                    "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
                ),
                "runtime": "ollama version is 0.34.2",
                "temperature": 0,
                "seed": 20261001,
            },
        },
        "semantic_screen": {
            "file_sha256": semantic_sha,
            "required_status": "PASS",
            "target_sha256": candidate_sha,
            "threshold": 0.9,
            "model_archive_sha256": "d" * 64,
            "exclusion_registry_sha256": exclusion_sha,
        },
        "exclusion_registry": {
            "file_sha256": exclusion_sha,
        },
        "final_audit_receipt_requirements": {
            "candidate_sha256": candidate_sha,
            "packet_sha256": packet_sha,
            "protocol_sha256": protocol_sha,
            "sample_rows": 1,
            "family_count": 1,
            "cases_per_family": 1,
            "summary_status_required": "PASS",
            "all_case_verdict_required": "ADMIT",
            "review_output_sha256_must_match_file": True,
        },
    })

    return {
        "candidate": candidate,
        "candidate_manifest": candidate_manifest,
        "packet": packet,
        "packet_manifest": packet_manifest,
        "audit": audit,
        "audit_receipt": audit_receipt,
        "protocol": protocol,
        "semantic": semantic,
        "exclusion": exclusion,
        "policy": policy,
        "evidence": evidence,
    }


def _verify(paths: dict[str, Path]):
    return admission_files.verify_retention_admission_files(
        candidate_path=paths["candidate"],
        candidate_manifest_path=paths["candidate_manifest"],
        exclusion_path=paths["exclusion"],
        packet_path=paths["packet"],
        packet_manifest_path=paths["packet_manifest"],
        protocol_path=paths["protocol"],
        audit_output_path=paths["audit"],
        audit_receipt_path=paths["audit_receipt"],
        semantic_path=paths["semantic"],
        admission_policy_path=paths["policy"],
        evidence_binding_path=paths["evidence"],
    )


def test_file_binding_passes_exact_subjects_before_core_validator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        admission_files,
        "RETENTION_ALLOCATION",
        {"knowledge_factuality": 1},
    )
    monkeypatch.setattr(
        admission_files,
        "validate_retention_admission_v3",
        lambda **kwargs: {
            "schema": "V10_QWEN35_RETENTION_ADMISSION_CHECK_V3",
            "status": "RETENTION_ADMITTED_V3",
            "reasons": [],
        },
    )
    result = _verify(paths)
    assert result["status"] == "RETENTION_ADMITTED_V3"
    assert result["binding_reasons"] == []


def test_protocol_substitution_fails_before_core_validator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    called = False

    def core(**kwargs):
        nonlocal called
        called = True
        return {"status": "RETENTION_ADMITTED_V3", "reasons": []}

    monkeypatch.setattr(admission_files, "validate_retention_admission_v3", core)
    paths["protocol"].write_text('{"schema":"substituted"}\n', encoding="utf-8")
    result = _verify(paths)
    assert result["status"] == "HOLD"
    assert "protocol_file_sha256_mismatch" in result["binding_reasons"]
    assert called is False


def test_audit_receipt_must_bind_exact_protocol(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["audit_receipt"].read_text(encoding="utf-8"))
    receipt["protocol_sha256"] = "f" * 64
    _write_json(paths["audit_receipt"], receipt)
    result = _verify(paths)
    assert result["status"] == "HOLD"
    assert "audit_receipt_protocol_sha256_mismatch" in result["binding_reasons"]


def test_semantic_receipt_substitution_fails_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["threshold"] = 0.89
    _write_json(paths["semantic"], semantic)
    result = _verify(paths)
    assert result["status"] == "HOLD"
    assert "semantic_file_sha256_mismatch" in result["binding_reasons"]
