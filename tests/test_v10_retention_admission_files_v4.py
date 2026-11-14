from __future__ import annotations

import hashlib
import json
from pathlib import Path

import successor.experiments.verify_v10_retention_admission_v4 as admission_files


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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    candidate = tmp_path / "candidate.jsonl"
    candidate_manifest = tmp_path / "candidate.manifest.json"
    exclusion = tmp_path / "exclusion.txt"
    packet = tmp_path / "packet.jsonl"
    packet_manifest = tmp_path / "packet.manifest.json"
    protocol = tmp_path / "protocol.json"
    runner = tmp_path / "runner.py"
    semantic = tmp_path / "semantic.json"
    policy = tmp_path / "policy.json"
    binding = tmp_path / "binding.json"
    audit = tmp_path / "audit.jsonl"
    receipt = tmp_path / "audit.receipt.json"

    row = {
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
    }
    candidate_sha = _write_jsonl(candidate, [row])
    _write_json(candidate_manifest, {
        "case_count": 1,
        "data_sha256": candidate_sha,
        "category_counts": {"knowledge_factuality": 1},
    })

    packet_sha = _write_jsonl(packet, [row])
    _write_json(packet_manifest, {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "sample_rows": 1,
        "family_count": 1,
        "per_family": 1,
    })
    packet_manifest_sha = _sha(packet_manifest)

    _write_json(protocol, {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "sample_rows": 1,
        "family_count": 1,
        "cases_per_family": 1,
    })
    protocol_sha = _sha(protocol)
    runner.write_text("# frozen runner\n", encoding="utf-8", newline="\n")
    runner_sha = _sha(runner)

    exclusion.write_bytes(b"")
    exclusion_sha = _sha(exclusion)
    _write_json(semantic, {
        "status": "PASS",
        "target_sha256": candidate_sha,
        "threshold": 0.9,
        "model_archive_sha256": "d" * 64,
        "registry": {"registry_sha256": exclusion_sha},
        "reasons": [],
    })
    semantic_sha = _sha(semantic)

    _write_json(policy, {
        "schema": "V10_QWEN35_RETENTION_ADMISSION_V4",
        "status": "PREREGISTERED_BEFORE_V4_AUDIT_RESULT",
        "subject": {
            "expected_rows": 1,
            "expected_candidate_sha256": candidate_sha,
        },
        "family_audit_v4": {
            "protocol_sha256": protocol_sha,
            "runner_sha256": runner_sha,
            "v3_rows_reusable": False,
        },
    })
    policy_sha = _sha(policy)

    review = {
        "prompt_well_posed": True,
        "grader_matches_prompt": True,
        "source_evidence_sufficient": True,
        "issue_code": "NONE",
        "reason": "No material defect.",
        "confidence": "high",
    }
    audit_sha = _write_jsonl(audit, [{
        "case_id": "c1",
        "family_id": "f1",
        "category": "knowledge_factuality",
        "verdict": "ADMIT",
        "review": review,
        "family_review_prompt_sha256": "e" * 64,
        "protocol_sha256": protocol_sha,
    }])

    reviewer = {
        "provider": "OLLAMA_LOCAL",
        "model": "ministral-3:14b",
        "model_blob_sha256": (
            "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
        ),
        "runtime": "ollama version is 0.34.2",
        "temperature": 0,
        "seed": 20261001,
    }
    _write_json(receipt, {
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "protocol_sha256": protocol_sha,
        "review_output_sha256": audit_sha,
        "sample_rows": 1,
        "family_count": 1,
        "cases_per_family": 1,
        "reviewer": reviewer,
        "summary": {
            "status": "PASS",
            "reviewed": 1,
            "families_reviewed": 1,
            "admit": 1,
            "reject": 0,
            "reasons": [],
        },
    })

    _write_json(binding, {
        "schema": "V10_QWEN35_RETENTION_ADMISSION_EVIDENCE_BINDING_V2",
        "status": "FROZEN_BEFORE_V4_AUDIT_RESULT",
        "admission_policy": {"file_sha256": policy_sha},
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
        "family_audit_v4": {
            "protocol_file_sha256": protocol_sha,
            "runner_file_sha256": runner_sha,
            "required_reviewer": reviewer,
        },
        "semantic_screen": {
            "file_sha256": semantic_sha,
            "required_status": "PASS",
            "target_sha256": candidate_sha,
            "threshold": 0.9,
            "model_archive_sha256": "d" * 64,
            "exclusion_registry_sha256": exclusion_sha,
        },
        "exclusion_registry": {"file_sha256": exclusion_sha},
        "final_v4_receipt_requirements": {
            "candidate_sha256": candidate_sha,
            "packet_sha256": packet_sha,
            "protocol_sha256": protocol_sha,
            "sample_rows": 1,
            "family_count": 1,
            "cases_per_family": 1,
            "summary_status_required": "PASS",
            "reviewed_rows_required": 1,
            "reviewed_families_required": 1,
            "reject_count_required": 0,
            "review_output_sha256_must_match_file": True,
        },
    })

    return {
        "candidate": candidate,
        "candidate_manifest": candidate_manifest,
        "exclusion": exclusion,
        "packet": packet,
        "packet_manifest": packet_manifest,
        "protocol": protocol,
        "runner": runner,
        "semantic": semantic,
        "policy": policy,
        "binding": binding,
        "audit": audit,
        "receipt": receipt,
    }


def _verify(paths: dict[str, Path]):
    return admission_files.verify_retention_admission_files_v4(
        candidate_path=paths["candidate"],
        candidate_manifest_path=paths["candidate_manifest"],
        exclusion_path=paths["exclusion"],
        packet_path=paths["packet"],
        packet_manifest_path=paths["packet_manifest"],
        protocol_path=paths["protocol"],
        runner_path=paths["runner"],
        audit_output_path=paths["audit"],
        audit_receipt_path=paths["receipt"],
        semantic_path=paths["semantic"],
        admission_policy_path=paths["policy"],
        evidence_binding_path=paths["binding"],
    )


def test_v4_file_binding_passes_exact_subjects_before_core(
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
        "validate_retention_admission_v4",
        lambda **kwargs: {
            "schema": "V10_QWEN35_RETENTION_ADMISSION_CHECK_V4",
            "status": "RETENTION_ADMITTED_V4",
            "reasons": [],
        },
    )
    result = _verify(paths)
    assert result["status"] == "RETENTION_ADMITTED_V4"
    assert result["binding_reasons"] == []


def test_v4_protocol_substitution_fails_before_core(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    called = False

    def core(**kwargs):
        nonlocal called
        called = True
        return {"status": "RETENTION_ADMITTED_V4", "reasons": []}

    monkeypatch.setattr(admission_files, "validate_retention_admission_v4", core)
    paths["protocol"].write_text('{"substituted":true}\n', encoding="utf-8")
    result = _verify(paths)
    assert result["status"] == "HOLD"
    assert "protocol_file_sha256_mismatch" in result["binding_reasons"]
    assert called is False


def test_v4_runner_substitution_fails_before_core(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    called = False

    def core(**kwargs):
        nonlocal called
        called = True
        return {"status": "RETENTION_ADMITTED_V4", "reasons": []}

    monkeypatch.setattr(admission_files, "validate_retention_admission_v4", core)
    paths["runner"].write_text("# modified runner\n", encoding="utf-8")
    result = _verify(paths)
    assert result["status"] == "HOLD"
    assert "runner_file_sha256_mismatch" in result["binding_reasons"]
    assert called is False


def test_v4_receipt_must_bind_exact_protocol(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["protocol_sha256"] = "f" * 64
    _write_json(paths["receipt"], receipt)
    result = _verify(paths)
    assert result["status"] == "HOLD"
    assert "audit_receipt_protocol_sha256_mismatch" in result["binding_reasons"]
