from __future__ import annotations

import hashlib

from successor.experiments.semantic_screen_binding import (
    verify_semantic_screen_binding,
)


def test_receipt_without_target_hash_is_hold(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text('{"prompt":"x"}\n', encoding="utf-8")
    result = verify_semantic_screen_binding(
        {
            "status": "PASS",
            "threshold": 0.9,
            "model_archive_sha256": "a" * 64,
            "registry": {"registry_sha256": "b" * 64},
        },
        target_path=target,
        expected_threshold=0.9,
        expected_model_archive_sha256="a" * 64,
        expected_registry_sha256="b" * 64,
    )
    assert result["status"] == "HOLD"
    assert "target_sha256_missing" in result["reasons"]


def test_exact_bound_receipt_passes(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text('{"prompt":"x"}\n', encoding="utf-8")
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    result = verify_semantic_screen_binding(
        {
            "status": "PASS",
            "target_sha256": target_sha,
            "threshold": 0.9,
            "model_archive_sha256": "a" * 64,
            "registry": {"registry_sha256": "b" * 64},
        },
        target_path=target,
        expected_threshold=0.9,
        expected_model_archive_sha256="a" * 64,
        expected_registry_sha256="b" * 64,
    )
    assert result["status"] == "PASS"
    assert result["target_sha256"] == target_sha


def test_changed_target_is_hold(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text('{"prompt":"changed"}\n', encoding="utf-8")
    result = verify_semantic_screen_binding(
        {
            "status": "PASS",
            "target_sha256": "0" * 64,
            "threshold": 0.9,
            "model_archive_sha256": "a" * 64,
            "registry": {"registry_sha256": "b" * 64},
        },
        target_path=target,
        expected_threshold=0.9,
        expected_model_archive_sha256="a" * 64,
        expected_registry_sha256="b" * 64,
    )
    assert result["status"] == "HOLD"
    assert any(reason.startswith("target_sha256_mismatch:") for reason in result["reasons"])
