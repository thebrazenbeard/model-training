from __future__ import annotations

import pytest

from successor.experiments.final_bank_synthetic_custodians import (
    build_identity_receipt,
    validate_synthetic_pair,
)


def _receipt(*, actor_id: str, provider: str, model: str, family: str) -> dict:
    return build_identity_receipt(
        actor_id=actor_id,
        provider=provider,
        model=model,
        model_family=family,
        revision="rev-1",
        runtime="runtime-1",
        seed=20261003,
        sampling={"temperature": 0.6, "top_p": 0.9},
        artifacts={"model_sha256": "a" * 64},
    )


def test_distinct_non_qwen_pair_is_ready() -> None:
    a = _receipt(
        actor_id="synthetic-a",
        provider="OLLAMA_LOCAL",
        model="ministral-3:14b",
        family="mistral3",
    )
    b = _receipt(
        actor_id="synthetic-b",
        provider="TRANSFORMERS_LOCAL",
        model="SmolLM3-3B",
        family="smollm3",
    )
    result = validate_synthetic_pair(a, b)
    assert result["status"] == "SYNTHETIC_CUSTODIANS_BOUND"
    assert result["reasons"] == []


def test_qwen_or_same_provider_runtime_pair_fails_closed() -> None:
    a = _receipt(
        actor_id="synthetic-a",
        provider="OLLAMA_LOCAL",
        model="ministral-3:14b",
        family="mistral3",
    )
    b = _receipt(
        actor_id="synthetic-b",
        provider="OLLAMA_LOCAL",
        model="qwen3:8b",
        family="Qwen3",
    )
    result = validate_synthetic_pair(a, b)
    assert result["status"] == "HOLD"
    assert "qwen_lineage_forbidden:B" in result["reasons"]
    assert "provider_runtime_identity_not_distinct" in result["reasons"]


def test_identity_receipt_self_hash_detects_mutation() -> None:
    receipt = _receipt(
        actor_id="synthetic-a",
        provider="OLLAMA_LOCAL",
        model="ministral-3:14b",
        family="mistral3",
    )
    receipt["model"] = "tampered"
    with pytest.raises(ValueError, match="identity_receipt_sha256_mismatch"):
        validate_synthetic_pair(
            receipt,
            _receipt(
                actor_id="synthetic-b",
                provider="TRANSFORMERS_LOCAL",
                model="SmolLM3-3B",
                family="smollm3",
            ),
        )
