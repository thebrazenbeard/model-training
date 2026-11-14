from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from successor.experiments.semantic_screen_v10_retention import (
    max_cosine_scores,
    reconstruct_prompt_map,
    verify_registry,
    verify_target_sha256,
)


def _jsonl(path: Path, prompts: list[str]) -> None:
    path.write_text(
        "\n".join(json.dumps({"prompt": p}) for p in prompts) + "\n",
        encoding="utf-8",
    )


def test_reconstruct_prompt_map_dedupes_normalized_prompts(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _jsonl(a, ["Alpha  Beta", "Gamma"])
    _jsonl(b, [" alpha beta ", "Delta"])
    mapping = reconstruct_prompt_map([a, b])
    assert len(mapping) == 3


def test_verify_registry_requires_exact_hash_set(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _jsonl(source, ["Alpha", "Beta"])
    mapping = reconstruct_prompt_map([source])
    registry = tmp_path / "hashes.txt"
    registry.write_text("\n".join(sorted(mapping)) + "\n", encoding="ascii")
    assert verify_registry(mapping, registry)["count"] == 2
    registry.write_text(next(iter(mapping)) + "\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="registry mismatch"):
        verify_registry(mapping, registry)


def test_max_cosine_scores_is_batched() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    scores, indices = max_cosine_scores(target, source, batch_size=1)
    assert np.allclose(scores, [1.0, 2 ** -0.5])
    assert indices.tolist() == [0, 0]


def test_verify_target_sha256_binds_exact_target_bytes(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(b'{"prompt":"Alpha"}\n')
    import hashlib
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert verify_target_sha256(target, expected) == expected
    with pytest.raises(RuntimeError, match="target hash mismatch"):
        verify_target_sha256(target, "0" * 64)
