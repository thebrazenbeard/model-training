from __future__ import annotations

import hashlib

import pytest
import torch

from successor.experiments.probe_v10r2_lane_b_backward import ProbeHold, _gradient_summary


def test_gradient_summary_covers_all_trainable_gradients(tmp_path) -> None:
    model = torch.nn.Linear(2, 1, bias=True)
    model(torch.tensor([[1.0, -2.0]])).sum().backward()
    vector_path = tmp_path / "grads.pt"
    summary = _gradient_summary(model, torch, vector_out=vector_path)

    assert summary["tensor_count"] == 2
    assert summary["element_count"] == 3
    assert summary["all_finite"] is True
    assert summary["nonfinite_tensor_count"] == 0
    assert summary["l2"] > 0
    assert vector_path.exists()
    assert summary["vector_file_sha256"] == hashlib.sha256(vector_path.read_bytes()).hexdigest()


def test_gradient_summary_fails_on_missing_trainable_gradient() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ProbeHold, match="missing trainable gradient"):
        _gradient_summary(model, torch)
