from __future__ import annotations

import torch

from successor.experiments.probe_v10r2_liger_micro_parity import _metric_pass, _metrics


def test_liger_micro_metric_identity_passes() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    metric = _metrics(a, a.clone())
    assert metric["relative_l2"] == 0.0
    assert metric["cosine"] > 0.999999
    assert _metric_pass(metric, relative_l2_max=0.02, cosine_min=0.995)


def test_liger_micro_metric_large_difference_fails() -> None:
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    metric = _metrics(a, b)
    assert metric["relative_l2"] > 1.0
    assert metric["cosine"] == 0.0
    assert not _metric_pass(metric, relative_l2_max=0.02, cosine_min=0.995)


def test_liger_micro_metric_nonfinite_fails() -> None:
    a = torch.tensor([1.0, float("nan")])
    b = torch.tensor([1.0, 2.0])
    metric = _metrics(a, b)
    assert metric["all_finite"] is False
    assert not _metric_pass(metric, relative_l2_max=0.02, cosine_min=0.995)


def test_liger_micro_high_cosine_large_magnitude_error_fails() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = a * 1.05
    metric = _metrics(a, b)
    assert metric["cosine"] > 0.999999
    assert metric["relative_l2"] > 0.02
    assert not _metric_pass(metric, relative_l2_max=0.02, cosine_min=0.995)
