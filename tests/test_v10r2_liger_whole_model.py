from __future__ import annotations

import hashlib
from pathlib import Path

from successor.experiments.probe_v10r2_lane_b_backward import (
    apply_liger_qwen35_candidate,
)


def test_liger_candidate_patch_is_opt_in_and_records_exact_provenance(
    tmp_path: Path,
) -> None:
    model = object()
    calls: list[dict] = []

    control = apply_liger_qwen35_candidate(model)
    assert control == {"enabled": False}

    source = tmp_path / "liger-src"
    source.mkdir()
    wheel = tmp_path / "liger_kernel-0.8.4-py3-none-any.whl"
    wheel.write_bytes(b"pinned-liger-wheel")

    def fake_patch(**kwargs) -> None:
        calls.append(kwargs)

    candidate = apply_liger_qwen35_candidate(
        model,
        liger_path=source,
        liger_wheel=wheel,
        patch_function=fake_patch,
        version="0.8.4",
    )

    assert calls == [
        {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": True,
            "swiglu": True,
            "model": model,
        }
    ]
    assert candidate == {
        "enabled": True,
        "version": "0.8.4",
        "source_path": str(source),
        "wheel_path": str(wheel),
        "wheel_sha256": hashlib.sha256(b"pinned-liger-wheel").hexdigest(),
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
        "swiglu": True,
    }
