import pytest

from successor.experiments import qualify_v10r3r3_stage0_runtime as stage0


def _receipt_kwargs():
    return {
        "runtime_binding_sha256": "1" * 64,
        "execution_spec_sha256": "2" * 64,
        "weight_digest_before": "a" * 64,
        "weight_digest_after": "b" * 64,
        "optimizer_step_count": 20,
        "gradient_accumulation_steps": 8,
        "microbatches_completed": 160,
        "losses": [0.01] * 160,
        "synthetic_input": True,
        "output_artifacts_written": False,
        "optimizer_name": "AdamW",
        "step_times_seconds": [50.0] * 20,
        "elapsed_seconds": 1050.0,
        "max_wall_seconds": 1200.0,
        "gpu_temperatures_c": [70] * 20,
        "gpu_temperature_abort_c": 88,
        "commit_headroom_mib": [18000.0] * 20,
        "minimum_commit_headroom_mib": 8192.0,
        "step_time_degradation_ratio": 1.5,
        "consecutive_degraded_steps": 5,
        "cuda_memory": {"peak_allocated_mib": 3950.0},
    }


def test_r6_allows_bounded_twenty_minute_stage0_wall_budget():
    receipt = stage0.finalize_stage0_soak_receipt(**_receipt_kwargs())
    assert receipt["status"] == "STAGE0_BOUNDED_SOAK_PASS"
    assert receipt["max_wall_seconds"] == 1200.0
    assert receipt["elapsed_seconds"] == 1050.0


def test_r6_still_rejects_wall_budget_above_twenty_minutes():
    kwargs = _receipt_kwargs()
    kwargs["max_wall_seconds"] = 1201.0
    with pytest.raises(stage0.Stage0Hold, match="maximum wall"):
        stage0.finalize_stage0_soak_receipt(**kwargs)
