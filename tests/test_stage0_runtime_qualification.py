import importlib.util
import os
from contextlib import contextmanager

from successor.experiments import qualify_v10r3r3_stage0_runtime as stage0


def test_stage0_runtime_qualification_module_exists():
    spec = importlib.util.find_spec(
        "successor.experiments.qualify_v10r3r3_stage0_runtime"
    )
    assert spec is not None


def test_synthetic_smoke_rows_are_fixed_and_corpus_free():
    assert hasattr(stage0, "synthetic_smoke_rows")
    rows = stage0.synthetic_smoke_rows(count=8)
    assert len(rows) == 8
    assert all(set(row) == {"prompt", "response"} for row in rows)
    assert all(row["prompt"].startswith("Stage-0 synthetic runtime smoke") for row in rows)
    assert all("corpus" not in row["prompt"].casefold() for row in rows)
    assert all("corpus" not in row["response"].casefold() for row in rows)
    assert rows == stage0.synthetic_smoke_rows(count=8)


def _valid_receipt_kwargs():
    return {
        "runtime_binding_sha256": "1" * 64,
        "execution_spec_sha256": "2" * 64,
        "weight_digest_before": "a" * 64,
        "weight_digest_after": "b" * 64,
        "optimizer_step_count": 1,
        "optimizer_created": True,
        "gradients_present": True,
        "nonzero_gradient_parameter_count": 3,
        "microbatches_completed": 8,
        "losses": [1.0] * 8,
        "synthetic_input": True,
        "output_artifacts_written": False,
        "optimizer_name": "AdamW8bit",
        "cuda_memory": {"peak_allocated_mib": 123.0},
    }


def test_stage0_receipt_requires_exactly_one_real_weight_change():
    assert hasattr(stage0, "finalize_stage0_smoke_receipt")
    receipt = stage0.finalize_stage0_smoke_receipt(**_valid_receipt_kwargs())
    assert receipt["status"] == "STAGE0_OPTIMIZER_SMOKE_PASS"
    assert receipt["optimizer_step_count"] == 1
    assert receipt["weight_digest_changed"] is True
    assert receipt["synthetic_input"] is True
    assert receipt["output_artifacts_written"] is False


def test_stage0_receipt_rejects_wrong_step_count_and_unchanged_digest():
    assert hasattr(stage0, "Stage0Hold")
    assert hasattr(stage0, "finalize_stage0_smoke_receipt")
    import pytest

    wrong_steps = _valid_receipt_kwargs()
    wrong_steps["optimizer_step_count"] = 2
    with pytest.raises(stage0.Stage0Hold, match="exactly one optimizer step"):
        stage0.finalize_stage0_smoke_receipt(**wrong_steps)

    unchanged = _valid_receipt_kwargs()
    unchanged["weight_digest_after"] = unchanged["weight_digest_before"]
    with pytest.raises(stage0.Stage0Hold, match="digest did not change"):
        stage0.finalize_stage0_smoke_receipt(**unchanged)


def test_stage0_receipt_rejects_non_synthetic_or_persisted_output():
    import pytest

    non_synthetic = _valid_receipt_kwargs()
    non_synthetic["synthetic_input"] = False
    with pytest.raises(stage0.Stage0Hold, match="synthetic input"):
        stage0.finalize_stage0_smoke_receipt(**non_synthetic)

    persisted = _valid_receipt_kwargs()
    persisted["output_artifacts_written"] = True
    with pytest.raises(stage0.Stage0Hold, match="output artifacts"):
        stage0.finalize_stage0_smoke_receipt(**persisted)



def test_live_stage0_optimizer_smoke_seam_exists():
    assert hasattr(stage0, "execute_stage0_optimizer_smoke")


def _valid_soak_kwargs():
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
        "step_times_seconds": [10.0] * 20,
        "elapsed_seconds": 240.0,
        "max_wall_seconds": 900.0,
        "gpu_temperatures_c": [70] * 20,
        "gpu_temperature_abort_c": 88,
        "commit_headroom_mib": [8192.0] * 20,
        "minimum_commit_headroom_mib": 4096.0,
        "step_time_degradation_ratio": 1.5,
        "consecutive_degraded_steps": 5,
        "cuda_memory": {"peak_allocated_mib": 4300.0},
    }


def test_stage0_soak_receipt_passes_only_bounded_20_step_synthetic_run():
    receipt = stage0.finalize_stage0_soak_receipt(**_valid_soak_kwargs())
    assert receipt["status"] == "STAGE0_BOUNDED_SOAK_PASS"
    assert receipt["optimizer_step_count"] == 20
    assert receipt["microbatches_completed"] == 160
    assert receipt["synthetic_input"] is True
    assert receipt["output_artifacts_written"] is False
    assert receipt["weight_digest_changed"] is True
    assert receipt["max_gpu_temperature_c"] == 70
    assert receipt["minimum_commit_headroom_observed_mib"] == 8192.0


def test_stage0_soak_receipt_rejects_short_thermal_low_headroom_and_overrun():
    import pytest

    cases = []

    short = _valid_soak_kwargs()
    short["optimizer_step_count"] = 19
    cases.append(("20 optimizer steps", short))

    hot = _valid_soak_kwargs()
    hot["gpu_temperatures_c"][-1] = 88
    cases.append(("temperature", hot))

    low_headroom = _valid_soak_kwargs()
    low_headroom["commit_headroom_mib"][-1] = 4095.0
    cases.append(("commit headroom", low_headroom))

    overrun = _valid_soak_kwargs()
    overrun["elapsed_seconds"] = 901.0
    cases.append(("wall", overrun))

    for expected, kwargs in cases:
        with pytest.raises(stage0.Stage0Hold, match=expected):
            stage0.finalize_stage0_soak_receipt(**kwargs)


def test_stage0_soak_receipt_rejects_sustained_step_time_degradation():
    import pytest

    degraded = _valid_soak_kwargs()
    degraded["step_times_seconds"] = [10.0] * 15 + [16.0] * 5
    with pytest.raises(stage0.Stage0Hold, match="step-time degradation"):
        stage0.finalize_stage0_soak_receipt(**degraded)


def test_live_stage0_optimizer_soak_seam_exists():
    assert hasattr(stage0, "execute_stage0_optimizer_soak")


def test_transformers_517_allocator_warmup_bypass_is_scoped_and_restored():
    class FakeModelingUtils:
        pass

    class FakeTransformers:
        __version__ = "5.17.0"

    modeling_utils = FakeModelingUtils()
    original_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))

    modeling_utils.caching_allocator_warmup = original

    with stage0.temporary_transformers_allocator_warmup_bypass(
        FakeTransformers(),
        modeling_utils_module=modeling_utils,
    ) as receipt:
        assert modeling_utils.caching_allocator_warmup is not original
        assert modeling_utils.caching_allocator_warmup("ignored") is None
        assert receipt["transformers_version"] == "5.17.0"
        assert receipt["bypass_active"] is True

    assert modeling_utils.caching_allocator_warmup is original
    assert original_calls == []


def test_stage0_qwen_loader_bypasses_only_allocator_warmup_during_from_pretrained():
    class FakeModelingUtils:
        pass

    class FakeTransformers:
        __version__ = "5.17.0"

    modeling_utils = FakeModelingUtils()
    original_called = []

    def original(*args, **kwargs):
        original_called.append(True)
        raise AssertionError("original allocator warmup should be bypassed")

    modeling_utils.caching_allocator_warmup = original

    class FakeQwen:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            modeling_utils.caching_allocator_warmup("model", {"x": 0}, None)
            return {"args": args, "kwargs": kwargs}

    stack = {
        "transformers": FakeTransformers(),
        "Qwen3_5ForCausalLM": FakeQwen,
    }
    model, receipt = stage0.load_stage0_qwen_model(
        stack,
        "C:/fake/base",
        quantization_config="quant",
        device_map={"": 0},
        dtype="bfloat16",
        modeling_utils_module=modeling_utils,
    )
    assert model["kwargs"]["quantization_config"] == "quant"
    assert receipt["allocator_warmup_bypassed"] is True
    assert modeling_utils.caching_allocator_warmup is original
    assert original_called == []


def test_stage0_model_load_probe_seam_exists_before_optimizer_execution():
    assert hasattr(stage0, "execute_stage0_model_load_probe")


def test_transformers_warmup_bypass_restores_original_on_exception():
    import pytest

    class FakeModelingUtils:
        pass

    class FakeTransformers:
        __version__ = "5.17.0"

    modeling_utils = FakeModelingUtils()

    def original(*args, **kwargs):
        return "original"

    modeling_utils.caching_allocator_warmup = original
    with pytest.raises(RuntimeError, match="boom"):
        with stage0.temporary_transformers_allocator_warmup_bypass(
            FakeTransformers(),
            modeling_utils_module=modeling_utils,
        ):
            raise RuntimeError("boom")
    assert modeling_utils.caching_allocator_warmup is original


def test_transformers_warmup_bypass_rejects_unfrozen_version():
    import pytest

    class FakeModelingUtils:
        pass

    class FakeTransformers:
        __version__ = "5.18.0"

    modeling_utils = FakeModelingUtils()
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    with pytest.raises(stage0.Stage0Hold, match="frozen to transformers 5.17.0"):
        with stage0.temporary_transformers_allocator_warmup_bypass(
            FakeTransformers(),
            modeling_utils_module=modeling_utils,
        ):
            pass


def test_stage0_kbit_preparation_passes_frozen_nonreentrant_checkpointing():
    calls = []

    def fake_prepare(model, **kwargs):
        calls.append((model, kwargs))
        return "prepared"

    stack = {"prepare_model_for_kbit_training": fake_prepare}
    prepared = stage0.prepare_stage0_kbit_model(
        stack,
        "model",
        use_gradient_checkpointing=True,
        use_reentrant=False,
    )
    assert prepared == "prepared"
    assert calls == [(
        "model",
        {
            "use_gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "auto_clear_cache": True,
        },
    )]


def test_stage0_gpu_lock_rejects_concurrent_holder_and_releases(tmp_path):
    import pytest

    lock_path = tmp_path / "stage0-gpu.lock"
    if os.name != "nt":
        with pytest.raises(stage0.Stage0Hold, match="requires Windows host"):
            with stage0.exclusive_stage0_gpu_lock(lock_path):
                raise AssertionError("unsupported host unexpectedly acquired GPU lock")
        assert not lock_path.exists()
        return

    with stage0.exclusive_stage0_gpu_lock(lock_path):
        with pytest.raises(stage0.Stage0Hold, match="exclusive Stage-0 GPU lock"):
            with stage0.exclusive_stage0_gpu_lock(lock_path):
                pass

    with stage0.exclusive_stage0_gpu_lock(lock_path):
        pass


def test_stage0_runtime_entrypoints_accept_machine_global_gpu_lock_path():
    import inspect

    for fn in (
        stage0.execute_stage0_model_load_probe,
        stage0.execute_stage0_optimizer_smoke,
        stage0.execute_stage0_optimizer_soak,
    ):
        assert "gpu_lock_path" in inspect.signature(fn).parameters


def test_stage0_preload_commit_gate_rejects_below_floor():
    import pytest

    with pytest.raises(stage0.Stage0Hold, match="commit headroom"):
        stage0.require_stage0_commit_headroom(
            minimum_commit_headroom_mib=4096.0,
            commit_headroom_provider=lambda: 1024.0,
        )


def test_stage0_gpu_entrypoints_check_commit_before_training_stack_load(monkeypatch, tmp_path):
    import pytest

    calls = []

    @contextmanager
    def fake_lock(path):
        # Isolate commit-preflight ordering from the real Windows lock.
        calls.append(("source_only_lock_seam", path.name))
        yield {"synthetic_lock_for_source_test_only": True}

    monkeypatch.setattr(stage0, "exclusive_stage0_gpu_lock", fake_lock)

    def forbidden_loader():
        calls.append("loader")
        raise AssertionError("training stack loader must not run below commit floor")

    entrypoints = (
        lambda: stage0.execute_stage0_model_load_probe(
            tmp_path,
            training_stack_loader=forbidden_loader,
            minimum_commit_headroom_mib=4096.0,
            commit_headroom_provider=lambda: 512.0,
            gpu_lock_path=tmp_path / "probe.lock",
        ),
        lambda: stage0.execute_stage0_optimizer_smoke(
            tmp_path,
            training_stack_loader=forbidden_loader,
            minimum_commit_headroom_mib=4096.0,
            commit_headroom_provider=lambda: 512.0,
            gpu_lock_path=tmp_path / "smoke.lock",
        ),
        lambda: stage0.execute_stage0_optimizer_soak(
            tmp_path,
            training_stack_loader=forbidden_loader,
            minimum_commit_headroom_mib=4096.0,
            commit_headroom_provider=lambda: 512.0,
            gpu_lock_path=tmp_path / "soak.lock",
        ),
    )

    for invoke in entrypoints:
        with pytest.raises(stage0.Stage0Hold, match="commit headroom"):
            invoke()

    assert calls == [
        ("source_only_lock_seam", "probe.lock"),
        ("source_only_lock_seam", "smoke.lock"),
        ("source_only_lock_seam", "soak.lock"),
    ]
