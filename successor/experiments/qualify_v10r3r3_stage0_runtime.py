from __future__ import annotations


def synthetic_smoke_rows(*, count: int = 8) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("count must be positive")
    return [
        {
            "prompt": f"Stage-0 synthetic runtime smoke {index + 1}: return the fixed token READY.",
            "response": "READY",
        }
        for index in range(count)
    ]


import hashlib
import json
import math
import re


class Stage0Hold(RuntimeError):
    pass


def temporary_transformers_allocator_warmup_bypass(
    transformers_module,
    *,
    modeling_utils_module=None,
):
    from contextlib import contextmanager
    import importlib

    @contextmanager
    def _scope():
        version = getattr(transformers_module, "__version__", None)
        if version != "5.17.0":
            raise Stage0Hold(
                "allocator warmup bypass is frozen to transformers 5.17.0"
            )
        module = modeling_utils_module
        if module is None:
            module = importlib.import_module("transformers.modeling_utils")
        original = getattr(module, "caching_allocator_warmup", None)
        if not callable(original):
            raise Stage0Hold("transformers allocator warmup seam is unavailable")

        def _bypass(*_args, **_kwargs):
            return None

        module.caching_allocator_warmup = _bypass
        try:
            yield {
                "transformers_version": version,
                "bypass_active": True,
            }
        finally:
            module.caching_allocator_warmup = original

    return _scope()


def load_stage0_qwen_model(
    stack: dict,
    base_path,
    *,
    quantization_config,
    device_map,
    dtype,
    modeling_utils_module=None,
):
    with temporary_transformers_allocator_warmup_bypass(
        stack["transformers"],
        modeling_utils_module=modeling_utils_module,
    ) as bypass_receipt:
        model = stack["Qwen3_5ForCausalLM"].from_pretrained(
            base_path,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=dtype,
        )
    return model, {
        **bypass_receipt,
        "allocator_warmup_bypassed": True,
    }


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Stage0Hold(f"{name} must be a lowercase sha256 digest")


def finalize_stage0_smoke_receipt(
    *,
    runtime_binding_sha256: str,
    execution_spec_sha256: str,
    weight_digest_before: str,
    weight_digest_after: str,
    optimizer_step_count: int,
    optimizer_created: bool,
    gradients_present: bool,
    nonzero_gradient_parameter_count: int,
    microbatches_completed: int,
    losses: list[float],
    synthetic_input: bool,
    output_artifacts_written: bool,
    optimizer_name: str,
    cuda_memory: dict,
) -> dict:
    for name, value in (
        ("runtime_binding_sha256", runtime_binding_sha256),
        ("execution_spec_sha256", execution_spec_sha256),
        ("weight_digest_before", weight_digest_before),
        ("weight_digest_after", weight_digest_after),
    ):
        _require_sha256(name, value)

    if optimizer_step_count != 1:
        raise Stage0Hold("Stage-0 requires exactly one optimizer step")
    if not optimizer_created:
        raise Stage0Hold("optimizer was not created")
    if not gradients_present or nonzero_gradient_parameter_count < 1:
        raise Stage0Hold("nonzero trainable gradients were not observed")
    if microbatches_completed < 1 or len(losses) != microbatches_completed:
        raise Stage0Hold("microbatch/loss accounting mismatch")
    if not all(isinstance(loss, (int, float)) and math.isfinite(float(loss)) for loss in losses):
        raise Stage0Hold("losses must all be finite")
    if not synthetic_input:
        raise Stage0Hold("Stage-0 smoke requires synthetic input")
    if output_artifacts_written:
        raise Stage0Hold("Stage-0 smoke must not write output artifacts")
    if weight_digest_before == weight_digest_after:
        raise Stage0Hold("trainable parameter digest did not change")
    if not optimizer_name:
        raise Stage0Hold("optimizer name missing")
    if not isinstance(cuda_memory, dict):
        raise Stage0Hold("cuda memory receipt missing")

    receipt = {
        "schema": "V10R3R3_STAGE0_RUNTIME_SMOKE_RECEIPT_V1",
        "status": "STAGE0_OPTIMIZER_SMOKE_PASS",
        "claim_ceiling": "DISPOSABLE_SYNTHETIC_ONE_STEP_RUNTIME_QUALIFICATION_ONLY",
        "runtime_binding_sha256": runtime_binding_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "optimizer_step_count": optimizer_step_count,
        "optimizer_created": optimizer_created,
        "optimizer_name": optimizer_name,
        "gradients_present": gradients_present,
        "nonzero_gradient_parameter_count": nonzero_gradient_parameter_count,
        "microbatches_completed": microbatches_completed,
        "losses": [float(loss) for loss in losses],
        "synthetic_input": synthetic_input,
        "output_artifacts_written": output_artifacts_written,
        "weight_digest_before": weight_digest_before,
        "weight_digest_after": weight_digest_after,
        "weight_digest_changed": True,
        "cuda_memory": cuda_memory,
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return receipt


def _execute_stage0_model_load_probe_unlocked(
    repo_root,
    *,
    training_stack_loader=None,
) -> dict:
    from pathlib import Path

    from successor.experiments.train_v10_qwen35_authorized import (
        _observe_live_runtime,
        _read_json,
        _validate_qwen_topology,
        load_training_stack,
        sha256_file,
        validate_runtime_observation,
    )

    root = Path(repo_root)
    experiment_dir = root / "successor" / "experiments"
    runtime_path = experiment_dir / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    execution_spec_path = (
        experiment_dir
        / "V10R3R3_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261005_V1.json"
    )
    runtime_binding = _read_json(runtime_path)
    execution_spec = _read_json(execution_spec_path)

    if training_stack_loader is None:
        training_stack_loader = load_training_stack
    stack = training_stack_loader()
    if not isinstance(stack, dict):
        raise Stage0Hold("training stack loader returned invalid object")

    live_runtime = _observe_live_runtime(runtime_binding, stack)
    validate_runtime_observation(runtime_binding, live_runtime)

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise Stage0Hold("CUDA is unavailable")

    quant = execution_spec["quantization"]
    bits_config = stack["BitsAndBytesConfig"](
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
    )
    model_load = execution_spec["model_load"]
    target = runtime_binding["target"]
    model, load_receipt = load_stage0_qwen_model(
        stack,
        Path(target["base_path"]),
        quantization_config=bits_config,
        device_map=model_load["device_map"],
        dtype=getattr(torch, model_load["dtype"]),
    )
    model.config.use_cache = model_load["use_cache"]
    _validate_qwen_topology(model)
    torch.cuda.synchronize()
    memory = {
        "allocated_after_load_mib": round(
            torch.cuda.memory_allocated() / (1024**2), 3
        ),
        "reserved_after_load_mib": round(
            torch.cuda.memory_reserved() / (1024**2), 3
        ),
    }
    model_class = model.__class__.__name__
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "schema": "V10R3R3_STAGE0_MODEL_LOAD_PROBE_RECEIPT_V1",
        "status": "STAGE0_MODEL_LOAD_PROBE_PASS",
        "claim_ceiling": "EXACT_MODEL_LOAD_ONLY_NO_OPTIMIZER",
        "runtime_binding_sha256": runtime_binding["binding_sha256"],
        "execution_spec_sha256": sha256_file(execution_spec_path),
        "transformers_version": load_receipt["transformers_version"],
        "allocator_warmup_bypassed": load_receipt[
            "allocator_warmup_bypassed"
        ],
        "topology_validated": True,
        "model_class": model_class,
        "cuda_available": True,
        "optimizer_created": False,
        "output_artifacts_written": False,
        "cuda_memory": memory,
    }


def prepare_stage0_kbit_model(
    stack: dict,
    model,
    *,
    use_gradient_checkpointing: bool,
    use_reentrant: bool,
):
    return stack["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
        auto_clear_cache=True,
    )


def _execute_stage0_optimizer_smoke_unlocked(
    repo_root,
    *,
    training_stack_loader=None,
) -> dict:
    import tempfile
    from pathlib import Path

    from successor.experiments.probe_v10_qwen35_backward import (
        _trainable_parameter_digest,
    )
    from successor.experiments.train_v10_qwen35_authorized import (
        _observe_live_runtime,
        _read_json,
        _validate_qwen_topology,
        load_training_stack,
        prepare_sft_rows,
        sha256_file,
        validate_lora_target_coverage,
        validate_runtime_observation,
    )

    root = Path(repo_root)
    experiment_dir = root / "successor" / "experiments"
    runtime_path = (
        experiment_dir / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    execution_spec_path = (
        experiment_dir
        / "V10R3R3_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261005_V1.json"
    )
    runtime_binding = _read_json(runtime_path)
    execution_spec = _read_json(execution_spec_path)

    if training_stack_loader is None:
        training_stack_loader = load_training_stack
    stack = training_stack_loader()
    if not isinstance(stack, dict):
        raise Stage0Hold("training stack loader returned invalid object")

    live_runtime = _observe_live_runtime(runtime_binding, stack)
    validate_runtime_observation(runtime_binding, live_runtime)

    trainer_spec = execution_spec["trainer"]
    accumulation_steps = int(trainer_spec["gradient_accumulation_steps"])
    if accumulation_steps != 8:
        raise Stage0Hold(
            "Stage-0 frozen synthetic smoke requires gradient accumulation 8"
        )
    rows = synthetic_smoke_rows(count=accumulation_steps)

    target = runtime_binding["target"]
    base_path = Path(target["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared, _token_budget = prepare_sft_rows(
        tokenizer,
        rows,
        max_length=int(trainer_spec["max_length"]),
    )

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise Stage0Hold("CUDA is unavailable")

    quant = execution_spec["quantization"]
    bits_config = stack["BitsAndBytesConfig"](
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
    )
    model_load = execution_spec["model_load"]
    model, _model_load_receipt = load_stage0_qwen_model(
        stack,
        base_path,
        quantization_config=bits_config,
        device_map=model_load["device_map"],
        dtype=getattr(torch, model_load["dtype"]),
    )
    model.config.use_cache = model_load["use_cache"]
    _validate_qwen_topology(model)
    model = prepare_stage0_kbit_model(
        stack,
        model,
        use_gradient_checkpointing=model_load[
            "prepare_model_for_kbit_training_use_gradient_checkpointing"
        ],
        use_reentrant=trainer_spec[
            "gradient_checkpointing_use_reentrant"
        ],
    )

    lora_spec = execution_spec["lora"]
    lora = stack["LoraConfig"](
        r=lora_spec["r"],
        lora_alpha=lora_spec["alpha"],
        lora_dropout=lora_spec["dropout"],
        bias=lora_spec["bias"],
        task_type=lora_spec["task_type"],
        target_modules=lora_spec["target_modules"],
    )
    stack["transformers"].set_seed(trainer_spec["seed"])

    with tempfile.TemporaryDirectory(
        prefix="vera-v10r3r3-stage0-smoke-"
    ) as temp_dir:
        sft_config = stack["SFTConfig"](
            output_dir=temp_dir,
            per_device_train_batch_size=trainer_spec[
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=accumulation_steps,
            num_train_epochs=1,
            learning_rate=trainer_spec["learning_rate"],
            lr_scheduler_type=trainer_spec["lr_scheduler_type"],
            warmup_steps=trainer_spec["warmup_optimizer_steps"],
            optim=trainer_spec["optimizer"],
            bf16=trainer_spec["bf16"],
            tf32=trainer_spec["tf32"],
            gradient_checkpointing=trainer_spec["gradient_checkpointing"],
            gradient_checkpointing_kwargs={
                "use_reentrant": trainer_spec[
                    "gradient_checkpointing_use_reentrant"
                ]
            },
            max_length=trainer_spec["max_length"],
            completion_only_loss=trainer_spec["completion_only_loss"],
            packing=False,
            shuffle_dataset=False,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=trainer_spec["seed"],
            data_seed=trainer_spec["seed"],
        )
        trainer = stack["SFTTrainer"](
            model=model,
            args=sft_config,
            train_dataset=stack["Dataset"].from_list(prepared),
            processing_class=tokenizer,
            peft_config=lora,
        )
        validate_lora_target_coverage(
            getattr(trainer.model, "targeted_module_names", [])
        )
        if trainer.optimizer is not None:
            raise Stage0Hold("optimizer existed before explicit Stage-0 creation")

        weight_before, trainable_count = _trainable_parameter_digest(
            trainer.model,
            torch,
        )
        expected_before = execution_spec["comparison"].get(
            "expected_initial_trainable_parameter_digest"
        )
        if weight_before != expected_before:
            raise Stage0Hold(
                "initial trainable digest mismatch:"
                f"{weight_before}!={expected_before}"
            )

        dataloader = trainer.get_train_dataloader()
        iterator = iter(dataloader)
        trainer.model.train()
        trainer.model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()

        losses: list[float] = []
        for microbatch_index in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise Stage0Hold(
                    "synthetic Stage-0 dataloader ended early:"
                    f"{microbatch_index}/{accumulation_steps}"
                ) from exc
            batch = trainer._prepare_inputs(batch)
            loss = trainer.compute_loss(trainer.model, batch)
            if not bool(torch.isfinite(loss.detach()).item()):
                raise Stage0Hold(
                    f"loss is not finite at microbatch:{microbatch_index}"
                )
            losses.append(float(loss.detach().float().cpu().item()))
            trainer.accelerator.backward(loss / accumulation_steps)

        torch.cuda.synchronize()
        nonzero_gradient_parameter_count = 0
        for _, parameter in trainer.model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
                and bool(torch.count_nonzero(parameter.grad.detach()).item())
            ):
                nonzero_gradient_parameter_count += 1
        if nonzero_gradient_parameter_count < 1:
            raise Stage0Hold("no nonzero trainable gradients observed")

        trainer.create_optimizer()
        if trainer.optimizer is None:
            raise Stage0Hold("optimizer creation failed")
        optimizer_name = trainer.optimizer.__class__.__name__
        trainer.optimizer.step()
        torch.cuda.synchronize()

        weight_after, post_trainable_count = _trainable_parameter_digest(
            trainer.model,
            torch,
        )
        if post_trainable_count != trainable_count:
            raise Stage0Hold("trainable parameter count changed")

        memory = {
            "allocated_before_backward_mib": round(
                allocated_before / (1024**2), 3
            ),
            "reserved_before_backward_mib": round(
                reserved_before / (1024**2), 3
            ),
            "allocated_after_step_mib": round(
                torch.cuda.memory_allocated() / (1024**2), 3
            ),
            "reserved_after_step_mib": round(
                torch.cuda.memory_reserved() / (1024**2), 3
            ),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / (1024**2), 3
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved() / (1024**2), 3
            ),
        }

        trainer.model.zero_grad(set_to_none=True)
        trainer.optimizer.zero_grad(set_to_none=True)

        return finalize_stage0_smoke_receipt(
            runtime_binding_sha256=runtime_binding["binding_sha256"],
            execution_spec_sha256=sha256_file(execution_spec_path),
            weight_digest_before=weight_before,
            weight_digest_after=weight_after,
            optimizer_step_count=1,
            optimizer_created=True,
            gradients_present=True,
            nonzero_gradient_parameter_count=nonzero_gradient_parameter_count,
            microbatches_completed=accumulation_steps,
            losses=losses,
            synthetic_input=True,
            output_artifacts_written=False,
            optimizer_name=optimizer_name,
            cuda_memory=memory,
        )


def _stage0_soak_has_sustained_degradation(
    step_times_seconds: list[float],
    *,
    ratio: float,
    consecutive_steps: int,
) -> bool:
    import statistics

    if consecutive_steps < 1:
        raise Stage0Hold("consecutive degraded steps must be positive")
    if ratio <= 1.0:
        raise Stage0Hold("step-time degradation ratio must exceed 1")
    if len(step_times_seconds) <= consecutive_steps:
        return False
    for start in range(1, len(step_times_seconds) - consecutive_steps + 1):
        baseline = step_times_seconds[:start]
        if not baseline:
            continue
        median = float(statistics.median(baseline))
        if median <= 0:
            raise Stage0Hold("step-time baseline median must be positive")
        window = step_times_seconds[start : start + consecutive_steps]
        if len(window) == consecutive_steps and all(
            float(value) > median * ratio for value in window
        ):
            return True
    return False


def finalize_stage0_soak_receipt(
    *,
    runtime_binding_sha256: str,
    execution_spec_sha256: str,
    weight_digest_before: str,
    weight_digest_after: str,
    optimizer_step_count: int,
    gradient_accumulation_steps: int,
    microbatches_completed: int,
    losses: list[float],
    synthetic_input: bool,
    output_artifacts_written: bool,
    optimizer_name: str,
    step_times_seconds: list[float],
    elapsed_seconds: float,
    max_wall_seconds: float,
    gpu_temperatures_c: list[int],
    gpu_temperature_abort_c: int,
    commit_headroom_mib: list[float],
    minimum_commit_headroom_mib: float,
    step_time_degradation_ratio: float,
    consecutive_degraded_steps: int,
    cuda_memory: dict,
) -> dict:
    import statistics

    for name, value in (
        ("runtime_binding_sha256", runtime_binding_sha256),
        ("execution_spec_sha256", execution_spec_sha256),
        ("weight_digest_before", weight_digest_before),
        ("weight_digest_after", weight_digest_after),
    ):
        _require_sha256(name, value)

    if optimizer_step_count != 20:
        raise Stage0Hold("Stage-0 soak requires exactly 20 optimizer steps")
    if gradient_accumulation_steps != 8:
        raise Stage0Hold("Stage-0 soak requires gradient accumulation 8")
    expected_microbatches = optimizer_step_count * gradient_accumulation_steps
    if microbatches_completed != expected_microbatches:
        raise Stage0Hold("Stage-0 soak microbatch accounting mismatch")
    if len(losses) != microbatches_completed or not all(
        isinstance(loss, (int, float)) and math.isfinite(float(loss))
        for loss in losses
    ):
        raise Stage0Hold("Stage-0 soak losses must be finite and complete")
    if not synthetic_input:
        raise Stage0Hold("Stage-0 soak requires synthetic input")
    if output_artifacts_written:
        raise Stage0Hold("Stage-0 soak must not write model output artifacts")
    if weight_digest_before == weight_digest_after:
        raise Stage0Hold("Stage-0 soak trainable parameter digest did not change")
    if not optimizer_name:
        raise Stage0Hold("Stage-0 soak optimizer name missing")

    if max_wall_seconds <= 0 or max_wall_seconds > 1200:
        raise Stage0Hold("Stage-0 soak maximum wall limit exceeds 20 minutes")
    if not math.isfinite(float(elapsed_seconds)) or elapsed_seconds > max_wall_seconds:
        raise Stage0Hold("Stage-0 soak wall-time limit exceeded")

    if len(step_times_seconds) != optimizer_step_count or not all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
        for value in step_times_seconds
    ):
        raise Stage0Hold("Stage-0 soak step-time accounting invalid")
    if _stage0_soak_has_sustained_degradation(
        [float(value) for value in step_times_seconds],
        ratio=float(step_time_degradation_ratio),
        consecutive_steps=int(consecutive_degraded_steps),
    ):
        raise Stage0Hold("Stage-0 soak sustained step-time degradation detected")

    if len(gpu_temperatures_c) != optimizer_step_count:
        raise Stage0Hold("Stage-0 soak temperature accounting invalid")
    max_temperature = max(int(value) for value in gpu_temperatures_c)
    if max_temperature >= int(gpu_temperature_abort_c):
        raise Stage0Hold("Stage-0 soak temperature abort threshold reached")

    if len(commit_headroom_mib) != optimizer_step_count:
        raise Stage0Hold("Stage-0 soak commit headroom accounting invalid")
    minimum_headroom = min(float(value) for value in commit_headroom_mib)
    if minimum_headroom < float(minimum_commit_headroom_mib):
        raise Stage0Hold("Stage-0 soak commit headroom threshold breached")

    sorted_times = sorted(float(value) for value in step_times_seconds)
    p95_index = max(0, math.ceil(0.95 * len(sorted_times)) - 1)
    median_time = float(statistics.median(sorted_times))
    receipt = {
        "schema": "V10R3R3_STAGE0_BOUNDED_SOAK_RECEIPT_V1",
        "status": "STAGE0_BOUNDED_SOAK_PASS",
        "claim_ceiling": "DISPOSABLE_SYNTHETIC_20_STEP_RUNTIME_SOAK_ONLY",
        "runtime_binding_sha256": runtime_binding_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "optimizer_step_count": optimizer_step_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "microbatches_completed": microbatches_completed,
        "losses": [float(loss) for loss in losses],
        "synthetic_input": synthetic_input,
        "output_artifacts_written": output_artifacts_written,
        "optimizer_name": optimizer_name,
        "weight_digest_before": weight_digest_before,
        "weight_digest_after": weight_digest_after,
        "weight_digest_changed": True,
        "step_times_seconds": [float(value) for value in step_times_seconds],
        "step_time_median_seconds": median_time,
        "step_time_p95_seconds": sorted_times[p95_index],
        "elapsed_seconds": float(elapsed_seconds),
        "max_wall_seconds": float(max_wall_seconds),
        "gpu_temperatures_c": [int(value) for value in gpu_temperatures_c],
        "max_gpu_temperature_c": max_temperature,
        "gpu_temperature_abort_c": int(gpu_temperature_abort_c),
        "commit_headroom_mib": [float(value) for value in commit_headroom_mib],
        "minimum_commit_headroom_observed_mib": minimum_headroom,
        "minimum_commit_headroom_required_mib": float(
            minimum_commit_headroom_mib
        ),
        "step_time_degradation_ratio": float(step_time_degradation_ratio),
        "consecutive_degraded_steps": int(consecutive_degraded_steps),
        "cuda_memory": cuda_memory,
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return receipt


def append_stage0_step_telemetry(telemetry_path, record: dict) -> None:
    import os
    from pathlib import Path

    path = Path(telemetry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _execute_stage0_optimizer_soak_unlocked(
    repo_root,
    *,
    optimizer_steps: int = 20,
    max_wall_seconds: float = 900.0,
    minimum_commit_headroom_mib: float = 4096.0,
    gpu_temperature_abort_c: int = 88,
    step_time_degradation_ratio: float = 1.5,
    consecutive_degraded_steps: int = 5,
    training_stack_loader=None,
    telemetry_path=None,
) -> dict:
    import ctypes
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    from successor.experiments.probe_v10_qwen35_backward import (
        _trainable_parameter_digest,
    )
    from successor.experiments.train_v10_qwen35_authorized import (
        _observe_live_runtime,
        _read_json,
        _validate_qwen_topology,
        load_training_stack,
        prepare_sft_rows,
        sha256_file,
        validate_lora_target_coverage,
        validate_runtime_observation,
    )

    if optimizer_steps != 20:
        raise Stage0Hold("live Stage-0 soak subject is frozen to 20 steps")
    if max_wall_seconds > 1200:
        raise Stage0Hold("live Stage-0 soak cannot exceed 20 minutes")

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def commit_headroom_mib() -> float:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            raise Stage0Hold("GlobalMemoryStatusEx failed")
        return float(status.ullAvailPageFile) / (1024**2)

    def gpu_temperature_c() -> int:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
        if len(raw) != 1:
            raise Stage0Hold("unexpected GPU temperature readback")
        return int(raw[0].strip())

    start_wall = time.perf_counter()
    root = Path(repo_root)
    experiment_dir = root / "successor" / "experiments"
    runtime_path = experiment_dir / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    execution_spec_path = (
        experiment_dir
        / "V10R3R3_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261005_V1.json"
    )
    runtime_binding = _read_json(runtime_path)
    execution_spec = _read_json(execution_spec_path)

    if training_stack_loader is None:
        training_stack_loader = load_training_stack
    stack = training_stack_loader()
    live_runtime = _observe_live_runtime(runtime_binding, stack)
    validate_runtime_observation(runtime_binding, live_runtime)

    trainer_spec = execution_spec["trainer"]
    accumulation_steps = int(trainer_spec["gradient_accumulation_steps"])
    if accumulation_steps != 8:
        raise Stage0Hold("Stage-0 soak requires frozen gradient accumulation 8")

    rows = synthetic_smoke_rows(count=accumulation_steps)
    target = runtime_binding["target"]
    base_path = Path(target["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared, _token_budget = prepare_sft_rows(
        tokenizer,
        rows,
        max_length=int(trainer_spec["max_length"]),
    )

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise Stage0Hold("CUDA is unavailable")

    quant = execution_spec["quantization"]
    bits_config = stack["BitsAndBytesConfig"](
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
    )
    model_load = execution_spec["model_load"]
    model, _model_load_receipt = load_stage0_qwen_model(
        stack,
        base_path,
        quantization_config=bits_config,
        device_map=model_load["device_map"],
        dtype=getattr(torch, model_load["dtype"]),
    )
    model.config.use_cache = model_load["use_cache"]
    _validate_qwen_topology(model)
    model = prepare_stage0_kbit_model(
        stack,
        model,
        use_gradient_checkpointing=model_load[
            "prepare_model_for_kbit_training_use_gradient_checkpointing"
        ],
        use_reentrant=trainer_spec[
            "gradient_checkpointing_use_reentrant"
        ],
    )

    lora_spec = execution_spec["lora"]
    lora = stack["LoraConfig"](
        r=lora_spec["r"],
        lora_alpha=lora_spec["alpha"],
        lora_dropout=lora_spec["dropout"],
        bias=lora_spec["bias"],
        task_type=lora_spec["task_type"],
        target_modules=lora_spec["target_modules"],
    )
    stack["transformers"].set_seed(trainer_spec["seed"])

    with tempfile.TemporaryDirectory(
        prefix="vera-v10r3r3-stage0-soak-"
    ) as temp_dir:
        sft_config = stack["SFTConfig"](
            output_dir=temp_dir,
            per_device_train_batch_size=trainer_spec[
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=accumulation_steps,
            num_train_epochs=optimizer_steps,
            learning_rate=trainer_spec["learning_rate"],
            lr_scheduler_type=trainer_spec["lr_scheduler_type"],
            warmup_steps=trainer_spec["warmup_optimizer_steps"],
            optim=trainer_spec["optimizer"],
            bf16=trainer_spec["bf16"],
            tf32=trainer_spec["tf32"],
            gradient_checkpointing=trainer_spec["gradient_checkpointing"],
            gradient_checkpointing_kwargs={
                "use_reentrant": trainer_spec[
                    "gradient_checkpointing_use_reentrant"
                ]
            },
            max_length=trainer_spec["max_length"],
            completion_only_loss=trainer_spec["completion_only_loss"],
            packing=False,
            shuffle_dataset=False,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=trainer_spec["seed"],
            data_seed=trainer_spec["seed"],
        )
        trainer = stack["SFTTrainer"](
            model=model,
            args=sft_config,
            train_dataset=stack["Dataset"].from_list(prepared),
            processing_class=tokenizer,
            peft_config=lora,
        )
        validate_lora_target_coverage(
            getattr(trainer.model, "targeted_module_names", [])
        )
        weight_before, trainable_count = _trainable_parameter_digest(
            trainer.model,
            torch,
        )
        expected_before = execution_spec["comparison"].get(
            "expected_initial_trainable_parameter_digest"
        )
        if weight_before != expected_before:
            raise Stage0Hold(
                "initial trainable digest mismatch:"
                f"{weight_before}!={expected_before}"
            )

        trainer.create_optimizer()
        if trainer.optimizer is None:
            raise Stage0Hold("Stage-0 soak optimizer creation failed")
        optimizer_name = trainer.optimizer.__class__.__name__

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        losses: list[float] = []
        step_times: list[float] = []
        temperatures: list[int] = []
        headrooms: list[float] = []
        microbatches_completed = 0
        trainer.model.train()

        for step_index in range(optimizer_steps):
            if time.perf_counter() - start_wall > max_wall_seconds:
                raise Stage0Hold(
                    f"Stage-0 soak wall-time limit exceeded before step {step_index + 1}"
                )
            step_start = time.perf_counter()
            trainer.model.zero_grad(set_to_none=True)
            trainer.optimizer.zero_grad(set_to_none=True)

            dataloader = trainer.get_train_dataloader()
            step_microbatches = 0
            for batch in dataloader:
                batch = trainer._prepare_inputs(batch)
                loss = trainer.compute_loss(trainer.model, batch)
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise Stage0Hold(
                        f"Stage-0 soak non-finite loss at step {step_index + 1}"
                    )
                losses.append(float(loss.detach().float().cpu().item()))
                trainer.accelerator.backward(loss / accumulation_steps)
                step_microbatches += 1
                microbatches_completed += 1
            if step_microbatches != accumulation_steps:
                raise Stage0Hold(
                    "Stage-0 soak dataloader did not produce exactly "
                    f"{accumulation_steps} microbatches"
                )

            trainer.optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

            temperature = gpu_temperature_c()
            headroom = commit_headroom_mib()
            temperatures.append(temperature)
            headrooms.append(headroom)
            if telemetry_path is not None:
                append_stage0_step_telemetry(
                    telemetry_path,
                    {
                        "step": step_index + 1,
                        "elapsed_seconds": time.perf_counter() - start_wall,
                        "step_time_seconds": step_times[-1],
                        "gpu_temperature_c": temperature,
                        "commit_headroom_mib": headroom,
                        "microbatches_completed": microbatches_completed,
                    },
                )
            if temperature >= gpu_temperature_abort_c:
                raise Stage0Hold(
                    "Stage-0 soak temperature abort threshold reached "
                    f"at step {step_index + 1}: {temperature}C"
                )
            if headroom < minimum_commit_headroom_mib:
                raise Stage0Hold(
                    "Stage-0 soak commit headroom threshold breached "
                    f"at step {step_index + 1}: {headroom:.1f} MiB"
                )
            if _stage0_soak_has_sustained_degradation(
                step_times,
                ratio=step_time_degradation_ratio,
                consecutive_steps=consecutive_degraded_steps,
            ):
                raise Stage0Hold(
                    "Stage-0 soak sustained step-time degradation detected "
                    f"at step {step_index + 1}"
                )

        elapsed = time.perf_counter() - start_wall
        weight_after, post_trainable_count = _trainable_parameter_digest(
            trainer.model,
            torch,
        )
        if post_trainable_count != trainable_count:
            raise Stage0Hold("Stage-0 soak trainable parameter count changed")

        memory = {
            "allocated_after_soak_mib": round(
                torch.cuda.memory_allocated() / (1024**2), 3
            ),
            "reserved_after_soak_mib": round(
                torch.cuda.memory_reserved() / (1024**2), 3
            ),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / (1024**2), 3
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved() / (1024**2), 3
            ),
        }
        return finalize_stage0_soak_receipt(
            runtime_binding_sha256=runtime_binding["binding_sha256"],
            execution_spec_sha256=sha256_file(execution_spec_path),
            weight_digest_before=weight_before,
            weight_digest_after=weight_after,
            optimizer_step_count=optimizer_steps,
            gradient_accumulation_steps=accumulation_steps,
            microbatches_completed=microbatches_completed,
            losses=losses,
            synthetic_input=True,
            output_artifacts_written=False,
            optimizer_name=optimizer_name,
            step_times_seconds=step_times,
            elapsed_seconds=elapsed,
            max_wall_seconds=max_wall_seconds,
            gpu_temperatures_c=temperatures,
            gpu_temperature_abort_c=gpu_temperature_abort_c,
            commit_headroom_mib=headrooms,
            minimum_commit_headroom_mib=minimum_commit_headroom_mib,
            step_time_degradation_ratio=step_time_degradation_ratio,
            consecutive_degraded_steps=consecutive_degraded_steps,
            cuda_memory=memory,
        )


_ACTIVE_STAGE0_GPU_LOCKS: set[str] = set()


def _default_stage0_gpu_lock_path():
    import os
    from pathlib import Path

    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise Stage0Hold("LOCALAPPDATA unavailable for Stage-0 GPU lock")
    return Path(base) / "ProjectRunner" / "locks" / "stage0-gpu-qualification.lock"


def exclusive_stage0_gpu_lock(lock_path=None):
    from contextlib import contextmanager
    from pathlib import Path
    import os

    if os.name == "nt":
        import msvcrt
    else:
        import fcntl

    @contextmanager
    def _scope():
        path = Path(lock_path) if lock_path is not None else _default_stage0_gpu_lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        key = str(path.resolve()).casefold()
        if key in _ACTIVE_STAGE0_GPU_LOCKS:
            raise Stage0Hold("exclusive Stage-0 GPU lock is already held")

        handle = path.open("a+b")
        try:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise Stage0Hold(
                    "exclusive Stage-0 GPU lock is already held"
                ) from exc

            _ACTIVE_STAGE0_GPU_LOCKS.add(key)
            try:
                yield {"lock_path": str(path.resolve())}
            finally:
                _ACTIVE_STAGE0_GPU_LOCKS.discard(key)
                handle.seek(0)
                try:
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            handle.close()

    return _scope()


def stage0_commit_headroom_mib() -> float:
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise Stage0Hold("GlobalMemoryStatusEx failed")
    return float(status.ullAvailPageFile) / (1024**2)


def require_stage0_commit_headroom(
    *,
    minimum_commit_headroom_mib: float = 4096.0,
    commit_headroom_provider=None,
) -> float:
    if minimum_commit_headroom_mib <= 0:
        raise Stage0Hold("minimum commit headroom must be positive")
    provider = commit_headroom_provider or stage0_commit_headroom_mib
    observed = float(provider())
    if not math.isfinite(observed):
        raise Stage0Hold("Stage-0 commit headroom readback is non-finite")
    if observed < float(minimum_commit_headroom_mib):
        raise Stage0Hold(
            "Stage-0 commit headroom below required minimum: "
            f"{observed:.1f} MiB < {float(minimum_commit_headroom_mib):.1f} MiB"
        )
    return observed


def execute_stage0_model_load_probe(
    repo_root,
    *,
    training_stack_loader=None,
    minimum_commit_headroom_mib: float = 4096.0,
    commit_headroom_provider=None,
    gpu_lock_path=None,
) -> dict:
    with exclusive_stage0_gpu_lock(gpu_lock_path):
        require_stage0_commit_headroom(
            minimum_commit_headroom_mib=minimum_commit_headroom_mib,
            commit_headroom_provider=commit_headroom_provider,
        )
        return _execute_stage0_model_load_probe_unlocked(
            repo_root,
            training_stack_loader=training_stack_loader,
        )


def execute_stage0_optimizer_smoke(
    repo_root,
    *,
    training_stack_loader=None,
    minimum_commit_headroom_mib: float = 4096.0,
    commit_headroom_provider=None,
    gpu_lock_path=None,
) -> dict:
    with exclusive_stage0_gpu_lock(gpu_lock_path):
        require_stage0_commit_headroom(
            minimum_commit_headroom_mib=minimum_commit_headroom_mib,
            commit_headroom_provider=commit_headroom_provider,
        )
        return _execute_stage0_optimizer_smoke_unlocked(
            repo_root,
            training_stack_loader=training_stack_loader,
        )


def execute_stage0_optimizer_soak(
    repo_root,
    *,
    optimizer_steps: int = 20,
    max_wall_seconds: float = 900.0,
    minimum_commit_headroom_mib: float = 4096.0,
    gpu_temperature_abort_c: int = 88,
    step_time_degradation_ratio: float = 1.5,
    consecutive_degraded_steps: int = 5,
    training_stack_loader=None,
    commit_headroom_provider=None,
    gpu_lock_path=None,
    telemetry_path=None,
) -> dict:
    with exclusive_stage0_gpu_lock(gpu_lock_path):
        require_stage0_commit_headroom(
            minimum_commit_headroom_mib=minimum_commit_headroom_mib,
            commit_headroom_provider=commit_headroom_provider,
        )
        return _execute_stage0_optimizer_soak_unlocked(
            repo_root,
            optimizer_steps=optimizer_steps,
            max_wall_seconds=max_wall_seconds,
            minimum_commit_headroom_mib=minimum_commit_headroom_mib,
            gpu_temperature_abort_c=gpu_temperature_abort_c,
            step_time_degradation_ratio=step_time_degradation_ratio,
            consecutive_degraded_steps=consecutive_degraded_steps,
            training_stack_loader=training_stack_loader,
            telemetry_path=telemetry_path,
        )
